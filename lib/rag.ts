import { Document } from "@langchain/core/documents";
import { ChatGroq } from "@langchain/groq";
import { QdrantClient } from "@qdrant/js-client-rest";
import { BaseMessage, HumanMessage, SystemMessage } from "@langchain/core/messages";
import { randomUUID } from "crypto";
import { extractText } from "unpdf";

// Constants

const DEFAULT_PREFIX = "sourcesage";
const DENSE_VECTOR_NAME = "dense";
const SPARSE_VECTOR_NAME = "sparse";

const EMBEDDING_MODEL = "embed-english-v3.0";
const EMBEDDING_DIM = 1024; // embed-english-v3.0 output dimension
const EMBED_BATCH_SIZE = 96; // Cohere embed accepts at most 96 texts per request
const RERANK_MODEL = "rerank-v3.5";
const DEFAULT_CHAT_MODEL = "openai/gpt-oss-120b";

// Sentence-window chunking config
const SENTENCES_PER_CHILD = 2; // small chunk indexed in Qdrant
const CONTEXT_SENTENCES = 2; // sentences added on each side to build the parent sent to the LLM
const MAX_SENTENCE_CHARS = 500; // longer "sentences" (tables, unpunctuated text) get split on words

// BM25 parameters (standard defaults)
const BM25_K1 = 1.2;
const BM25_B = 0.75;

// Retrieval config
const PREFETCH_LIMIT = 20; // candidates per search path (dense/sparse x each query)
const RERANK_CANDIDATES = 20; // fused candidates sent to the reranker

export const NOT_FOUND_ANSWER = "I could not find that in the uploaded document.";

export type SourceChunk = {
  chunk: number;
  page?: number;
  text: string;
  rerankScore?: number;
};

export type Grounding = {
  verdict: "grounded" | "not_grounded" | "unverified" | "not_found";
  confidence: number;
  reasoning: string;
  retried: boolean;
};

export type AnswerResult = {
  answer: string;
  sources: SourceChunk[];
  grounding: Grounding;
  queries: string[];
  timingsMs: Record<string, number>;
};

type SparseVector = {
  indices: number[];
  values: number[];
};

function requireEnv(name: string): string {
  const value = process.env[name];
  if (!value) {
    throw new Error(
      `${name} is not configured. Add it to your environment variables.`
    );
  }
  return value;
}

function getCollectionPrefix(): string {
  return process.env.QDRANT_COLLECTION_PREFIX || DEFAULT_PREFIX;
}

export function createCollectionName(): string {
  return `${getCollectionPrefix()}_${randomUUID().replace(/-/g, "_")}`;
}

function messageText(content: BaseMessage["content"]): string {
  return typeof content === "string"
    ? content
    : content.map((part: any) => ("text" in part ? part.text : "")).join("");
}

// Cohere (embeddings + rerank)

async function cohereRequest(path: string, body: unknown): Promise<any> {
  const response = await fetch(`https://api.cohere.com/v2/${path}`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${requireEnv("COHERE_API_KEY")}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw new Error(`Cohere ${path} failed (${response.status}): ${await response.text()}`);
  }

  return response.json();
}

// Documents and queries use different input_type values so Cohere embeds them
// into the asymmetric spaces it was trained for.
async function embed(
  texts: string[],
  inputType: "search_document" | "search_query"
): Promise<number[][]> {
  const vectors: number[][] = [];

  for (let i = 0; i < texts.length; i += EMBED_BATCH_SIZE) {
    const data = await cohereRequest("embed", {
      texts: texts.slice(i, i + EMBED_BATCH_SIZE),
      model: EMBEDDING_MODEL,
      input_type: inputType,
      embedding_types: ["float"],
    });
    vectors.push(...data.embeddings.float);
  }

  return vectors;
}

export function getChatModel(): ChatGroq {
  requireEnv("GROQ_API_KEY");
  return new ChatGroq({
    model: process.env.GROQ_MODEL || DEFAULT_CHAT_MODEL,
    temperature: 0.1,
    // gpt-oss is a reasoning model: its hidden reasoning tokens count against this limit
    maxTokens: 4096,
  });
}

function getQdrantClient(): QdrantClient {
  return new QdrantClient({
    url: requireEnv("QDRANT_URL"),
    apiKey: process.env.QDRANT_API_KEY || undefined,
    checkCompatibility: false,
  });
}

// Parsing

async function parsePdf(buffer: ArrayBuffer): Promise<Document[]> {
  const { text, totalPages } = await extractText(
    buffer,
    { mergePages: false }
  );
  const pages = Array.isArray(text) ? text : [text];

  return pages
    .map((pageText, index) =>
      new Document({
        pageContent: pageText.replace(/\s+/g, " ").trim(),
        metadata: {
          type: "pdf",
          source: "",
          loc: { pageNumber: index + 1 },
          totalPages,
        },
      })
    )
    .filter((doc) => doc.pageContent.length > 20);
}

export async function parseUploadedFile(file: File): Promise<Document[]> {
  const fileName = file.name || "uploaded-document";
  const mimeType = file.type;

  if (mimeType === "text/plain" || fileName.toLowerCase().endsWith(".txt")) {
    const text = await file.text();
    return [
      new Document({
        pageContent: text,
        metadata: { source: fileName, type: "text" },
      }),
    ];
  }

  if (
    mimeType === "application/pdf" ||
    fileName.toLowerCase().endsWith(".pdf")
  ) {
    const buffer = await file.arrayBuffer();
    const pages = await parsePdf(buffer);

    if (pages.length === 0) {
      throw new Error(
        "No readable text found in the PDF. It may be a scanned image-only PDF."
      );
    }

    return pages.map(
      (page) =>
        new Document({
          pageContent: page.pageContent,
          metadata: { ...page.metadata, source: fileName },
        })
    );
  }

  throw new Error("Unsupported file type. Upload a PDF or .txt file.");
}

// Chunking

// Splits text without sentence punctuation (tables, lists, headings) into
// word-bounded pieces so one "sentence" can't swallow a whole page.
function splitLongSentence(sentence: string): string[] {
  if (sentence.length <= MAX_SENTENCE_CHARS) return [sentence];

  const pieces: string[] = [];
  let current = "";
  for (const word of sentence.split(" ")) {
    if (current && current.length + word.length + 1 > MAX_SENTENCE_CHARS) {
      pieces.push(current);
      current = word;
    } else {
      current = current ? `${current} ${word}` : word;
    }
  }
  if (current) pieces.push(current);
  return pieces;
}

function splitIntoSentences(text: string): string[] {
  // Split on sentence-ending punctuation followed by whitespace + capital letter
  return text
    .replace(/\s+/g, " ")
    .split(/(?<=[.!?])\s+(?=[A-Z"'])/)
    .map((s) => s.trim())
    .filter((s) => s.length > 10)
    .flatMap(splitLongSentence);
}

export async function chunkDocuments(
  documents: Document[]
): Promise<Document[]> {
  const allChildren: Document[] = [];
  let globalChunkIndex = 1;

  for (const doc of documents) {
    const sentences = splitIntoSentences(doc.pageContent);
    const totalSentences = sentences.length;

    for (let i = 0; i < totalSentences; i += SENTENCES_PER_CHILD) {
      // Child: small, precise chunk that gets embedded and searched
      const childText = sentences.slice(i, i + SENTENCES_PER_CHILD).join(" ");

      // Parent: the child plus surrounding sentences, sent to the LLM
      const windowStart = Math.max(0, i - CONTEXT_SENTENCES);
      const windowEnd = Math.min(
        totalSentences,
        i + SENTENCES_PER_CHILD + CONTEXT_SENTENCES
      );
      const parentText = sentences.slice(windowStart, windowEnd).join(" ");

      allChildren.push(
        new Document({
          pageContent: childText,
          metadata: {
            ...doc.metadata,
            chunk: globalChunkIndex,
            chunking: `sentence-window-child-${SENTENCES_PER_CHILD}-context-${CONTEXT_SENTENCES}`,
            parentText,
            sentenceStart: i,
            sentenceEnd: i + SENTENCES_PER_CHILD,
          },
        })
      );

      globalChunkIndex++;
    }
  }

  return allChildren.filter((c) => c.pageContent.trim().length > 0);
}

// BM25 sparse vectors
//
// BM25 = IDF(term) * TF-saturation(term, doc). The TF part only needs the
// chunk and the average chunk length, so it is computed here at index time.
// The IDF part needs corpus statistics, so the collection is created with
// modifier "idf" and Qdrant applies IDF at query time from its own counts.

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((t) => t.length > 1);
}

function hashToken(term: string): number {
  let hash = 0;
  for (let i = 0; i < term.length; i++) {
    hash = (hash * 31 + term.charCodeAt(i)) & 0xfffff;
  }
  return hash;
}

// Counts per hashed index; summing by index means two words that hash to the
// same slot never produce duplicate indices (which Qdrant rejects).
function hashedTermCounts(tokens: string[]): Map<number, number> {
  const counts = new Map<number, number>();
  for (const token of tokens) {
    const index = hashToken(token);
    counts.set(index, (counts.get(index) || 0) + 1);
  }
  return counts;
}

function buildDocumentSparseVector(text: string, avgDocLength: number): SparseVector {
  const tokens = tokenize(text);
  const lengthNorm = 1 - BM25_B + BM25_B * (tokens.length / avgDocLength);
  const indices: number[] = [];
  const values: number[] = [];

  for (const [index, tf] of hashedTermCounts(tokens)) {
    indices.push(index);
    values.push((tf * (BM25_K1 + 1)) / (tf + BM25_K1 * lengthNorm));
  }

  return { indices, values };
}

function buildQuerySparseVector(text: string): SparseVector {
  const indices = Array.from(hashedTermCounts(tokenize(text)).keys());
  return { indices, values: indices.map(() => 1) };
}

// Indexing

export async function indexChunks(
  collectionName: string,
  chunks: Document[]
): Promise<void> {
  if (chunks.length === 0) {
    throw new Error("No readable text was found in the uploaded document.");
  }

  const client = getQdrantClient();

  await client.createCollection(collectionName, {
    vectors: {
      [DENSE_VECTOR_NAME]: {
        size: EMBEDDING_DIM,
        distance: "Cosine",
      },
    },
    sparse_vectors: {
      [SPARSE_VECTOR_NAME]: { modifier: "idf" },
    },
  });

  const texts = chunks.map((c) => c.pageContent);
  const denseVectors = await embed(texts, "search_document");

  if (denseVectors.length !== chunks.length || denseVectors.some((v) => !v?.length)) {
    throw new Error(
      `Embedding failed: got ${denseVectors.length} vectors for ${chunks.length} chunks. ` +
      `Check that COHERE_API_KEY is valid.`
    );
  }

  const avgDocLength =
    texts.reduce((sum, text) => sum + tokenize(text).length, 0) / texts.length || 1;

  const points = chunks.map((chunk, i) => ({
    id: randomUUID(),
    vector: {
      [DENSE_VECTOR_NAME]: denseVectors[i],
      [SPARSE_VECTOR_NAME]: buildDocumentSparseVector(chunk.pageContent, avgDocLength),
    },
    payload: {
      pageContent: chunk.pageContent,
      metadata: chunk.metadata,
    },
  }));

  const BATCH = 100;
  for (let i = 0; i < points.length; i += BATCH) {
    await client.upsert(collectionName, {
      wait: true,
      points: points.slice(i, i + BATCH),
    });
  }
}

// Only deletes collections this app created, so a client can't pass an
// arbitrary name and drop someone else's data.
export async function deleteCollection(collectionName: string): Promise<void> {
  if (!collectionName.startsWith(`${getCollectionPrefix()}_`)) return;
  try {
    await getQdrantClient().deleteCollection(collectionName);
  } catch (error) {
    console.warn(`Could not delete collection ${collectionName}:`, error);
  }
}

// Query translation

// Multi-query generation: LLM rewrites user question into 3 different phrasings
export async function generateQueryVariants(
  question: string,
  llm: ChatGroq
): Promise<string[]> {
  const prompt = `Given a user question, generate 3 different ways to rephrase it for better retrieval.
Each variation should focus on a different angle or terminology while preserving the core intent.

Question: "${question}"

Output exactly 3 rephrased questions, one per line, numbered 1., 2., 3.

Do not include any other text or explanations. Format:
1. <variation 1>
2. <variation 2>
3. <variation 3>`;

  try {
    const response = await llm.invoke([new HumanMessage(prompt)]);
    return messageText(response.content)
      .split("\n")
      .map((line) => line.match(/^\s*[1-3][.)]\s*(.+)$/)?.[1]?.trim())
      .filter((line): line is string => Boolean(line))
      .slice(0, 3);
  } catch (error) {
    console.error("Error generating query variants:", error);
    return [];
  }
}

// Reranking

// Cohere Rerank is a cross-encoder: it reads the question and each chunk
// together, which scores relevance far better than comparing two embeddings.
export async function rerank(
  question: string,
  docs: Document[],
  topK: number = 5
): Promise<Document[]> {
  if (docs.length === 0) return docs;

  try {
    const data = await cohereRequest("rerank", {
      model: RERANK_MODEL,
      query: question,
      documents: docs.map((doc) => doc.pageContent),
      top_n: topK,
    });

    return data.results.map(
      (result: { index: number; relevance_score: number }) =>
        new Document({
          pageContent: docs[result.index].pageContent,
          metadata: {
            ...docs[result.index].metadata,
            rerankScore: result.relevance_score,
          },
        })
    );
  } catch (error) {
    // Fusion order is still a reasonable ranking if the reranker is down
    console.error("Rerank failed, falling back to fusion order:", error);
    return docs.slice(0, topK);
  }
}

// Retrieval
//
// Every query (original + rewrites) is searched both ways (dense and BM25),
// and Qdrant merges all those ranked lists with Reciprocal Rank Fusion in a
// single request.

export async function hybridSearch(
  collectionName: string,
  queries: string[],
  limit = RERANK_CANDIDATES
): Promise<Document[]> {
  const queryVectors = await embed(queries, "search_query");

  const prefetch = queries.flatMap((query, i) => {
    const sparse = buildQuerySparseVector(query);
    return [
      { query: queryVectors[i], using: DENSE_VECTOR_NAME, limit: PREFETCH_LIMIT },
      ...(sparse.indices.length
        ? [{ query: sparse, using: SPARSE_VECTOR_NAME, limit: PREFETCH_LIMIT }]
        : []),
    ];
  });

  const { points } = await getQdrantClient().query(collectionName, {
    prefetch,
    query: { fusion: "rrf" },
    limit,
    with_payload: true,
  });

  return points.map((point) => {
    const payload = point.payload as { pageContent: string; metadata: any };
    const meta = payload.metadata || {};
    return new Document({
      // Search matched the small child; the LLM gets the wider parent window
      pageContent: meta.parentText || payload.pageContent,
      metadata: {
        ...meta,
        childText: payload.pageContent,
        fusionScore: point.score,
      },
    });
  });
}

export async function retrieveChunks(
  collectionName: string,
  question: string,
  topK = 5,
  timingsMs: Record<string, number> = {}
): Promise<{ chunks: Document[]; queries: string[] }> {
  const llm = getChatModel();
  const time = async <T>(stage: string, fn: () => Promise<T>): Promise<T> => {
    const start = Date.now();
    try {
      return await fn();
    } finally {
      timingsMs[stage] = Date.now() - start;
    }
  };

  // Step 1: Query translation. The original question is always searched too,
  // so a bad rewrite can only add candidates, never lose the obvious ones.
  const variants = await time("queryRewrite", () => generateQueryVariants(question, llm));
  const queries = Array.from(new Set([question, ...variants]));

  // Step 2: Hybrid search + RRF fusion
  const candidates = await time("hybridSearch", () => hybridSearch(collectionName, queries));

  // Step 3: Cross-encoder reranking
  const reranked = await time("rerank", () => rerank(question, candidates, topK));

  // Step 4: Relevance grading (CRAG). Empty result means "not in the document".
  const relevant = await time("gradeChunks", () => gradeChunks(question, reranked, llm));
  if (relevant.length === 0) return { chunks: [], queries };

  // Step 5: Context compression
  const compressed = await time("compress", () => compressContext(question, relevant, llm));
  return { chunks: compressed, queries };
}

// Post-retrieval

export type ChunkGrade = { index: number; relevant: boolean; confidence: number };

// LLM scores every chunk in one call. Chunks missing from the reply (bad JSON)
// are marked relevant so a grader failure never hides real context.
export async function scoreChunks(
  question: string,
  docs: Document[],
  llm?: ChatGroq
): Promise<ChunkGrade[]> {
  if (docs.length === 0) return [];
  const model = llm || getChatModel();

  let prompt = `You are a relevance grader for a RAG system.
Given a question and a document chunk, determine if the chunk is relevant to answering the question.

Question: "${question}"

For each chunk below, output "RELEVANT" or "NOT_RELEVANT" followed by a confidence score (0.0-1.0).

Format your response as JSON:
[
  {"index": 0, "label": "RELEVANT|NOT_RELEVANT", "confidence": 0.95},
  {"index": 1, "label": "RELEVANT|NOT_RELEVANT", "confidence": 0.72},
  ...
]

Document chunks:
`;

  docs.forEach((doc, i) => {
    prompt += `\n\n--- Chunk ${i} ---\n${doc.pageContent}\n--- End Chunk ${i} ---\n`;
  });

  prompt += `\n\nOutput JSON:`;

  let scores: Array<{ index: number; label: string; confidence: number }> = [];
  try {
    const response = await model.invoke([new HumanMessage(prompt)]);
    const jsonMatch = messageText(response.content).match(/\[\s*\{[\s\S]*\}\s*\]/);
    if (jsonMatch) scores = JSON.parse(jsonMatch[0]);
  } catch (error) {
    console.error("Error grading chunks:", error);
  }

  return docs.map((_, i) => {
    const score = scores.find((s) => s.index === i);
    if (!score) return { index: i, relevant: true, confidence: 0 };
    return {
      index: i,
      relevant: score.label === "RELEVANT" && score.confidence >= 0.5,
      confidence: score.confidence,
    };
  });
}

export async function gradeChunks(
  question: string,
  docs: Document[],
  llm?: ChatGroq
): Promise<Document[]> {
  const grades = await scoreChunks(question, docs, llm);
  return docs.filter((_, i) => grades[i].relevant);
}

const NO_RELEVANT_CONTENT = "No relevant content found.";

// Extracts only the question-relevant sentences from each chunk (in parallel)
export async function compressContext(
  question: string,
  docs: Document[],
  llm?: ChatGroq
): Promise<Document[]> {
  const model = llm || getChatModel();

  const prompt = `You are a context compressor for a RAG system.
Given a question and a document chunk, extract only the sentences that are relevant to answering the question.

Question: "${question}"

Copy the relevant sentences exactly as written, combined into a single paragraph.
If no sentences are relevant, output "${NO_RELEVANT_CONTENT}"

Chunk to compress:
`;

  const compressed = await Promise.all(
    docs.map(async (doc) => {
      try {
        const response = await model.invoke([
          new HumanMessage(
            prompt +
              `\n\n--- CHUNK START ---\n${doc.pageContent}\n--- CHUNK END ---\n\nOutput compressed context:`
          ),
        ]);
        const content = messageText(response.content).trim();
        if (!content || content.includes(NO_RELEVANT_CONTENT)) return null;

        return new Document({
          pageContent: content,
          metadata: {
            ...doc.metadata,
            compressed: true,
            originalLength: doc.pageContent.length,
          },
        });
      } catch (error) {
        console.error("Error compressing chunk:", error);
        return doc; // Keep original if compression fails
      }
    })
  );

  const kept = compressed.filter((doc): doc is Document => doc !== null);
  // If the compressor dropped everything, the grader already judged these
  // chunks relevant, so fall back to the uncompressed text.
  return kept.length > 0 ? kept : docs;
}

// Hallucination grader - check if answer is grounded in chunks
export async function gradeAnswer(
  answer: string,
  docs: Document[],
  question: string,
  llm?: ChatGroq
): Promise<Omit<Grounding, "retried">> {
  const model = llm || getChatModel();

  let prompt = `You are a hallucination grader for a RAG system.
Given a question, an answer, and the source chunks used to generate the answer, determine if the answer is fully grounded in the provided sources.

Question: "${question}"

Answer to evaluate: "${answer}"

Source chunks:
`;

  docs.forEach((doc, i) => {
    prompt += `\n\n--- Source ${i + 1} ---\n${doc.pageContent}\n--- End Source ${i + 1} ---\n`;
  });

  prompt += `\n\nIs the answer fully grounded in the sources? Answer with JSON:
{"grounded": true|false, "confidence": 0.0-1.0, "reasoning": "brief explanation"}`;

  try {
    const response = await model.invoke([new HumanMessage(prompt)]);
    const jsonMatch = messageText(response.content).match(/\{[\s\S]*\}/);
    if (jsonMatch) {
      const parsed = JSON.parse(jsonMatch[0]);
      return {
        verdict: parsed.grounded ? "grounded" : "not_grounded",
        confidence: Number(parsed.confidence) || 0,
        reasoning: String(parsed.reasoning || ""),
      };
    }
  } catch (error) {
    console.error("Error grading answer:", error);
  }

  return { verdict: "unverified", confidence: 0, reasoning: "Grader output could not be parsed" };
}

// Answer generation

export function formatSources(chunks: Document[]): SourceChunk[] {
  return chunks.map((chunk, index) => ({
    chunk: Number(chunk.metadata.chunk || index + 1),
    page:
      Number(
        chunk.metadata.loc?.pageNumber ||
          chunk.metadata.pdf?.pageNumber ||
          0
      ) || undefined,
    // Show child text in UI (concise), not the expanded parent
    text: (chunk.metadata.childText || chunk.pageContent)
      .replace(/\s+/g, " ")
      .slice(0, 420),
    rerankScore: chunk.metadata.rerankScore,
  }));
}

export function buildGroundedPrompt(chunks: Document[], feedback?: string): string {
  const context = chunks
    .map((chunk, index) => {
      const chunkNumber = chunk.metadata.chunk || index + 1;
      const page =
        chunk.metadata.loc?.pageNumber || chunk.metadata.pdf?.pageNumber;
      return `SOURCE ${index + 1} | chunk ${chunkNumber}${page ? ` | page ${page}` : ""}\n${chunk.pageContent}`;
    })
    .join("\n\n---\n\n");

  const retryNote = feedback
    ? `\nA previous draft was rejected because it contained claims the sources do not support: ${feedback}
State only facts that appear explicitly in the sources.\n`
    : "";

  return `You are a document question-answering assistant.

Answer using only the sources below. If the sources do not contain the answer, say: "${NOT_FOUND_ANSWER}"

Rules:
- Do not use outside knowledge.
- Be concise but complete.
- Cite the sources you used inline as [1], [2], using the SOURCE numbers below.
- Never invent facts, names, page numbers, or examples not present in the sources.
${retryNote}
Sources:
${context}`;
}

export async function generateGroundedAnswer(
  chunks: Document[],
  question: string,
  feedback?: string
): Promise<string> {
  const model = getChatModel();
  const response = await model.invoke([
    new SystemMessage(buildGroundedPrompt(chunks, feedback)),
    new HumanMessage(question),
  ]);

  return messageText(response.content).trim() || NOT_FOUND_ANSWER;
}

// Full query pipeline: retrieve -> generate -> check grounding -> retry once
export async function answerQuestion(
  collectionName: string,
  question: string
): Promise<AnswerResult> {
  const timingsMs: Record<string, number> = {};
  const started = Date.now();

  const { chunks, queries } = await retrieveChunks(collectionName, question, 5, timingsMs);

  if (chunks.length === 0) {
    timingsMs.total = Date.now() - started;
    return {
      answer: NOT_FOUND_ANSWER,
      sources: [],
      grounding: {
        verdict: "not_found",
        confidence: 1,
        reasoning: "No retrieved chunk was relevant to the question.",
        retried: false,
      },
      queries,
      timingsMs,
    };
  }

  let start = Date.now();
  let answer = await generateGroundedAnswer(chunks, question);
  timingsMs.generate = Date.now() - start;

  if (answer === NOT_FOUND_ANSWER) {
    timingsMs.total = Date.now() - started;
    return {
      answer,
      sources: formatSources(chunks),
      grounding: { verdict: "not_found", confidence: 1, reasoning: "The model found no answer in the sources.", retried: false },
      queries,
      timingsMs,
    };
  }

  start = Date.now();
  let grounding = await gradeAnswer(answer, chunks, question);
  let retried = false;

  // One retry with the grader's feedback. More retries rarely help and double latency.
  if (grounding.verdict === "not_grounded") {
    retried = true;
    answer = await generateGroundedAnswer(chunks, question, grounding.reasoning);
    grounding = await gradeAnswer(answer, chunks, question);
  }
  timingsMs.groundingCheck = Date.now() - start;
  timingsMs.total = Date.now() - started;

  return {
    answer,
    sources: formatSources(chunks),
    grounding: { ...grounding, retried },
    queries,
    timingsMs,
  };
}
