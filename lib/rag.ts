import { Document } from "@langchain/core/documents";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { ChatGroq } from "@langchain/groq";
import { QdrantClient } from "@qdrant/js-client-rest";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { randomUUID } from "crypto";
import { extractText } from "unpdf";
import {
  AutoTokenizer,
  pipeline,
  env,
} from "@xenova/transformers";

// Constants 

const DEFAULT_PREFIX = "notebook";
const DENSE_VECTOR_NAME = "dense";
const SPARSE_VECTOR_NAME = "sparse";
const EMBEDDING_DIM = 768; // text-embedding-004 output dimension

// Sentence-window chunking config
const SENTENCES_PER_CHILD = 2;   // small chunk indexed in Qdrant
const WINDOW_SIZE = 5;            // parent = child + surrounding sentences sent to LLM


export type SourceChunk = {
  chunk: number;
  page?: number;
  text: string;
};

type SparseVector = {
  indices: number[];
  values: number[];
};

type ParentChildChunk = {
  child: Document;   // small — goes into Qdrant
  parent: Document;  // large — sent to LLM at generation time
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


export function createCollectionName(): string {
  const prefix = process.env.QDRANT_COLLECTION_PREFIX || DEFAULT_PREFIX;
  return `${prefix}_${randomUUID().replace(/-/g, "_")}`;
}


export function getEmbeddings(): GoogleGenerativeAIEmbeddings {
  requireEnv("GOOGLE_API_KEY");
  return new GoogleGenerativeAIEmbeddings({
    model: "text-embedding-004",
  });
}

export function getChatModel(): ChatGroq {
  requireEnv("GROQ_API_KEY");
  return new ChatGroq({
    model: "llama-3.3-70b-versatile",
    temperature: 0.1,
    maxTokens: 1200,
  });
}


function getQdrantClient(): QdrantClient {
  return new QdrantClient({
    url: requireEnv("QDRANT_URL"),
    apiKey: process.env.QDRANT_API_KEY,
  });
}


async function parsePdf(buffer: ArrayBuffer): Promise<Document[]> {
  const { text, totalPages } = await extractText(
    new Uint8Array(buffer),
    { mergePages: false }  // get per-page text as array
  );

  // extractText with mergePages:false returns text as string[] (one per page)
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


function splitIntoSentences(text: string): string[] {
  // Split on sentence-ending punctuation followed by whitespace + capital letter
  // Handles: "Dr. Smith went..." edge cases reasonably well
  return text
    .split(/(?<=[.!?])\s+(?=[A-Z"'])/)
    .map((s) => s.trim())
    .filter((s) => s.length > 10);
}

// Semantic chunking: sentence-window + parent-child 
//
// For each document (page):
//   1. Split into individual sentences
//   2. Group every SENTENCES_PER_CHILD sentences → child chunk (indexed)
//   3. Build parent = child ± WINDOW_SIZE/2 surrounding sentences (context window)
//   4. Store parent text in child metadata so retrieval can return it

export async function chunkDocuments(
  documents: Document[]
): Promise<Document[]> {
  const allChildren: Document[] = [];
  let globalChunkIndex = 1;

  for (const doc of documents) {
    const sentences = splitIntoSentences(doc.pageContent);
    if (sentences.length === 0) continue;

    const totalSentences = sentences.length;

    // Slide through sentences, stepping by SENTENCES_PER_CHILD
    for (
      let i = 0;
      i < totalSentences;
      i += SENTENCES_PER_CHILD
    ) {
      // Child: small, precise chunk
      const childSentences = sentences.slice(i, i + SENTENCES_PER_CHILD);
      const childText = childSentences.join(" ");

      // Parent: wider window around the child for richer LLM context
      const windowStart = Math.max(0, i - Math.floor(WINDOW_SIZE / 2));
      const windowEnd = Math.min(
        totalSentences,
        i + SENTENCES_PER_CHILD + Math.floor(WINDOW_SIZE / 2)
      );
      const parentText = sentences.slice(windowStart, windowEnd).join(" ");

      allChildren.push(
        new Document({
          pageContent: childText,
          metadata: {
            ...doc.metadata,
            chunk: globalChunkIndex,
            chunking: `sentence-window-child-${SENTENCES_PER_CHILD}-window-${WINDOW_SIZE}`,
            parentText,           // attached here; sent to LLM at generation time
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

// BM25 sparse vector 
//
// True BM25 requires a corpus-level IDF table. Here we use a lightweight
// TF-only approximation sufficient for hybrid search quality in Phase 1.
// Phase 3 can swap this for a proper BM25 library if needed.

function buildSparseVector(text: string): SparseVector {
  const tokens = text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, "")
    .split(/\s+/)
    .filter((t) => t.length > 2);

  const termFreq: Record<string, number> = {};
  for (const token of tokens) {
    termFreq[token] = (termFreq[token] || 0) + 1;
  }

  const indices: number[] = [];
  const values: number[] = [];
  const totalTokens = tokens.length || 1;

  for (const [term, freq] of Object.entries(termFreq)) {
    // Stable hash: sum of (charCode * position) mod 2^20
    let hash = 0;
    for (let i = 0; i < term.length; i++) {
      hash = (hash * 31 + term.charCodeAt(i)) & 0xfffff;
    }
    indices.push(hash);
    values.push(freq / totalTokens); // normalized TF
  }

  return { indices, values };
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
  const embeddings = getEmbeddings();

  // Create collection with both dense and sparse vector configs
  await client.createCollection(collectionName, {
    vectors: {
      [DENSE_VECTOR_NAME]: {
        size: EMBEDDING_DIM,
        distance: "Cosine",
      },
    },
    sparse_vectors: {
      [SPARSE_VECTOR_NAME]: {},
    },
  });

  // Embed all child texts (dense)
  const texts = chunks.map((c) => c.pageContent);
  const denseVectors = await embeddings.embedDocuments(texts);

  // Build points with both dense + sparse vectors
  const points = chunks.map((chunk, i) => ({
    id: randomUUID(),
    vector: {
      [DENSE_VECTOR_NAME]: denseVectors[i],
      [SPARSE_VECTOR_NAME]: buildSparseVector(chunk.pageContent),
    },
    payload: {
      pageContent: chunk.pageContent,
      metadata: chunk.metadata,
    },
  }));

  // Upsert in batches of 100
  const BATCH = 100;
  for (let i = 0; i < points.length; i += BATCH) {
    await client.upsert(collectionName, {
      wait: true,
      points: points.slice(i, i + BATCH),
    });
  }
}

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
    const content =
      typeof response.content === "string"
        ? response.content
        : response.content
            .map((part) => ("text" in part ? part.text : ""))
            .join("");

    // Parse the numbered variations
    const lines = content.split("\n").filter((line) => line.trim());
    const variations: string[] = [];

    for (const line of lines) {
      // Match "1. text", "1) text", or just "text"
      const match = line.match(/^[1-3][.)]\s*(.+)$/i);
      if (match) {
        variations.push(match[1].trim());
      } else if (line.trim().length > 5) {
        // Fallback: if no number prefix, include as variation if seems valid
        variations.push(line.trim());
      }
    }

    // Ensure we have at least 3 variations
    while (variations.length < 3) {
      variations.push(question); // fallback to original if missing
    }

    return variations.slice(0, 3);
  } catch (error) {
    console.error("Error generating query variants:", error);
    // Fallback to original question repeated 3 times
    return [question, question, question];
  }
}

// CrossEncoder re-ranking
let reranker: any = null;
let rerankerTokenizer: any = null;

async function ensureRerankerLoaded() {
  if (reranker === null) {
    // Disable remote model downloads and use local only
    env.allowRemoteModels = false;

    try {
      rerankerTokenizer = await AutoTokenizer.from_pretrained(
        "cross-encoder/st5-base"
      );
      reranker = await pipeline(
        "text-classification", // binary relevance: [irrelevant, relevant]
        "cross-encoder/st5-base",
        {
          progress_callback: () => {},
        }
      );
    } catch (err) {
      console.warn(
        "CrossEncoder model not found locally. Using fallback reranker (cosine scoring)."
      );
      reranker = null;
    }
  }
}

export async function rerank(
  question: string,
  docs: Document[],
  topK: number = 5
): Promise<Document[]> {
  if (docs.length === 0) return docs;

  await ensureRerankerLoaded();

  // If CrossEncoder not available, return topK via simple scoring fallback
  if (reranker === null) {
    // Use simple scoring fallback: keep topK by original score order
    // In real implementation, this could use BM25 scoring from metadata
    return docs.slice(0, topK);
  }

  // Prepare query + document pairs for CrossEncoder
  const pairs = docs.map((doc) => ({
    query: question,
    document: doc.pageContent,
    doc,
  }));

  // Run CrossEncoder scoring (batched)
  try {
    const scores = await reranker(
      pairs.map((p) => [p.query, p.document]),
      { truncation: true }
    );

    // Sort by score (higher = more relevant)
    const scoredDocs = pairs.map((p, idx) => ({
      doc: p.doc,
      score: Array.isArray(scores[idx]) ? scores[idx][1] : scores[idx],
    }));

    scoredDocs.sort((a, b) => b.score - a.score);

    return scoredDocs.slice(0, topK).map((item) => item.doc);
  } catch (error) {
    console.error("CrossEncoder reranking failed, falling back:", error);
    return docs.slice(0, topK);
  }
}
// Retrieval (hybrid: dense + sparse → RRF merge)

// RRF fusion supporting N result sets (for multi-query RAG fusion)
function reciprocalRankFusion(
  ...resultSets: Array<{ id: string; score: number; payload: any }[]>
): Array<{ id: string; payload: any; rrfScore: number }> {
  const scores: Record<string, { payload: any; score: number }> = {};
  const k = 60;

  const addRank = (
    hits: Array<{ id: string; score: number; payload: any }>,
    weight: number = 1.0
  ) => {
    hits.forEach((hit, rank) => {
      const id = String(hit.id);
      if (!scores[id]) scores[id] = { payload: hit.payload, score: 0 };
      scores[id].score += weight / (k + rank + 1);
    });
  };

  // Add each result set with equal weight
  resultSets.forEach((set) => addRank(set, 1.0));

  return Object.entries(scores)
    .map(([id, { payload, score }]) => ({ id, payload, rrfScore: score }))
    .sort((a, b) => b.rrfScore - a.rrfScore);
}

export async function retrieveChunks(
  collectionName: string,
  question: string,
  topK = 5
): Promise<Document[]> {
  const client = getQdrantClient();
  const embeddings = getEmbeddings();
  const llm = getChatModel();

  // Step 1: Generate query variants
  const variants = await generateQueryVariants(question, llm);
  console.log("Query variants:", variants);

  // Step 2: For each variant, run hybrid retrieval (dense + sparse)
  const variantResults: Array<{ dense: { id: string; score: number; payload: any }[]; sparse: { id: string; score: number; payload: any }[] }> =
    [];

  for (const variant of variants) {
    const FETCH_K = topK * 4;

    // Dense search
    const queryVector = await embeddings.embedQuery(variant);
    const denseResults = await client.search(collectionName, {
      vector: { name: DENSE_VECTOR_NAME, vector: queryVector },
      limit: FETCH_K,
      with_payload: true,
    });

    // Sparse search
    const sparseQuery = buildSparseVector(variant);
    const sparseResults = await client.search(collectionName, {
      vector: {
        name: SPARSE_VECTOR_NAME,
        vector: sparseQuery as any,
      },
      limit: FETCH_K,
      with_payload: true,
    });

    variantResults.push({
      dense: denseResults.map((r) => ({
        id: String(r.id),
        score: r.score,
        payload: r.payload,
      })),
      sparse: sparseResults.map((r) => ({
        id: String(r.id),
        score: r.score,
        payload: r.payload,
      })),
    });
  }

  // Step 3: RRF fusion across all result sets (dense from all variants + sparse from all variants)
  const allDenseHits: Array<{ id: string; score: number; payload: any }[]> = [];
  const allSparseHits: Array<{ id: string; score: number; payload: any }[]> = [];

  for (const result of variantResults) {
    allDenseHits.push(result.dense);
    allSparseHits.push(result.sparse);
  }

  const merged = reciprocalRankFusion(
    ...allDenseHits,
    ...allSparseHits
  );

  // Step 4: Deduplicate by point ID (merge docs with same ID)
  const uniqueDocs = new Map<string, { payload: any; combinedScore: number }>();
  const docScores: Record<string, number> = {};

  for (const entry of merged) {
    const existing = uniqueDocs.get(entry.id);
    if (existing) {
      // Combine scores for same ID
      uniqueDocs.set(entry.id, {
        payload: entry.payload,
        combinedScore: existing.combinedScore + entry.rrfScore,
      });
    } else {
      uniqueDocs.set(entry.id, {
        payload: entry.payload,
        combinedScore: entry.rrfScore,
      });
    }
    docScores[entry.id] = entry.rrfScore;
  }

  // Step 5: Convert to Documents and apply CrossEncoder re-ranking
  const candidates = Array.from(uniqueDocs.values()).slice(0, 20); // Take top 20 for reranking

  const docsToRerank = candidates.map(({ payload }) => {
    const meta = (payload as any).metadata || {};
    const childText = (payload as any).pageContent || "";
    const parentText: string = meta.parentText || childText;

    return new Document({
      pageContent: parentText,
      metadata: {
        ...meta,
        childText,
        rrfScore: docScores[payload.id],
      },
    });
  });

  // Step 6: CrossEncoder re-ranking
  const rerankedDocs = await rerank(question, docsToRerank, topK);

  return rerankedDocs;
}

//  Answer generation

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
  }));
}

export function buildGroundedPrompt(chunks: Document[]): string {
  const context = chunks
    .map((chunk, index) => {
      const chunkNumber = chunk.metadata.chunk || index + 1;
      const page =
        chunk.metadata.loc?.pageNumber || chunk.metadata.pdf?.pageNumber;
      return `SOURCE ${index + 1} | chunk ${chunkNumber}${page ? ` | page ${page}` : ""}\n${chunk.pageContent}`;
    })
    .join("\n\n---\n\n");

  return `You are a document question-answering assistant.

Answer using only the sources below. If the sources do not contain the answer, say: "I could not find that in the uploaded document."

Rules:
- Do not use outside knowledge.
- Be concise but complete.
- Mention source chunk numbers naturally when useful.
- Never invent facts, names, page numbers, or examples not present in the sources.

Sources:
${context}`;
}

export async function generateGroundedAnswer(
  chunks: Document[],
  question: string
): Promise<string> {
  const model = getChatModel();
  const response = await model.invoke([
    new SystemMessage(buildGroundedPrompt(chunks)),
    new HumanMessage(question),
  ]);

  return typeof response.content === "string"
    ? response.content
    : response.content
        .map((part) => ("text" in part ? part.text : ""))
        .join("");
}