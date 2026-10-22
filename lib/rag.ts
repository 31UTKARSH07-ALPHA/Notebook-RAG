import { BaseCache } from "@langchain/core/caches";
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

const EMBED_BATCH_SIZE = 96; // Cohere embed accepts at most 96 texts per request
const RERANK_MODEL = "rerank-v3.5";
const DEFAULT_CHAT_MODEL = "openai/gpt-oss-120b";
// Smaller, faster model for bulk utility work (context headers at upload)
const DEFAULT_UTILITY_MODEL = "openai/gpt-oss-20b";
const HEADER_PAGE_CHARS = 6000; // page text sent when writing a context header

// Conversation history used to resolve follow-up questions. Enforced here as
// well as in the client, since the request body can't be trusted to be small.
const MAX_HISTORY_TURNS = 6; // last 3 question/answer pairs
const MAX_HISTORY_TURN_CHARS = 1000; // long answers are cut; their opening carries the gist

// Sentence-window chunking config
const SENTENCES_PER_CHILD = 2; // small chunk indexed in Qdrant
const CONTEXT_SENTENCES = 2; // sentences added on each side to build the parent sent to the LLM
export const MAX_SENTENCE_CHARS = 500; // longer "sentences" (tables, unpunctuated text) get split on words

// Fixed-size chunking: the default. In the evaluation it retrieved the most
// evidence of the three strategies, with less context than sentence-window and
// no extra upload cost (see DECISIONS.md and eval/RESULTS.md).
export const FIXED_CHUNK_CHARS = 1000;
export const FIXED_OVERLAP_CHARS = 200;

// Semantic chunking: cut where consecutive sentences stop being about the
// same thing, judged by the similarity of their embeddings.
// Similarities below this percentile of the document's own consecutive-pair
// similarities count as a topic shift. A per-document percentile instead of a
// fixed cutoff, because absolute cosine values vary a lot between documents
// (a policy text that repeats "AI risk" everywhere scores high throughout).
// At 25%, roughly one sentence gap in four is a candidate cut, which with
// ~150-character sentences lands near the max size below rather than far under it.
const SEMANTIC_BREAKPOINT_PERCENTILE = 25;
// Same ceiling as "fixed", so the comparison is about where chunks are cut,
// not about chunk size.
const SEMANTIC_MAX_CHUNK_CHARS = 1000;
// Below this a similarity drop doesn't cut: a 1-2 sentence fragment has too
// little context to answer anything on its own.
const SEMANTIC_MIN_CHUNK_CHARS = 300;

export type ChunkingStrategy = "sentence-window" | "fixed" | "semantic";
export const DEFAULT_CHUNKING: ChunkingStrategy = "fixed";

// Embedding models
//
// The dense vector size is fixed when a collection is created, and vectors
// from different models (or sizes) are not comparable, so a collection must
// be queried with exactly the model and size it was built with.
//
// embed-v4.0 is a Matryoshka model: its smaller output sizes are prefixes of
// the full vector, trained to stand on their own. Smaller vectors cost less to
// store and search (float32 = dim x 4 bytes: 256 -> 1 KB, 1024 -> 4 KB,
// 1536 -> 6 KB per chunk, before Qdrant's index overhead) at some loss of
// accuracy, which the eval measures.
export type EmbeddingConfig = { model: string; dim: number };

const EMBEDDING_MODELS: Record<string, { dims: number[]; outputDimension: boolean }> = {
  // Fixed size; Cohere rejects output_dimension for v3 models
  "embed-english-v3.0": { dims: [1024], outputDimension: false },
  // First entry is the model's native size, used when no size is given
  "embed-v4.0": { dims: [1536, 1024, 512, 256], outputDimension: true },
};

export const DEFAULT_EMBEDDING: EmbeddingConfig = { model: "embed-english-v3.0", dim: 1024 };

export function formatEmbedding(config: EmbeddingConfig): string {
  return `${config.model}@${config.dim}`;
}

// Throws on an unknown model or a size the model can't produce, so a typo in
// an env var fails on the first request instead of building a
// collection that later queries can't match.
export function resolveEmbedding(model: string, dim?: number): EmbeddingConfig {
  const spec = EMBEDDING_MODELS[model];
  if (!spec) {
    throw new Error(
      `Unknown embedding model "${model}". Supported: ${Object.keys(EMBEDDING_MODELS).join(", ")}.`
    );
  }
  const size = dim ?? spec.dims[0];
  if (!spec.dims.includes(size)) {
    throw new Error(`${model} cannot produce ${size}-d vectors. Supported sizes: ${spec.dims.join(", ")}.`);
  }
  return { model, dim: size };
}

// "embed-v4.0@512" -> { model: "embed-v4.0", dim: 512 }; the size is optional
export function parseEmbedding(value: string): EmbeddingConfig {
  const [model, dim] = value.split("@");
  if (dim !== undefined && !/^\d+$/.test(dim)) {
    throw new Error(`Invalid embedding "${value}". Expected <model>@<dim>, e.g. embed-v4.0@512.`);
  }
  return resolveEmbedding(model, dim === undefined ? undefined : Number(dim));
}

// BM25 parameters (standard defaults)
const BM25_K1 = 1.2;
const BM25_B = 0.75;

// Retrieval config
const PREFETCH_LIMIT = 20; // candidates per search path (dense/sparse x each query)
const RERANK_CANDIDATES = 20; // fused candidates sent to the reranker

export const NOT_FOUND_ANSWER = "I could not find that in the uploaded document.";

export type SourceChunk = {
  fileName?: string;
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
  // What was actually searched and answered: the question itself, or the
  // follow-up rewritten to stand alone when there was conversation history
  standaloneQuestion: string;
  sources: SourceChunk[];
  grounding: Grounding;
  queries: string[];
  timingsMs: Record<string, number>;
  context: Document[]; // what the LLM was given (after grading/compression)
  retrieved: Document[]; // top-K after search and reranking, before grading
};

export type PipelineOptions = {
  search: "dense" | "sparse" | "hybrid";
  multiQuery: boolean; // add LLM rewrites of the question
  decompose: boolean; // split multi-part questions and retrieve for each part
  rerank: boolean;
  grade: boolean; // LLM relevance filter (CRAG)
  compress: boolean;
  groundingCheck: boolean; // hallucination grader + one retry
  topK: number;
};

// One earlier message of the conversation, sent by the client
export type ChatTurn = { role: "user" | "assistant"; content: string };

// Reported to onStage as each stage starts, so the UI can show progress.
// Names match the timingsMs keys where a stage has its own timing.
export type PipelineStage =
  | "condense"
  | "decompose"
  | "queryRewrite"
  | "search"
  | "rerank"
  | "gradeChunks"
  | "compress"
  | "generate"
  | "groundingCheck"
  | "retry";

export type StageListener = (stage: PipelineStage) => void;

export type AnswerHooks = {
  history?: ChatTurn[];
  onStage?: StageListener;
};

export const DEFAULT_PIPELINE: PipelineOptions = {
  search: "hybrid",
  // Measured on fixed chunks: rewrites added nothing (same recall and MRR),
  // while decomposition lifted multi-hop recall from 76.7% to 91.7%. Both
  // cost one LLM call, so decomposition takes the slot.
  multiQuery: false,
  decompose: true,
  rerank: true,
  grade: true,
  compress: true,
  groundingCheck: true,
  topK: 5,
};

// Process-wide settings. The app uses the defaults; the evaluation harness
// turns on strict mode (so a failed stage is an error, not a silent fallback
// that would skew the scores), more retries, and response caches.
export type KeyValueCache = {
  get(key: string): Promise<unknown | undefined>;
  set(key: string, value: unknown): Promise<void>;
};

type RuntimeConfig = {
  strict: boolean;
  maxRetries: number;
  llmCache?: BaseCache;
  apiCache?: KeyValueCache;
  // Model for new collections; overrides EMBEDDING_MODEL / EMBEDDING_DIM
  embedding?: EmbeddingConfig;
};

const runtimeConfig: RuntimeConfig = { strict: false, maxRetries: 2 };

export function configureRuntime(config: Partial<RuntimeConfig>): void {
  Object.assign(runtimeConfig, config);
}

// The model new collections are built with. Existing collections keep the
// model recorded in their metadata (see getCollectionEmbedding).
export function getEmbeddingConfig(): EmbeddingConfig {
  if (runtimeConfig.embedding) return runtimeConfig.embedding;
  const model = process.env.EMBEDDING_MODEL || DEFAULT_EMBEDDING.model;
  const dim = process.env.EMBEDDING_DIM;
  if (dim && !/^\d+$/.test(dim)) throw new Error(`EMBEDDING_DIM must be a number, got "${dim}".`);
  return resolveEmbedding(model, dim ? Number(dim) : undefined);
}

function handleStageFailure(stage: string, error: unknown): void {
  if (runtimeConfig.strict) throw error;
  console.error(`${stage} failed, using fallback:`, error);
}

// Progress is cosmetic: a listener that throws (e.g. its client disconnected)
// must never fail the answer itself.
function reportStage(onStage: StageListener | undefined, stage: PipelineStage): void {
  try {
    onStage?.(stage);
  } catch (error) {
    console.warn(`Stage listener failed on "${stage}":`, error);
  }
}

export type SparseVector = {
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
  const text =
    typeof content === "string"
      ? content
      : content.map((part: any) => ("text" in part ? part.text : "")).join("");
  // Reasoning models such as Qwen3 may put their thinking inline
  return text.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
}

// Cohere (embeddings + rerank)

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

// Retries rate limits (429) and server errors, honouring Retry-After when sent
async function cohereRequest(path: string, body: unknown): Promise<any> {
  const payload = JSON.stringify(body);
  const cacheKey = `cohere:${path}:${payload}`;
  const cached = await runtimeConfig.apiCache?.get(cacheKey);
  if (cached !== undefined) return cached;

  for (let attempt = 0; ; attempt++) {
    const response = await fetch(`https://api.cohere.com/v2/${path}`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${requireEnv("COHERE_API_KEY")}`,
        "Content-Type": "application/json",
      },
      body: payload,
    });

    if (response.ok) {
      const data = await response.json();
      await runtimeConfig.apiCache?.set(cacheKey, data);
      return data;
    }

    const retryable = response.status === 429 || response.status >= 500;
    if (!retryable || attempt >= runtimeConfig.maxRetries) {
      throw new Error(`Cohere ${path} failed (${response.status}): ${await response.text()}`);
    }
    const retryAfter = Number(response.headers.get("retry-after"));
    await sleep(retryAfter > 0 ? retryAfter * 1000 : Math.min(60_000, 2 ** attempt * 2000));
  }
}

// Documents and queries use different input_type values so Cohere embeds them
// into the asymmetric spaces it was trained for.
async function embed(
  texts: string[],
  inputType: "search_document" | "search_query",
  config: EmbeddingConfig
): Promise<number[][]> {
  const vectors: number[][] = [];
  const { outputDimension } = EMBEDDING_MODELS[config.model];

  for (let i = 0; i < texts.length; i += EMBED_BATCH_SIZE) {
    // output_dimension is only sent to models that take it, so the v3 request
    // (and therefore its cache key) is byte-for-byte what it was before.
    const data = await cohereRequest("embed", {
      texts: texts.slice(i, i + EMBED_BATCH_SIZE),
      model: config.model,
      input_type: inputType,
      embedding_types: ["float"],
      ...(outputDimension ? { output_dimension: config.dim } : {}),
    });
    vectors.push(...data.embeddings.float);
  }

  const wrongSize = vectors.find((v) => v?.length !== config.dim);
  if (wrongSize) {
    throw new Error(
      `${config.model} returned ${wrongSize?.length ?? 0}-d vectors, expected ${config.dim}.`
    );
  }
  return vectors;
}

function getUtilityModel(): ChatGroq {
  requireEnv("GROQ_API_KEY");
  return new ChatGroq({
    model: process.env.GROQ_UTILITY_MODEL || DEFAULT_UTILITY_MODEL,
    temperature: 0,
    maxTokens: 2048,
    maxRetries: runtimeConfig.maxRetries,
    cache: runtimeConfig.llmCache,
  });
}

export function getChatModel(): ChatGroq {
  requireEnv("GROQ_API_KEY");
  return new ChatGroq({
    model: process.env.GROQ_MODEL || DEFAULT_CHAT_MODEL,
    temperature: 0.1,
    // gpt-oss is a reasoning model: its hidden reasoning tokens count against this limit
    maxTokens: 4096,
    maxRetries: runtimeConfig.maxRetries,
    cache: runtimeConfig.llmCache,
  });
}

export function getQdrantClient(): QdrantClient {
  return new QdrantClient({
    url: requireEnv("QDRANT_URL"),
    apiKey: process.env.QDRANT_API_KEY || undefined,
    checkCompatibility: false,
  });
}

// Parsing

// Collapses whitespace and rejoins words hyphenated at line breaks
// ("abil- ity" -> "ability"). Real compounds ("long-context") have no space
// after the hyphen, and "best- and worst-case" is a suspended hyphen, so
// and/or/to are kept.
export function cleanPageText(pageText: string): string {
  return pageText
    .replace(/\s+/g, " ")
    .replace(/([a-z])- (?!(?:and|or|to)\b)([a-z])/g, "$1$2")
    .trim();
}

async function parsePdf(buffer: ArrayBuffer): Promise<Document[]> {
  const { text, totalPages } = await extractText(
    buffer,
    { mergePages: false }
  );
  const pages = Array.isArray(text) ? text : [text];

  return pages
    .map((pageText, index) =>
      new Document({
        pageContent: cleanPageText(pageText),
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
export function splitLongSentence(sentence: string): string[] {
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

export function splitIntoSentences(text: string): string[] {
  // Split on sentence-ending punctuation followed by whitespace + capital letter
  return text
    .replace(/\s+/g, " ")
    .split(/(?<=[.!?])\s+(?=[A-Z"'])/)
    .map((s) => s.trim())
    .filter((s) => s.length > 10)
    .flatMap(splitLongSentence);
}

// ~1000-character chunks with 200 characters of overlap, cut on word
// boundaries. The chunk that is searched is also what the LLM sees.
function chunkFixedSize(documents: Document[]): Document[] {
  const chunks: Document[] = [];
  let chunkIndex = 1;

  for (const doc of documents) {
    const words = doc.pageContent.replace(/\s+/g, " ").trim().split(" ");
    if (!words[0]) continue; // empty page: "".split(" ") is [""], which would make an empty chunk
    let start = 0;
    while (start < words.length) {
      let end = start;
      let length = 0;
      while (end < words.length && length + words[end].length + 1 <= FIXED_CHUNK_CHARS) {
        length += words[end].length + 1;
        end++;
      }
      end = Math.max(end, start + 1); // a single over-long word still advances
      const text = words.slice(start, end).join(" ");
      chunks.push(
        new Document({
          pageContent: text,
          metadata: { ...doc.metadata, chunk: chunkIndex++, chunking: "fixed-1000-200", parentText: text },
        })
      );
      if (end >= words.length) break;

      // Step back ~200 characters for the overlap
      let back = end;
      let overlap = 0;
      while (back > start + 1 && overlap < FIXED_OVERLAP_CHARS) {
        back--;
        overlap += words[back].length + 1;
      }
      start = back;
    }
  }

  return chunks;
}

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA * normB) || 1);
}

// Groups consecutive sentences into chunks, starting a new chunk where the
// embedding similarity between two neighbouring sentences drops below the
// document's breakpoint threshold, or when the chunk would exceed the max
// size. Chunks never cross a page, so each keeps one page number. Like
// "fixed", the chunk that is searched is also what the LLM sees.
//
// Cost: every sentence is embedded once at upload, only to find the cut
// points, and then the chunks are embedded again by indexChunks(). That
// doubles the embedded text and adds one Cohere call per 96 sentences: for
// the 48-page NIST PDF, ~7 sentence calls on top of 2 for the chunks.
async function chunkSemantic(documents: Document[]): Promise<Document[]> {
  const pages = documents.map((doc) => splitIntoSentences(doc.pageContent));
  const vectors = await embed(pages.flat(), "search_document", getEmbeddingConfig());

  // Similarity between each sentence and the next one on the same page
  let offset = 0;
  const pageSimilarities = pages.map((sentences) => {
    const similarities = sentences
      .slice(1)
      .map((_, i) => cosineSimilarity(vectors[offset + i], vectors[offset + i + 1]));
    offset += sentences.length;
    return similarities;
  });

  // Threshold from the whole document's distribution, not per page, so a
  // page that is all one topic isn't forced to cut somewhere anyway
  const allSimilarities = pageSimilarities.flat().sort((a, b) => a - b);
  const threshold = allSimilarities.length
    ? allSimilarities[Math.floor((SEMANTIC_BREAKPOINT_PERCENTILE / 100) * (allSimilarities.length - 1))]
    : 0;

  const chunks: Document[] = [];
  let chunkIndex = 1;

  documents.forEach((doc, pageIndex) => {
    const sentences = pages[pageIndex];
    const groups: string[][] = [];
    let current: string[] = [];
    let length = 0;

    sentences.forEach((sentence, i) => {
      if (current.length > 0) {
        const tooLong = length + sentence.length + 1 > SEMANTIC_MAX_CHUNK_CHARS;
        const topicShift =
          pageSimilarities[pageIndex][i - 1] < threshold && length >= SEMANTIC_MIN_CHUNK_CHARS;
        if (tooLong || topicShift) {
          groups.push(current);
          current = [];
          length = 0;
        }
      }
      current.push(sentence);
      length += sentence.length + (current.length > 1 ? 1 : 0);
    });

    // A short tail at the end of the page joins the previous chunk when it fits
    if (current.length > 0) {
      const previous = groups[groups.length - 1];
      if (
        previous &&
        length < SEMANTIC_MIN_CHUNK_CHARS &&
        previous.join(" ").length + 1 + length <= SEMANTIC_MAX_CHUNK_CHARS
      ) {
        previous.push(...current);
      } else {
        groups.push(current);
      }
    }

    for (const group of groups) {
      const text = group.join(" ");
      chunks.push(
        new Document({
          pageContent: text,
          metadata: {
            ...doc.metadata,
            chunk: chunkIndex++,
            chunking: `semantic-p${SEMANTIC_BREAKPOINT_PERCENTILE}-${SEMANTIC_MIN_CHUNK_CHARS}-${SEMANTIC_MAX_CHUNK_CHARS}`,
            parentText: text,
          },
        })
      );
    }
  });

  return chunks;
}

export async function chunkDocuments(
  documents: Document[],
  strategy: ChunkingStrategy = DEFAULT_CHUNKING
): Promise<Document[]> {
  if (strategy === "fixed") return chunkFixedSize(documents);
  if (strategy === "semantic") return chunkSemantic(documents);

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

export function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((t) => t.length > 1);
}

export function hashToken(term: string): number {
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

export function buildDocumentSparseVector(text: string, avgDocLength: number): SparseVector {
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

export function buildQuerySparseVector(text: string): SparseVector {
  const indices = Array.from(hashedTermCounts(tokenize(text)).keys());
  return { indices, values: indices.map(() => 1) };
}

// Indexing

// Each collection records the embedding model it was built with in its
// Qdrant metadata (Qdrant >= 1.16). Queries and later uploads read it back, so
// changing EMBEDDING_MODEL only affects new notebooks and never mixes vector
// spaces inside one collection. Cached per process: it can't change while the
// collection exists.
const collectionEmbeddings = new Map<string, EmbeddingConfig>();

async function createNotebookCollection(
  collectionName: string,
  embedding: EmbeddingConfig
): Promise<void> {
  collectionEmbeddings.delete(collectionName);
  // Goes through api() because the client's createCollection() wrapper
  // (js-client-rest 1.18) accepts `metadata` in its type but doesn't send it.
  await getQdrantClient().api().createCollection({
    collection_name: collectionName,
    vectors: {
      [DENSE_VECTOR_NAME]: {
        size: embedding.dim,
        distance: "Cosine",
      },
    },
    sparse_vectors: {
      [SPARSE_VECTOR_NAME]: { modifier: "idf" },
    },
    metadata: { embedding },
  });
}

export async function getCollectionEmbedding(collectionName: string): Promise<EmbeddingConfig> {
  const cached = collectionEmbeddings.get(collectionName);
  if (cached) return cached;

  const { config } = await getQdrantClient().getCollection(collectionName);
  const vectors = config.params.vectors as Record<string, { size: number }> | undefined;
  const size = vectors?.[DENSE_VECTOR_NAME]?.size;
  const stored = (config.metadata as { embedding?: EmbeddingConfig } | null | undefined)?.embedding;

  // Collections from before this setting existed have no metadata; the only
  // model then was embed-english-v3.0 at 1024-d.
  const embedding = stored
    ? resolveEmbedding(stored.model, stored.dim)
    : DEFAULT_EMBEDDING;
  if (size !== embedding.dim) {
    throw new Error(
      `Collection ${collectionName} has ${size}-d vectors but ` +
      `${stored ? "its metadata says" : "has no embedding metadata, so it is assumed to be"} ` +
      `${formatEmbedding(embedding)}. Re-index the documents into a new collection.`
    );
  }

  collectionEmbeddings.set(collectionName, embedding);
  return embedding;
}

export async function indexChunks(
  collectionName: string,
  chunks: Document[]
): Promise<void> {
  if (chunks.length === 0) {
    throw new Error("No readable text was found in the uploaded document.");
  }

  const client = getQdrantClient();

  // A notebook is one collection; later documents are appended to it, and
  // are embedded with the collection's model even if the setting changed since.
  const { exists } = await client.collectionExists(collectionName);
  if (!exists) await createNotebookCollection(collectionName, getEmbeddingConfig());
  const embedding = await getCollectionEmbedding(collectionName);

  // Context headers (if any) are only used for search; the stored text and
  // what the LLM later sees stay the original chunk.
  const texts = chunks.map((c) =>
    c.metadata.contextHeader ? `${c.metadata.contextHeader}\n${c.pageContent}` : c.pageContent
  );
  const denseVectors = await embed(texts, "search_document", embedding);

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
      [SPARSE_VECTOR_NAME]: buildDocumentSparseVector(texts[i], avgDocLength),
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

// Contextual chunk headers
//
// A 2-sentence chunk like "It can be spent on courses and books." doesn't say
// what "it" is. For each page, a small LLM writes 1-2 sentences placing the
// page in the document (document name, section, topic), and that header is
// prepended to the page's chunks before embedding and BM25. This is a
// page-level version of Anthropic's "contextual retrieval": one call per page
// instead of one call per chunk with the whole document.
export async function addContextHeaders(pages: Document[], chunks: Document[]): Promise<void> {
  const llm = getUtilityModel();
  const opening = (pages[0]?.pageContent ?? "").split(" ").slice(0, 250).join(" ");
  const pageKey = (doc: Document) => doc.metadata.loc?.pageNumber ?? 1;
  const headers = new Map<number, string>();

  // A few pages at a time keeps upload fast without hammering rate limits
  const CONCURRENCY = 3;
  for (let i = 0; i < pages.length; i += CONCURRENCY) {
    await Promise.all(
      pages.slice(i, i + CONCURRENCY).map(async (page) => {
        const prompt = `Here is the beginning of a document:
<opening>
${opening}
</opening>

Here is one page of the same document:
<page>
${page.pageContent.slice(0, HEADER_PAGE_CHARS)}
</page>

Write 1-2 sentences that situate this page within the document, to improve search over its passages: name the document, then the section and the topics this page covers, using the document's own key terms. Output only the sentences.`;
        try {
          const response = await llm.invoke([new HumanMessage(prompt)]);
          const header = messageText(response.content).replace(/\s+/g, " ").trim();
          if (header) headers.set(pageKey(page), header);
        } catch (error) {
          handleStageFailure("Context header", error); // chunks just go without a header
        }
      })
    );
  }

  for (const chunk of chunks) {
    const header = headers.get(pageKey(chunk));
    if (header) chunk.metadata.contextHeader = header;
  }
}

// Only collections this app created count as notebooks, so a client can't
// pass an arbitrary name and read or drop someone else's data.
export function isNotebookCollection(collectionName: string): boolean {
  return collectionName.startsWith(`${getCollectionPrefix()}_`);
}

export async function deleteCollection(collectionName: string): Promise<void> {
  if (!isNotebookCollection(collectionName)) return;
  collectionEmbeddings.delete(collectionName);
  try {
    await getQdrantClient().deleteCollection(collectionName);
  } catch (error) {
    console.warn(`Could not delete collection ${collectionName}:`, error);
  }
}

// Removes one document's chunks from a notebook (used when indexing fails midway)
export async function deleteDocumentChunks(collectionName: string, documentId: string): Promise<void> {
  if (!isNotebookCollection(collectionName)) return;
  try {
    await getQdrantClient().delete(collectionName, {
      wait: true,
      filter: { must: [{ key: "metadata.documentId", match: { value: documentId } }] },
    });
  } catch (error) {
    console.warn(`Could not remove document ${documentId} from ${collectionName}:`, error);
  }
}

// Query translation

// Keeps only the most recent, well-formed turns and cuts long ones, so a
// follow-up costs a bounded number of tokens however long the chat gets.
export function capHistory(history: unknown): ChatTurn[] {
  if (!Array.isArray(history)) return [];
  return history
    .filter(
      (turn): turn is ChatTurn =>
        (turn?.role === "user" || turn?.role === "assistant") &&
        typeof turn?.content === "string" &&
        turn.content.trim().length > 0
    )
    .slice(-MAX_HISTORY_TURNS)
    .map(({ role, content }) => {
      const text = content.replace(/\s+/g, " ").trim();
      return {
        role,
        content:
          text.length > MAX_HISTORY_TURN_CHARS ? `${text.slice(0, MAX_HISTORY_TURN_CHARS)}…` : text,
      };
    });
}

// Follow-ups like "what about sales staff?" can't be searched on their own:
// neither the embedding nor BM25 knows what "what about" refers to. One LLM
// call rewrites the follow-up into a standalone question using the
// conversation, and everything downstream (retrieval, generation, grading)
// uses that. The history is only used here, to resolve references; it is
// never shown to the answer generator, so earlier answers can't become a
// source of facts.
export async function condenseQuestion(
  question: string,
  history: ChatTurn[],
  llm?: ChatGroq
): Promise<string> {
  const turns = capHistory(history);
  if (turns.length === 0) return question;
  // The small model is enough for a rewrite and keeps follow-ups fast
  const model = llm || getUtilityModel();

  const conversation = turns
    .map((turn) => `${turn.role === "user" ? "User" : "Assistant"}: ${turn.content}`)
    .join("\n");

  const prompt = `Given a conversation and a follow-up question, rewrite the follow-up as a standalone question that can be understood without the conversation.

- Replace pronouns and vague references ("it", "they", "that policy", "what about X?") with what they refer to in the conversation.
- Otherwise keep the user's intent and wording. Do not answer the question or add facts.
- If the follow-up is already standalone or changes topic, return it unchanged.

<conversation>
${conversation}
</conversation>

Follow-up question: "${question}"

Output only the standalone question.`;

  try {
    const response = await model.invoke([new HumanMessage(prompt)]);
    const rewritten = messageText(response.content)
      .replace(/^\s*standalone question:\s*/i, "")
      .replace(/^["'“]+|["'”]+$/g, "")
      .replace(/\s+/g, " ")
      .trim();
    // An empty or rambling reply is worse than the raw follow-up
    if (!rewritten || rewritten.length > Math.max(300, question.length * 4)) return question;
    return rewritten;
  } catch (error) {
    handleStageFailure("Question condensing", error);
    return question;
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
    return messageText(response.content)
      .split("\n")
      .map((line) => line.match(/^\s*[1-3][.)]\s*(.+)$/)?.[1]?.trim())
      .filter((line): line is string => Boolean(line))
      .slice(0, 3);
  } catch (error) {
    handleStageFailure("Query rewriting", error);
    return [];
  }
}

const MAX_SUB_QUESTIONS = 3;

// Query decomposition: a question like "What does GOVERN 1.7 require, and
// which function covers third-party risk?" needs two passages that rarely sit
// in the same chunk, and a single search tends to fill the top-K with
// whichever half matches best. The LLM splits it into standalone
// sub-questions so each half gets its own search. Single-part questions come
// back unchanged as [question].
export async function decomposeQuestion(
  question: string,
  llm: ChatGroq
): Promise<string[]> {
  const prompt = `Decide whether the user question asks for more than one distinct piece of information.

Question: "${question}"

If it asks for one thing, output the question unchanged as a single line numbered 1.
If it asks for several things, split it into 2 or 3 standalone sub-questions, one per line, numbered 1., 2., 3.
Each sub-question must make sense on its own: repeat names and terms instead of using "it" or "they".

Do not include any other text or explanations.`;

  try {
    const response = await llm.invoke([new HumanMessage(prompt)]);
    return parseSubQuestions(messageText(response.content), question);
  } catch (error) {
    handleStageFailure("Query decomposition", error);
    return [question];
  }
}

// Accepts numbered ("1." / "1)") or bulleted lines, tolerating code fences and
// quotes. Anything that doesn't yield 2+ distinct sub-questions means "don't
// decompose", so messy output degrades to the normal single-question path.
export function parseSubQuestions(text: string, question: string): string[] {
  const subQuestions = Array.from(
    new Set(
      text
        .split("\n")
        .map((line) => line.match(/^\s*(?:\d+[.)]|[-*•])\s*(.+)$/)?.[1])
        .map((line) => line?.replace(/^["'`*]+|["'`*]+$/g, "").trim())
        .filter((line): line is string => Boolean(line && line.length > 3))
    )
  ).slice(0, MAX_SUB_QUESTIONS);

  return subQuestions.length > 1 ? subQuestions : [question];
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
          id: docs[result.index].id,
          pageContent: docs[result.index].pageContent,
          metadata: {
            ...docs[result.index].metadata,
            rerankScore: result.relevance_score,
          },
        })
    );
  } catch (error) {
    // Fusion order is still a reasonable ranking if the reranker is down
    handleStageFailure("Rerank", error);
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
  limit = RERANK_CANDIDATES,
  mode: PipelineOptions["search"] = "hybrid"
): Promise<Document[]> {
  const useDense = mode !== "sparse";
  const useSparse = mode !== "dense";
  // Queries must land in the same vector space as the stored chunks
  const queryVectors = useDense
    ? await embed(queries, "search_query", await getCollectionEmbedding(collectionName))
    : [];

  const prefetch = queries.flatMap((query, i) => {
    const sparse = buildQuerySparseVector(query);
    return [
      ...(useDense
        ? [{ query: queryVectors[i], using: DENSE_VECTOR_NAME, limit: PREFETCH_LIMIT }]
        : []),
      ...(useSparse && sparse.indices.length
        ? [{ query: sparse, using: SPARSE_VECTOR_NAME, limit: PREFETCH_LIMIT }]
        : []),
    ];
  });
  if (prefetch.length === 0) return [];

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
      id: String(point.id), // identity for de-duplicating across searches
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

// Merges per-sub-question rankings round-robin: best of list 1, best of
// list 2, ..., then the second-best of each, until topK. This guarantees every
// sub-question a share of the context, which a merge by score would not:
// rerank scores from different queries aren't comparable, and one easy part
// could take every slot. A chunk found by two sub-questions is kept once, and
// that list's turn passes to its next unseen chunk.
export function interleaveRankings(lists: Document[][], topK: number): Document[] {
  const chunkKey = (doc: Document) =>
    doc.id ?? `${doc.metadata.documentId ?? doc.metadata.source}:${doc.metadata.chunk}`;
  const seen = new Set<string>();
  const merged: Document[] = [];
  const cursors = lists.map(() => 0);

  while (merged.length < topK && cursors.some((c, i) => c < lists[i].length)) {
    for (const [i, list] of lists.entries()) {
      while (cursors[i] < list.length && seen.has(chunkKey(list[cursors[i]]))) cursors[i]++;
      if (cursors[i] >= list.length || merged.length >= topK) continue;
      const doc = list[cursors[i]++];
      seen.add(chunkKey(doc));
      merged.push(doc);
    }
  }
  return merged;
}

export async function retrieveChunks(
  collectionName: string,
  question: string,
  options: PipelineOptions = DEFAULT_PIPELINE,
  timingsMs: Record<string, number> = {},
  onStage?: StageListener
): Promise<{ chunks: Document[]; candidates: Document[]; queries: string[] }> {
  const llm = getChatModel();
  const time = async <T>(stage: PipelineStage, fn: () => Promise<T>): Promise<T> => {
    reportStage(onStage, stage);
    const start = Date.now();
    try {
      return await fn();
    } finally {
      timingsMs[stage] = Date.now() - start;
    }
  };

  // Step 0: Decomposition. A multi-part question becomes 2-3 sub-questions,
  // each retrieved on its own; a single-part question stays as it is.
  const subQuestions = options.decompose
    ? await time("decompose", () => decomposeQuestion(question, llm))
    : [question];

  // Step 1: Query translation. The original question is always searched too,
  // so a bad rewrite can only add candidates, never lose the obvious ones.
  //
  // Rewrites are only generated when the question was NOT decomposed. Per
  // sub-question rewrites would cost 2-3 extra LLM calls per question (the
  // scarcest resource on Groq's free tier), and the sub-questions already are
  // focused rephrasings of the parts. Rewriting the whole question and adding
  // it to every sub-question's search would pull each list back towards
  // whichever part dominates, which is the problem decomposition fixes.
  const variants = options.multiQuery && subQuestions.length === 1
    ? await time("queryRewrite", () => generateQueryVariants(question, llm))
    : [];
  const queryLists =
    subQuestions.length > 1
      ? subQuestions.map((subQuestion) => [subQuestion])
      : [Array.from(new Set([question, ...variants]))];

  // Step 2: Hybrid search + RRF fusion, one fused list per sub-question.
  // Without a reranker there is nothing to choose between, so only the final
  // top-K is fetched.
  const fetchLimit = options.rerank ? RERANK_CANDIDATES : options.topK;
  const fusedLists = await time("search", () =>
    Promise.all(
      queryLists.map((queries) =>
        hybridSearch(collectionName, queries, fetchLimit, options.search)
      )
    )
  );

  // Step 3: Cross-encoder reranking, each list against its own sub-question
  // (the whole question would score one part's passages low).
  const rankedLists = options.rerank
    ? await time("rerank", () =>
        Promise.all(
          fusedLists.map((fused, i) => rerank(queryLists[i][0], fused, options.topK))
        )
      )
    : fusedLists.map((fused) => fused.slice(0, options.topK));
  const candidates = interleaveRankings(rankedLists, options.topK);
  const queries = queryLists.flat();

  // Step 4: Relevance grading (CRAG). Empty result means "not in the document".
  const relevant = options.grade
    ? await time("gradeChunks", () => gradeChunks(question, candidates, llm))
    : candidates;
  if (relevant.length === 0) return { chunks: [], candidates, queries };

  // Step 5: Context compression
  const chunks = options.compress
    ? await time("compress", () => compressContext(question, relevant, llm))
    : relevant;
  return { chunks, candidates, queries };
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
    handleStageFailure("Relevance grading", error);
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
        handleStageFailure("Compression", error);
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
    handleStageFailure("Grounding check", error);
  }

  return { verdict: "unverified", confidence: 0, reasoning: "Grader output could not be parsed" };
}

// Answer generation

export function formatSources(chunks: Document[]): SourceChunk[] {
  return chunks.map((chunk, index) => ({
    fileName: chunk.metadata.source || undefined,
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
      const file = chunk.metadata.source ? ` | ${chunk.metadata.source}` : "";
      return `SOURCE ${index + 1}${file} | chunk ${chunkNumber}${page ? ` | page ${page}` : ""}\n${chunk.pageContent}`;
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

  return normaliseCitations(messageText(response.content)).trim() || NOT_FOUND_ANSWER;
}

// gpt-oss sometimes ignores the [n] instruction and writes its own markers,
// e.g. "【1†L1】" or "【SOURCE 2 | chunk 7】". Rewrite them to [n].
export function normaliseCitations(answer: string): string {
  return answer.replace(/【\s*(?:SOURCE\s*)?(\d+)[^】]*】/gi, "[$1]");
}

// The model sometimes drops the period or rewords the refusal slightly
export function isNotFoundAnswer(answer: string): boolean {
  return answer.toLowerCase().includes("could not find that in the uploaded document");
}

// Full query pipeline: condense follow-up -> retrieve -> generate -> check
// grounding -> retry once.
//
// hooks.onStage reports progress as each stage starts. The answer itself is
// deliberately not streamed token by token: it is only final after the
// grounding check, and a draft that fails the check is thrown away and
// regenerated. Streaming tokens would show the user text that may then be
// replaced (including the unsupported claim the grader caught), so callers
// get stage updates while waiting and the answer once it has been checked.
export async function answerQuestion(
  collectionName: string,
  question: string,
  options: PipelineOptions = DEFAULT_PIPELINE,
  hooks: AnswerHooks = {}
): Promise<AnswerResult> {
  const { onStage } = hooks;
  const timingsMs: Record<string, number> = {};
  const started = Date.now();

  // Without history there is nothing to resolve, so no extra LLM call
  let standaloneQuestion = question;
  if (capHistory(hooks.history).length > 0) {
    reportStage(onStage, "condense");
    const start = Date.now();
    standaloneQuestion = await condenseQuestion(question, hooks.history ?? []);
    timingsMs.condense = Date.now() - start;
  }

  const { chunks, candidates, queries } = await retrieveChunks(
    collectionName,
    standaloneQuestion,
    options,
    timingsMs,
    onStage
  );

  const result = (answer: string, grounding: Grounding): AnswerResult => {
    timingsMs.total = Date.now() - started;
    return {
      answer,
      standaloneQuestion,
      sources: formatSources(chunks),
      grounding,
      queries,
      timingsMs,
      context: chunks,
      retrieved: candidates,
    };
  };

  if (chunks.length === 0) {
    return result(NOT_FOUND_ANSWER, {
      verdict: "not_found",
      confidence: 1,
      reasoning: "No retrieved chunk was relevant to the question.",
      retried: false,
    });
  }

  reportStage(onStage, "generate");
  let start = Date.now();
  let answer = await generateGroundedAnswer(chunks, standaloneQuestion);
  timingsMs.generate = Date.now() - start;

  if (isNotFoundAnswer(answer)) {
    return result(answer, {
      verdict: "not_found",
      confidence: 1,
      reasoning: "The model found no answer in the sources.",
      retried: false,
    });
  }

  if (!options.groundingCheck) {
    return result(answer, {
      verdict: "unverified",
      confidence: 0,
      reasoning: "Grounding check disabled.",
      retried: false,
    });
  }

  reportStage(onStage, "groundingCheck");
  start = Date.now();
  let grounding = await gradeAnswer(answer, chunks, standaloneQuestion);
  let retried = false;

  // One retry with the grader's feedback. More retries rarely help and double latency.
  if (grounding.verdict === "not_grounded") {
    retried = true;
    reportStage(onStage, "retry");
    answer = await generateGroundedAnswer(chunks, standaloneQuestion, grounding.reasoning);
    reportStage(onStage, "groundingCheck");
    grounding = await gradeAnswer(answer, chunks, standaloneQuestion);
  }
  timingsMs.groundingCheck = Date.now() - start;

  return result(answer, { ...grounding, retried });
}
