// Shared pieces of the evaluation harness: dataset loading, document
// download + indexing, disk caches, evidence matching and the LLM judge.

import { createHash } from "crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "fs";
import { dirname, join } from "path";
import { fileURLToPath } from "url";
import {
  BaseCache,
  deserializeStoredGeneration,
  serializeGeneration,
} from "@langchain/core/caches";
import { Generation } from "@langchain/core/outputs";
import { Document } from "@langchain/core/documents";
import { HumanMessage } from "@langchain/core/messages";
import { ChatGroq } from "@langchain/groq";
import {
  ChunkingStrategy,
  DEFAULT_EMBEDDING,
  EmbeddingConfig,
  KeyValueCache,
  addContextHeaders,
  chunkDocuments,
  deleteCollection,
  formatEmbedding,
  getCollectionEmbedding,
  getEmbeddingConfig,
  getQdrantClient,
  indexChunks,
  parseUploadedFile,
} from "../lib/rag";

export const EVAL_DIR = dirname(fileURLToPath(import.meta.url));
const DATA_DIR = join(EVAL_DIR, ".data");
const CACHE_DIR = join(EVAL_DIR, ".cache");

// Judge is a different model from the generator, to avoid a model grading
// its own answers favourably (self-preference bias).
export const JUDGE_MODEL = process.env.EVAL_JUDGE_MODEL || "openai/gpt-oss-20b";

export type QuestionType = "factual" | "keyword" | "multi_hop" | "unanswerable";

export type Question = {
  id: string;
  doc: string;
  type: QuestionType;
  question: string;
  reference_answer: string;
  evidence: string[];
  page: number[];
};

export type EvalDocument = {
  id: string;
  title: string;
  url: string;
  sha256: string;
};

export function loadQuestions(): Question[] {
  return JSON.parse(readFileSync(join(EVAL_DIR, "questions.json"), "utf8"));
}

export function loadDocuments(): EvalDocument[] {
  return JSON.parse(readFileSync(join(EVAL_DIR, "documents.json"), "utf8"));
}

const sha256 = (data: string | Buffer) => createHash("sha256").update(data).digest("hex");

// Caches: every embedding, rerank and LLM response is stored on disk keyed by
// its exact request, so re-running a suite (or a config that shares stages
// with another) costs nothing and gives identical results.

function cacheFile(namespace: string, key: string): string {
  const dir = join(CACHE_DIR, namespace);
  mkdirSync(dir, { recursive: true });
  return join(dir, `${sha256(key)}.json`);
}

export const apiCache: KeyValueCache = {
  async get(key) {
    const file = cacheFile("api", key);
    return existsSync(file) ? JSON.parse(readFileSync(file, "utf8")) : undefined;
  },
  async set(key, value) {
    writeFileSync(cacheFile("api", key), JSON.stringify(value));
  },
};

export class FileLLMCache extends BaseCache {
  async lookup(prompt: string, llmKey: string): Promise<Generation[] | null> {
    const file = cacheFile("llm", `${llmKey}\n${prompt}`);
    if (!existsSync(file)) return null;
    return JSON.parse(readFileSync(file, "utf8")).map(deserializeStoredGeneration);
  }

  async update(prompt: string, llmKey: string, value: Generation[]): Promise<void> {
    writeFileSync(
      cacheFile("llm", `${llmKey}\n${prompt}`),
      JSON.stringify(value.map(serializeGeneration))
    );
  }
}

// Documents

async function downloadDocument(doc: EvalDocument): Promise<Buffer> {
  mkdirSync(DATA_DIR, { recursive: true });
  const path = join(DATA_DIR, `${doc.id}.pdf`);

  if (!existsSync(path)) {
    const response = await fetch(doc.url);
    if (!response.ok) throw new Error(`Download failed for ${doc.url}: ${response.status}`);
    writeFileSync(path, Buffer.from(await response.arrayBuffer()));
  }

  const data = readFileSync(path);
  if (sha256(data) !== doc.sha256) {
    throw new Error(`${doc.id}: checksum mismatch, the evidence quotes may not match this file`);
  }
  return data;
}

export async function loadChunks(
  doc: EvalDocument,
  chunking: ChunkingStrategy,
  contextHeaders = false
): Promise<Document[]> {
  const data = await downloadDocument(doc);
  const file = new File([new Uint8Array(data)], `${doc.id}.pdf`, { type: "application/pdf" });
  const pages = await parseUploadedFile(file);
  const chunks = await chunkDocuments(pages, chunking);
  if (contextHeaders) await addContextHeaders(pages, chunks);
  return chunks;
}

// e.g. "sentence-window", "sentence-window+headers" or
// "sentence-window+embed-v4.0@512"; used in file and collection names. The
// default embedding model adds no suffix, so baseline names stay as they were.
export function indexVariant(
  chunking: ChunkingStrategy,
  contextHeaders: boolean,
  embedding: EmbeddingConfig = DEFAULT_EMBEDDING
): string {
  let variant: string = chunking;
  if (contextHeaders) variant += "+headers";
  if (formatEmbedding(embedding) !== formatEmbedding(DEFAULT_EMBEDDING)) {
    variant += `+${formatEmbedding(embedding)}`;
  }
  return variant;
}

// Qdrant names are kept to [A-Za-z0-9_] ("embed-v4.0@512" -> "embed_v4_0_512")
export function evalCollectionName(docId: string, variant: string): string {
  const prefix = process.env.QDRANT_COLLECTION_PREFIX || "sourcesage";
  return `${prefix}_eval_${docId}_${variant}`.replace(/[^A-Za-z0-9_]/g, "_");
}

// Reuses an existing collection when it already holds exactly these chunks.
// indexChunks() builds new collections with getEmbeddingConfig(), which the
// harness sets from --embedding.
export async function ensureIndexed(
  doc: EvalDocument,
  variant: string,
  chunks: Document[]
): Promise<string> {
  const name = evalCollectionName(doc.id, variant);
  const client = getQdrantClient();
  const { exists } = await client.collectionExists(name);

  // Reused only if it also holds vectors from the model being evaluated
  if (exists) {
    const { count } = await client.count(name, { exact: true });
    const built = formatEmbedding(await getCollectionEmbedding(name));
    if (count === chunks.length && built === formatEmbedding(getEmbeddingConfig())) return name;
    await deleteCollection(name);
  }

  await indexChunks(name, chunks);
  return name;
}

// Evidence matching. Quotes are matched against the text that was indexed
// (the child chunk), after normalising case and punctuation. A quote cut in
// two by a chunk boundary counts if at least half of it is in the chunk.

const normalise = (text: string) =>
  text.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();

export function chunkContainsEvidence(chunkText: string, quote: string): boolean {
  const chunk = normalise(chunkText);
  const words = normalise(quote).split(" ");
  if (chunk.includes(words.join(" "))) return true;

  const half = Math.ceil(words.length / 2);
  for (let n = words.length - 1; n >= half; n--) {
    if (chunk.endsWith(words.slice(0, n).join(" ")) || chunk.startsWith(words.slice(-n).join(" "))) {
      return true;
    }
  }
  return false;
}

export function indexedText(chunk: Document): string {
  return chunk.metadata.childText || chunk.pageContent;
}

// Retrieval metrics for one question over a ranked list of chunks
export function retrievalMetrics(question: Question, ranked: Document[]) {
  const texts = ranked.map(indexedText);
  const found = question.evidence.map((quote) => texts.some((t) => chunkContainsEvidence(t, quote)));
  const firstRelevant = texts.findIndex((t) =>
    question.evidence.some((quote) => chunkContainsEvidence(t, quote))
  );

  return {
    hit: found.some(Boolean) ? 1 : 0,
    recall: found.filter(Boolean).length / question.evidence.length,
    reciprocalRank: firstRelevant === -1 ? 0 : 1 / (firstRelevant + 1),
  };
}

// LLM judge

export type Judgement = {
  correctness: number; // 0, 0.5 or 1
  faithfulness: number | null; // supported claims / claims; null when the answer makes no claims
  claims: Array<{ claim: string; supported: boolean }>;
  reasoning: string;
};

let judgeModel: ChatGroq | undefined;

function getJudge(): ChatGroq {
  judgeModel ??= new ChatGroq({
    model: JUDGE_MODEL,
    temperature: 0,
    maxTokens: 4096,
    maxRetries: 8,
    cache: new FileLLMCache(),
  });
  return judgeModel;
}

export async function judgeAnswer(
  question: Question,
  answer: string,
  context: Document[]
): Promise<Judgement> {
  const contextText = context.length
    ? context.map((chunk, i) => `[${i + 1}] ${chunk.pageContent}`).join("\n\n")
    : "(no context was retrieved)";

  const prompt = `You are grading a question-answering system that must answer ONLY from a given document.

Question: ${question.question}

Reference answer (written by a human expert from the document): ${question.reference_answer}

System answer: ${answer}

Context the system was given:
${contextText}

Grade two things.

1. correctness, comparing the system answer with the reference answer:
   1 = contains the key facts of the reference and nothing that contradicts it
   0.5 = partially correct: some key facts missing, or partly wrong
   0 = wrong, contradicts the reference, or says the answer is not available when the reference has one
   If the reference says the document does not contain the answer, give 1 only when the system also says the information is not available and does not invent an answer; otherwise 0.

2. claims: list every factual claim the system answer makes, and mark whether the CONTEXT above supports it (not whether it is true in general). A statement that the information is not available is not a claim.

Return only JSON:
{"correctness": 0 | 0.5 | 1, "claims": [{"claim": "...", "supported": true | false}], "reasoning": "one sentence"}`;

  const response = await getJudge().invoke([new HumanMessage(prompt)]);
  const text = typeof response.content === "string" ? response.content : JSON.stringify(response.content);
  const json = text.match(/\{[\s\S]*\}/);
  if (!json) throw new Error(`Judge returned no JSON for ${question.id}: ${text.slice(0, 200)}`);

  const parsed = JSON.parse(json[0]);
  const claims: Judgement["claims"] = Array.isArray(parsed.claims) ? parsed.claims : [];
  return {
    correctness: [0, 0.5, 1].includes(Number(parsed.correctness)) ? Number(parsed.correctness) : 0,
    faithfulness: claims.length ? claims.filter((c) => c.supported).length / claims.length : null,
    claims,
    reasoning: String(parsed.reasoning || ""),
  };
}

// Small stats helpers

export const mean = (values: number[]) =>
  values.length ? values.reduce((a, b) => a + b, 0) / values.length : NaN;

export function percentile(values: number[], p: number): number {
  if (!values.length) return NaN;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.ceil((p / 100) * sorted.length) - 1)];
}
