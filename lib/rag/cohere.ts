// Cohere API access: requests with retries and the response cache, the
// embedding models and embed(), and rerank().

import { Document } from "@langchain/core/documents";
import { handleStageFailure, requireEnv, runtimeConfig } from "./config";
import type { EmbeddingConfig } from "./types";

const EMBED_BATCH_SIZE = 96; // Cohere embed accepts at most 96 texts per request
const RERANK_MODEL = "rerank-v3.5";

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

// The model new collections are built with. Existing collections keep the
// model recorded in their metadata (see getCollectionEmbedding).
export function getEmbeddingConfig(): EmbeddingConfig {
  if (runtimeConfig.embedding) return runtimeConfig.embedding;
  const model = process.env.EMBEDDING_MODEL || DEFAULT_EMBEDDING.model;
  const dim = process.env.EMBEDDING_DIM;
  if (dim && !/^\d+$/.test(dim)) throw new Error(`EMBEDDING_DIM must be a number, got "${dim}".`);
  return resolveEmbedding(model, dim ? Number(dim) : undefined);
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
export async function embed(
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
