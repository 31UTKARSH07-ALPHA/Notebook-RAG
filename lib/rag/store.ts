// Qdrant storage: the client, notebook collections (create, embedding
// metadata, delete) and indexing chunks as dense + BM25 vectors.

import { Document } from "@langchain/core/documents";
import { QdrantClient } from "@qdrant/js-client-rest";
import { randomUUID } from "crypto";
import { requireEnv } from "./config";
import {
  DEFAULT_EMBEDDING,
  embed,
  formatEmbedding,
  getEmbeddingConfig,
  resolveEmbedding,
} from "./cohere";
import { buildDocumentSparseVector, tokenize } from "./bm25";
import type { EmbeddingConfig } from "./types";

const DEFAULT_PREFIX = "sourcesage";
export const DENSE_VECTOR_NAME = "dense";
export const SPARSE_VECTOR_NAME = "sparse";

function getCollectionPrefix(): string {
  return process.env.QDRANT_COLLECTION_PREFIX || DEFAULT_PREFIX;
}

export function createCollectionName(): string {
  return `${getCollectionPrefix()}_${randomUUID().replace(/-/g, "_")}`;
}

export function getQdrantClient(): QdrantClient {
  return new QdrantClient({
    url: requireEnv("QDRANT_URL"),
    apiKey: process.env.QDRANT_API_KEY || undefined,
    checkCompatibility: false,
  });
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
