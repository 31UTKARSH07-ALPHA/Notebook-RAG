// Process-wide settings and helpers every stage uses: the default pipeline
// options, runtime config (strict mode, retries, caches), stage failure and
// progress reporting, and required environment variables.

import { BaseCache } from "@langchain/core/caches";
import type { EmbeddingConfig, PipelineOptions, PipelineStage, StageListener } from "./types";

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

export const runtimeConfig: RuntimeConfig = { strict: false, maxRetries: 2 };

export function configureRuntime(config: Partial<RuntimeConfig>): void {
  Object.assign(runtimeConfig, config);
}

export function handleStageFailure(stage: string, error: unknown): void {
  if (runtimeConfig.strict) throw error;
  console.error(`${stage} failed, using fallback:`, error);
}

// Progress is cosmetic: a listener that throws (e.g. its client disconnected)
// must never fail the answer itself.
export function reportStage(onStage: StageListener | undefined, stage: PipelineStage): void {
  try {
    onStage?.(stage);
  } catch (error) {
    console.warn(`Stage listener failed on "${stage}":`, error);
  }
}

export function requireEnv(name: string): string {
  const value = process.env[name];
  if (!value) {
    throw new Error(
      `${name} is not configured. Add it to your environment variables.`
    );
  }
  return value;
}
