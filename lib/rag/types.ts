// Shared types: what the pipeline takes as options and returns, and the
// stage names it reports. Kept separate so every module can use them without
// importing each other.

import { Document } from "@langchain/core/documents";

// A dense embedding model and the vector size it produces (see cohere.ts)
export type EmbeddingConfig = { model: string; dim: number };

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
