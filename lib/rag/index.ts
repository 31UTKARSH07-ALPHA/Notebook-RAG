// Public API of the RAG pipeline. Everything outside lib/rag imports from
// "@/lib/rag" (or "../lib/rag"), so the files inside can be reorganised
// without touching the app, the evaluation harness or the tests.

export type {
  AnswerHooks,
  AnswerResult,
  ChatTurn,
  EmbeddingConfig,
  GradeMode,
  Grounding,
  PipelineOptions,
  PipelineStage,
  SourceChunk,
  StageListener,
} from "./types";
export { DEFAULT_PIPELINE, type KeyValueCache, configureRuntime } from "./config";
export { getChatModel } from "./llm";
export {
  DEFAULT_EMBEDDING,
  formatEmbedding,
  getEmbeddingConfig,
  parseEmbedding,
  rerank,
  resolveEmbedding,
} from "./cohere";
export { cleanPageText, parseUploadedFile } from "./parsing";
export {
  type ChunkingStrategy,
  DEFAULT_CHUNKING,
  FIXED_CHUNK_CHARS,
  FIXED_OVERLAP_CHARS,
  MAX_SENTENCE_CHARS,
  chunkDocuments,
  splitIntoSentences,
  splitLongSentence,
} from "./chunking";
export {
  type SparseVector,
  buildDocumentSparseVector,
  buildQuerySparseVector,
  hashToken,
  tokenize,
} from "./bm25";
export {
  createCollectionName,
  deleteCollection,
  deleteDocumentChunks,
  getCollectionEmbedding,
  getQdrantClient,
  indexChunks,
  isNotebookCollection,
} from "./store";
export { addContextHeaders } from "./context-headers";
export { capHistory, condenseQuestion } from "./chat";
export {
  decomposeQuestion,
  generateQueryVariants,
  hybridSearch,
  interleaveRankings,
  parseSubQuestions,
  retrieveChunks,
} from "./retrieval";
export {
  type ChunkGrade,
  DEFAULT_RERANK_THRESHOLD,
  compressContext,
  filterByRerankScore,
  gradeChunks,
  resolveGradeMode,
  scoreChunks,
} from "./postprocess";
export {
  NOT_FOUND_ANSWER,
  buildGroundedPrompt,
  formatSources,
  generateGroundedAnswer,
  gradeAnswer,
  isNotFoundAnswer,
  normaliseCitations,
} from "./generation";
export { answerQuestion } from "./pipeline";
