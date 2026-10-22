import { beforeEach, describe, expect, it, vi } from "vitest";
import { Document } from "@langchain/core/documents";

// Offline stand-ins for Qdrant, Cohere and Groq, so retrieveChunks() and
// answerQuestion() run end to end without a network. Any LLM call fails the
// test: the rerank-threshold filter must not make one.
const scores: Record<string, number> = {};

vi.mock("../lib/rag/store", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../lib/rag/store")>()),
  getCollectionEmbedding: async () => ({ model: "embed-english-v3.0", dim: 2 }),
  getQdrantClient: () => ({
    query: async () => ({
      points: Object.keys(scores).map((id, i) => ({
        id,
        score: 1 / (i + 1),
        payload: { pageContent: `text ${id}`, metadata: { chunk: i + 1 } },
      })),
    }),
  }),
}));

vi.mock("../lib/rag/cohere", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../lib/rag/cohere")>()),
  embed: async (texts: string[]) => texts.map(() => [0, 1]),
  rerank: async (_question: string, docs: Document[], topK: number) =>
    docs
      .map((doc) => new Document({ id: doc.id, pageContent: doc.pageContent, metadata: { ...doc.metadata, rerankScore: scores[doc.id!] } }))
      .sort((a, b) => b.metadata.rerankScore - a.metadata.rerankScore)
      .slice(0, topK),
}));

const llmCalls = vi.fn();
vi.mock("../lib/rag/llm", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../lib/rag/llm")>()),
  getChatModel: () => ({
    invoke: async () => {
      llmCalls();
      throw new Error("unexpected LLM call");
    },
  }),
}));

const {
  DEFAULT_PIPELINE,
  DEFAULT_RERANK_THRESHOLD,
  NOT_FOUND_ANSWER,
  answerQuestion,
  configureRuntime,
  filterByRerankScore,
  resolveGradeMode,
  retrieveChunks,
} = await import("../lib/rag");
type PipelineOptions = import("../lib/rag").PipelineOptions;

const scored = (id: string, rerankScore?: number) =>
  new Document({ id, pageContent: id, metadata: rerankScore === undefined ? {} : { rerankScore } });
const ids = (docs: Document[]) => docs.map((d) => d.id);

describe("filterByRerankScore", () => {
  const docs = [scored("a", 0.9), scored("b", 0.3), scored("c", 0.29), scored("d", 0.01)];

  it("keeps chunks at or above the threshold, in order", () => {
    expect(ids(filterByRerankScore(docs, 0.3))).toEqual(["a", "b"]);
  });

  it("uses DEFAULT_RERANK_THRESHOLD when no threshold is given", () => {
    expect(ids(filterByRerankScore(docs))).toEqual(
      ids(docs.filter((d) => (d.metadata.rerankScore ?? 0) >= DEFAULT_RERANK_THRESHOLD))
    );
  });

  it("returns nothing when every chunk is below the threshold", () => {
    expect(filterByRerankScore(docs, 0.95)).toEqual([]);
  });

  it("keeps chunks without a rerank score (reranker off or failed)", () => {
    expect(ids(filterByRerankScore([scored("x"), scored("y", 0.1)], 0.3))).toEqual(["x"]);
  });

  it("keeps everything at threshold 0", () => {
    expect(filterByRerankScore(docs, 0)).toHaveLength(4);
  });
});

describe("resolveGradeMode", () => {
  it("maps the boolean form to the modes it always meant", () => {
    expect(resolveGradeMode(true)).toBe("llm");
    expect(resolveGradeMode(false)).toBe("off");
  });

  it("passes the named modes through", () => {
    expect(resolveGradeMode("llm")).toBe("llm");
    expect(resolveGradeMode("rerank-threshold")).toBe("rerank-threshold");
    expect(resolveGradeMode("off")).toBe("off");
  });

  it("leaves the app default on the LLM grader", () => {
    expect(resolveGradeMode(DEFAULT_PIPELINE.grade)).toBe("llm");
    expect(DEFAULT_PIPELINE.rerankThreshold).toBeUndefined();
  });
});

describe("retrieveChunks with grade: \"rerank-threshold\"", () => {
  const options: PipelineOptions = {
    search: "hybrid",
    multiQuery: false,
    decompose: false,
    rerank: true,
    grade: "rerank-threshold",
    compress: false,
    groundingCheck: false,
    topK: 5,
  };

  beforeEach(() => {
    process.env.GROQ_API_KEY ||= "test";
    configureRuntime({ strict: true });
    llmCalls.mockClear();
    for (const key of Object.keys(scores)) delete scores[key];
    Object.assign(scores, { a: 0.8, b: 0.5, c: 0.31, d: 0.2, e: 0.05 });
  });

  it("drops low-scoring chunks without any LLM call", async () => {
    const timings: Record<string, number> = {};
    const { chunks, candidates } = await retrieveChunks("col", "q", options, timings);
    expect(ids(candidates)).toEqual(["a", "b", "c", "d", "e"]);
    expect(ids(chunks)).toEqual(["a", "b", "c"]);
    expect(llmCalls).not.toHaveBeenCalled();
    expect(timings.gradeChunks).toBeUndefined();
  });

  it("honours a rerankThreshold override", async () => {
    const { chunks } = await retrieveChunks("col", "q", { ...options, rerankThreshold: 0.6 });
    expect(ids(chunks)).toEqual(["a"]);
  });

  it("keeps every chunk with grade off (false or \"off\")", async () => {
    for (const grade of [false, "off"] as const) {
      const { chunks } = await retrieveChunks("col", "q", { ...options, grade });
      expect(chunks).toHaveLength(5);
    }
    expect(llmCalls).not.toHaveBeenCalled();
  });

  it("still calls the LLM grader for grade: true", async () => {
    // The mocked model throws, and strict mode surfaces it
    await expect(retrieveChunks("col", "q", { ...options, grade: true })).rejects.toThrow("unexpected LLM call");
    expect(llmCalls).toHaveBeenCalledTimes(1);
  });

  it("answers \"not found\" without generating when no chunk passes", async () => {
    const result = await answerQuestion("col", "q", { ...options, rerankThreshold: 0.95 });
    expect(result.answer).toBe(NOT_FOUND_ANSWER);
    expect(result.grounding.verdict).toBe("not_found");
    expect(result.context).toEqual([]);
    expect(result.retrieved).toHaveLength(5);
    expect(llmCalls).not.toHaveBeenCalled();
  });
});
