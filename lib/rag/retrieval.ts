// Retrieval: query rewriting and decomposition, hybrid search with RRF
// fusion, merging per-sub-question rankings, and retrieveChunks(), which runs
// search -> rerank -> relevance filter -> compression.

import { Document } from "@langchain/core/documents";
import { ChatGroq } from "@langchain/groq";
import { HumanMessage } from "@langchain/core/messages";
import { DEFAULT_PIPELINE, handleStageFailure, reportStage } from "./config";
import { getChatModel, messageText } from "./llm";
import { embed, rerank } from "./cohere";
import { buildQuerySparseVector } from "./bm25";
import {
  DENSE_VECTOR_NAME,
  SPARSE_VECTOR_NAME,
  getCollectionEmbedding,
  getQdrantClient,
} from "./store";
import { compressContext, filterByRerankScore, gradeChunks, resolveGradeMode } from "./postprocess";
import type { PipelineOptions, PipelineStage, StageListener } from "./types";

// Retrieval config
const PREFETCH_LIMIT = 20; // candidates per search path (dense/sparse x each query)
const RERANK_CANDIDATES = 20; // fused candidates sent to the reranker

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

  // Step 4: Relevance filter. Empty result means "not in the document".
  // The LLM grader (CRAG) costs a call; the rerank-score cut-off is free.
  const gradeMode = resolveGradeMode(options.grade);
  const relevant =
    gradeMode === "llm"
      ? await time("gradeChunks", () => gradeChunks(question, candidates, llm))
      : gradeMode === "rerank-threshold"
        ? filterByRerankScore(candidates, options.rerankThreshold)
        : candidates;
  if (relevant.length === 0) return { chunks: [], candidates, queries };

  // Step 5: Context compression
  const chunks = options.compress
    ? await time("compress", () => compressContext(question, relevant, llm))
    : relevant;
  return { chunks, candidates, queries };
}
