// SourceSage evaluation CLI.
//
//   npm run eval -- retrieval [--chunking=fixed] [--headers] [--embedding=embed-v4.0@512]
//                                                  ranking quality per pipeline config (no generation)
//   npm run eval -- generation                     answer quality, judged by a second LLM
//   npm run eval -- latency                        uncached end-to-end timings on a question subset
//   npm run eval -- filter                         rerank-score threshold sweep over the saved
//                                                  retrieval results (no API calls)
//   npm run eval -- report                         rebuild eval/RESULTS.md from eval/results/*.json
//
// Options: --configs=a,b  --questions=id1,id2  --limit=N
// --chunking, --headers and --embedding pick the index, and work with every suite.
//
// Every config calls the same retrieveChunks()/answerQuestion() the app uses,
// with stages switched on or off through PipelineOptions.

import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from "fs";
import { join } from "path";
import {
  ChunkingStrategy,
  DEFAULT_CHUNKING,
  DEFAULT_EMBEDDING,
  DEFAULT_PIPELINE,
  DEFAULT_RERANK_THRESHOLD,
  EmbeddingConfig,
  PipelineOptions,
  answerQuestion,
  configureRuntime,
  formatEmbedding,
  isNotFoundAnswer,
  parseEmbedding,
  retrieveChunks,
} from "../lib/rag";
import {
  EVAL_DIR,
  FileLLMCache,
  JUDGE_MODEL,
  Question,
  QuestionType,
  apiCache,
  chunkContainsEvidence,
  ensureIndexed,
  indexVariant,
  indexedText,
  llmText,
  judgeAnswer,
  loadChunks,
  loadDocuments,
  loadQuestions,
  mean,
  percentile,
  retrievalMetrics,
  FilterQuestion,
  sweepThresholds,
} from "./lib";

const RESULTS_DIR = join(EVAL_DIR, "results");

// Every config spells out all options rather than spreading DEFAULT_PIPELINE:
// otherwise changing an app default silently changes what an experiment
// measures (this happened once when decomposition became the default).
const RETRIEVAL_BASE: PipelineOptions = {
  search: "hybrid",
  multiQuery: false,
  decompose: false,
  rerank: false,
  grade: false,
  compress: false,
  groundingCheck: false,
  topK: 5,
};

const RETRIEVAL_CONFIGS: Record<string, PipelineOptions> = {
  dense: { ...RETRIEVAL_BASE, search: "dense", multiQuery: false, rerank: false },
  bm25: { ...RETRIEVAL_BASE, search: "sparse", multiQuery: false, rerank: false },
  hybrid: { ...RETRIEVAL_BASE, search: "hybrid", multiQuery: false, rerank: false },
  "hybrid+multiquery": { ...RETRIEVAL_BASE, search: "hybrid", multiQuery: true, rerank: false },
  "hybrid+rerank": { ...RETRIEVAL_BASE, search: "hybrid", multiQuery: false, rerank: true },
  "hybrid+multiquery+rerank": { ...RETRIEVAL_BASE, search: "hybrid", multiQuery: true, rerank: true },
  // Decomposition targets multi-hop recall; rewrites only apply to questions it leaves whole
  "hybrid+rerank+decompose": { ...RETRIEVAL_BASE, search: "hybrid", multiQuery: false, rerank: true, decompose: true },
  "hybrid+multiquery+rerank+decompose": { ...RETRIEVAL_BASE, search: "hybrid", multiQuery: true, rerank: true, decompose: true },
};

const GENERATION_CONFIGS: Record<string, PipelineOptions> = {
  // Plain top-5 vector search straight into the generator
  naive: {
    search: "dense",
    multiQuery: false,
    decompose: false,
    rerank: false,
    grade: false,
    compress: false,
    groundingCheck: false,
    topK: 5,
  },
  // Strong retrieval, no LLM post-processing
  "hybrid+rerank": { ...RETRIEVAL_BASE, rerank: true },
  // The app's current default, whatever it is (recorded in the results file)
  full: DEFAULT_PIPELINE,
  // The previous default: multi-query rewrites instead of decomposition
  "full-multiquery": {
    search: "hybrid",
    multiQuery: true,
    decompose: false,
    rerank: true,
    grade: true,
    compress: true,
    groundingCheck: true,
    topK: 5,
  },
  // Cheaper alternatives to the three post-retrieval LLM checks (DECISIONS.md
  // entry 28). All three use the current default retrieval (hybrid + rerank +
  // decomposition), spelled out. The threshold is the value the filter suite
  // picked, written as a number so a later change to the app constant can't
  // silently change this experiment.
  // The LLM grader replaced by the free rerank-score cut-off; rest as full
  "full-threshold": {
    search: "hybrid",
    multiQuery: false,
    decompose: true,
    rerank: true,
    grade: "rerank-threshold",
    rerankThreshold: 0.3,
    compress: true,
    groundingCheck: true,
    topK: 5,
  },
  // Full without compression: the generator gets the whole chunks
  "full-no-compress": {
    search: "hybrid",
    multiQuery: false,
    decompose: true,
    rerank: true,
    grade: "llm",
    compress: false,
    groundingCheck: true,
    topK: 5,
  },
  // Only free checks: rerank-score cut-off, no compression, no grounding
  // check. After retrieval the answer is the only LLM call (full makes 4+).
  "lean-threshold": {
    search: "hybrid",
    multiQuery: false,
    decompose: true,
    rerank: true,
    grade: "rerank-threshold",
    rerankThreshold: 0.3,
    compress: false,
    groundingCheck: false,
    topK: 5,
  },
};

// Seconds to wait between questions in the latency suite so Groq's free-tier
// token-per-minute limit is never hit; otherwise retry back-off would be
// measured as pipeline latency.
const LATENCY_PAUSE_S: Record<string, number> = {
  naive: 10,
  "hybrid+rerank": 10,
  full: 60,
  "full-multiquery": 60,
  "full-threshold": 45,
  "full-no-compress": 45,
  "lean-threshold": 15,
};

type Args = {
  suite: string;
  chunking: ChunkingStrategy;
  contextHeaders: boolean;
  embedding: EmbeddingConfig;
  configs?: string[];
  questionIds?: string[];
  limit?: number;
};

function parseArgs(): Args {
  const [suite = "retrieval", ...rest] = process.argv.slice(2);
  const flags = Object.fromEntries(
    rest.map((arg) => arg.replace(/^--/, "").split("=") as [string, string])
  );
  return {
    suite,
    chunking: (flags.chunking as ChunkingStrategy) || DEFAULT_CHUNKING,
    contextHeaders: "headers" in flags,
    // Not read from EMBEDDING_MODEL: a run without the flag is always the
    // baseline, whatever .env.local says, so result files stay comparable.
    embedding: flags.embedding ? parseEmbedding(flags.embedding) : DEFAULT_EMBEDDING,
    configs: flags.configs?.split(","),
    questionIds: flags.questions?.split(","),
    limit: flags.limit ? Number(flags.limit) : undefined,
  };
}

function selectQuestions(args: Args): Question[] {
  let questions = loadQuestions();
  if (args.questionIds) questions = questions.filter((q) => args.questionIds!.includes(q.id));
  if (args.limit) questions = questions.slice(0, args.limit);
  return questions;
}

function selectConfigs(all: Record<string, PipelineOptions>, args: Args) {
  const names = args.configs ?? Object.keys(all);
  for (const name of names) if (!all[name]) throw new Error(`Unknown config "${name}"`);
  return names.map((name) => [name, all[name]] as const);
}

// Indexes each document once per chunking strategy and checks that every
// evidence quote is findable in the chunks; a quote that matches no chunk
// would make that question impossible to score.
async function prepareCollections(questions: Question[], args: Args) {
  const { chunking, contextHeaders } = args;
  const variant = indexVariant(chunking, contextHeaders, args.embedding);
  const collections: Record<string, string> = {};
  const chunkCounts: Record<string, number> = {};

  for (const doc of loadDocuments()) {
    if (!questions.some((q) => q.doc === doc.id)) continue;
    const chunks = await loadChunks(doc, chunking, contextHeaders);
    const texts = chunks.map(indexedText);

    for (const q of questions.filter((q) => q.doc === doc.id)) {
      for (const quote of q.evidence) {
        if (!texts.some((t) => chunkContainsEvidence(t, quote))) {
          throw new Error(`${q.id}: evidence not found in any ${chunking} chunk: "${quote}"`);
        }
      }
    }

    collections[doc.id] = await ensureIndexed(doc, variant, chunks);
    chunkCounts[doc.id] = chunks.length;
    console.log(`Indexed ${doc.id} (${variant}): ${chunks.length} chunks -> ${collections[doc.id]}`);
  }

  return { collections, chunkCounts };
}

const ANSWERABLE: QuestionType[] = ["factual", "keyword", "multi_hop"];

function byType<T>(rows: Array<T & { type: QuestionType }>, metric: (rows: T[]) => number) {
  const result: Record<string, number> = {};
  for (const type of [...ANSWERABLE, "unanswerable"] as QuestionType[]) {
    const subset = rows.filter((r) => r.type === type);
    if (subset.length) result[type] = metric(subset);
  }
  return result;
}

// Running a subset with --configs must not throw away the other configs'
// results, so configs are merged into the existing file for the same suite.
function save(name: string, data: { configs?: Record<string, unknown>; [key: string]: unknown }) {
  mkdirSync(RESULTS_DIR, { recursive: true });
  const path = join(RESULTS_DIR, `${name}.json`);
  const previous = existsSync(path) ? JSON.parse(readFileSync(path, "utf8")) : null;
  const merged = previous?.configs
    ? { ...data, configs: { ...previous.configs, ...data.configs } }
    : data;
  writeFileSync(path, JSON.stringify(merged, null, 2));
}

function useEvalRuntime(cached: boolean, args: Args) {
  configureRuntime({
    strict: true,
    embedding: args.embedding,
    maxRetries: 8,
    llmCache: cached ? new FileLLMCache() : undefined,
    apiCache: cached ? apiCache : undefined,
  });
}

// Suites

async function runRetrieval(args: Args) {
  useEvalRuntime(true, args);
  const questions = selectQuestions(args).filter((q) => q.type !== "unanswerable");
  const { collections, chunkCounts } = await prepareCollections(questions, args);
  const configs: Record<string, unknown> = {};

  for (const [name, options] of selectConfigs(RETRIEVAL_CONFIGS, args)) {
    const rows = [];
    for (const [i, q] of questions.entries()) {
      const { candidates, queries } = await retrieveChunks(collections[q.doc], q.question, options);
      const metrics = retrievalMetrics(q, candidates);
      const seen = retrievalMetrics(q, candidates, llmText);
      rows.push({
        id: q.id,
        type: q.type,
        ...metrics,
        llmHit: seen.hit,
        llmRecall: seen.recall,
        // Size of the context the generator would get: the "budget" behind the recall
        contextChars: candidates.reduce((sum, c) => sum + llmText(c).length, 0),
        queries,
        retrieved: candidates.map((c) => ({
          chunk: c.metadata.chunk,
          page: c.metadata.loc?.pageNumber,
          rerankScore: c.metadata.rerankScore,
          text: indexedText(c).slice(0, 200),
        })),
      });
      console.log(`[${name}] ${i + 1}/${questions.length} ${q.id} hit=${metrics.hit} rr=${metrics.reciprocalRank.toFixed(2)}`);
    }

    configs[name] = {
      options,
      summary: {
        hitAt5: mean(rows.map((r) => r.hit)),
        recallAt5: mean(rows.map((r) => r.recall)),
        mrrAt5: mean(rows.map((r) => r.reciprocalRank)),
        llmHitAt5: mean(rows.map((r) => r.llmHit)),
        llmRecallAt5: mean(rows.map((r) => r.llmRecall)),
        llmRecallMultiHop: mean(rows.filter((r) => r.type === "multi_hop").map((r) => r.llmRecall)),
        meanContextChars: mean(rows.map((r) => r.contextChars)),
        recallByType: byType(rows, (s) => mean(s.map((r) => r.recall))),
        mrrByType: byType(rows, (s) => mean(s.map((r) => r.reciprocalRank))),
      },
      questions: rows,
    };
  }

  save(`retrieval-${indexVariant(args.chunking, args.contextHeaders, args.embedding)}`, {
    suite: "retrieval",
    chunking: indexVariant(args.chunking, args.contextHeaders, args.embedding),
    embedding: formatEmbedding(args.embedding),
    chunkCounts,
    questionCount: questions.length,
    generatedAt: new Date().toISOString(),
    configs,
  });
}

async function runGeneration(args: Args) {
  useEvalRuntime(true, args);
  const questions = selectQuestions(args);
  const { collections } = await prepareCollections(questions, args);
  const configs: Record<string, unknown> = {};
  let dailyLimitHit = false;

  for (const [name, options] of selectConfigs(GENERATION_CONFIGS, args)) {
    const rows = [];
    const errors = [];

    for (const [i, q] of questions.entries()) {
      try {
        const result = await answerQuestion(collections[q.doc], q.question, options);
        const abstained = isNotFoundAnswer(result.answer);
        // A plain refusal needs no LLM to grade: it's right exactly when the
        // question is unanswerable, and it makes no claims to check. Spot
        // checks found the judge getting both wrong (scoring a correct refusal
        // 0, and counting the refusal sentence as an unsupported claim).
        const judgement = abstained
          ? {
              correctness: q.type === "unanswerable" ? 1 : 0,
              faithfulness: null,
              claims: [],
              reasoning: "Scored without the judge: the system said the answer is not in the document.",
            }
          : await judgeAnswer(q, result.answer, result.context);
        const contextTexts = result.context.map(indexedText);

        rows.push({
          id: q.id,
          type: q.type,
          question: q.question,
          answer: result.answer,
          abstained,
          verdict: result.grounding.verdict,
          retried: result.grounding.retried,
          ...judgement,
          // Share of evidence quotes that survived into the LLM's context
          contextRecall: q.evidence.length
            ? q.evidence.filter((e) => contextTexts.some((t) => chunkContainsEvidence(t, e))).length /
              q.evidence.length
            : null,
          retrieval: q.evidence.length ? retrievalMetrics(q, result.retrieved) : null,
        });
        console.log(
          `[${name}] ${i + 1}/${questions.length} ${q.id} correctness=${judgement.correctness}` +
            ` faithfulness=${judgement.faithfulness?.toFixed(2) ?? "-"}${abstained ? " (abstained)" : ""}`
        );
      } catch (error) {
        // Out of daily tokens: every remaining question would fail too. Stop,
        // keep what's done; a re-run replays it from cache and carries on.
        if (isDailyLimit(error)) {
          dailyLimitHit = true;
          console.error(`[${name}] daily token limit reached after ${rows.length} questions; stopping.`);
          break;
        }
        errors.push({ id: q.id, error: String(error) });
        console.error(`[${name}] ${q.id} FAILED:`, error);
      }
    }

    const answerable = rows.filter((r) => r.type !== "unanswerable");
    const unanswerable = rows.filter((r) => r.type === "unanswerable");
    const faithful = rows.map((r) => r.faithfulness).filter((f): f is number => f !== null);

    configs[name] = {
      options,
      summary: {
        correctness: mean(rows.map((r) => r.correctness)),
        correctnessAnswerable: mean(answerable.map((r) => r.correctness)),
        correctnessByType: byType(rows, (s) => mean(s.map((r) => r.correctness))),
        faithfulness: mean(faithful),
        fullyFaithfulRate: mean(faithful.map((f) => (f === 1 ? 1 : 0))),
        correctAbstentionRate: mean(unanswerable.map((r) => r.correctness)),
        falseRefusalRate: mean(answerable.map((r) => (r.abstained ? 1 : 0))),
        contextRecall: mean(answerable.map((r) => r.contextRecall ?? 0)),
        retries: rows.filter((r) => r.retried).length,
        errors: errors.length,
        answered: rows.length,
      },
      questions: rows,
      errors,
    };

    // Saved after every config so a stopped run keeps the finished ones
    saveGeneration(args, questions.length, configs);
    if (dailyLimitHit) break;
  }
}

function saveGeneration(args: Args, questionCount: number, configs: Record<string, unknown>) {
  save(`generation-${indexVariant(args.chunking, args.contextHeaders, args.embedding)}`, {
    suite: "generation",
    chunking: indexVariant(args.chunking, args.contextHeaders, args.embedding),
    embedding: formatEmbedding(args.embedding),
    judgeModel: JUDGE_MODEL,
    questionCount,
    generatedAt: new Date().toISOString(),
    configs,
  });
}

function isDailyLimit(error: unknown): boolean {
  return /tokens per day|\(TPD\)/i.test(String(error));
}

async function runLatency(args: Args) {
  useEvalRuntime(false, args); // real network calls only
  // A fixed spread of question types: every 6th question
  const questions = args.questionIds || args.limit
    ? selectQuestions(args)
    : loadQuestions().filter((_, i) => i % 6 === 0);
  const { collections } = await prepareCollections(questions, args);
  const configs: Record<string, unknown> = {};

  for (const [name, options] of selectConfigs(GENERATION_CONFIGS, args)) {
    const timings: Array<Record<string, number>> = [];
    for (const [i, q] of questions.entries()) {
      const result = await answerQuestion(collections[q.doc], q.question, options);
      timings.push(result.timingsMs);
      console.log(`[${name}] ${i + 1}/${questions.length} ${q.id} ${result.timingsMs.total} ms`);
      await new Promise((r) => setTimeout(r, (LATENCY_PAUSE_S[name] ?? 10) * 1000));
    }

    const stages = Array.from(new Set(timings.flatMap((t) => Object.keys(t)))).filter((s) => s !== "total");
    configs[name] = {
      p50: percentile(timings.map((t) => t.total), 50),
      p90: percentile(timings.map((t) => t.total), 90),
      stageMeans: Object.fromEntries(stages.map((s) => [s, mean(timings.map((t) => t[s] ?? 0))])),
      timings,
    };
  }

  save(`latency-${indexVariant(args.chunking, args.contextHeaders, args.embedding)}`, {
    suite: "latency",
    chunking: indexVariant(args.chunking, args.contextHeaders, args.embedding),
    embedding: formatEmbedding(args.embedding),
    questions: questions.map((q) => q.id),
    generatedAt: new Date().toISOString(),
    configs,
  });
}

// Filter: picks the rerank-score cut-off for grade: "rerank-threshold" from
// data already on disk, with no API call at all. It reads the reranked top-5
// (with rerank scores) that the retrieval suite saved per question, looks up
// each chunk's full text by its chunk number (the saved text is cut to 200
// characters, too little to find the evidence quotes in; the prefix is
// checked to make sure the chunk number still points at the same text),
// labels the chunk relevant iff it contains an evidence quote, and sweeps
// thresholds. Re-running retrieval from the cache would not work: Qdrant
// orders tied RRF scores differently between runs, so most rerank requests
// would no longer match their cached copies.

const FILTER_SOURCE_CONFIGS = ["hybrid+rerank", "hybrid+rerank+decompose"];
const FILTER_THRESHOLDS = Array.from({ length: 13 }, (_, i) => (i * 5) / 100); // 0.00-0.60

async function runFilter(args: Args) {
  const variant = indexVariant(args.chunking, args.contextHeaders, args.embedding);
  const path = join(RESULTS_DIR, `retrieval-${variant}.json`);
  if (!existsSync(path)) throw new Error(`Run the retrieval suite first: ${path} not found`);
  const retrieval = JSON.parse(readFileSync(path, "utf8"));
  const questions = new Map(selectQuestions(args).map((q) => [q.id, q]));

  // Full chunk text by document and chunk number (local parsing only)
  const chunkText = new Map<string, string>();
  for (const doc of loadDocuments()) {
    for (const chunk of await loadChunks(doc, args.chunking, args.contextHeaders)) {
      chunkText.set(`${doc.id}:${chunk.metadata.chunk}`, indexedText(chunk));
    }
  }

  const configs: Record<string, unknown> = {};
  for (const name of args.configs ?? FILTER_SOURCE_CONFIGS) {
    const saved = retrieval.configs[name];
    if (!saved?.options?.rerank) throw new Error(`Filter suite needs a reranked config in ${path}, got "${name}"`);

    const rows: FilterQuestion[] = [];
    for (const row of saved.questions) {
      const q = questions.get(row.id);
      if (!q) continue;
      rows.push({
        id: q.id,
        type: q.type,
        chunks: row.retrieved.map((c: { chunk: number; rerankScore: number; text: string }) => {
          const text = chunkText.get(`${q.doc}:${c.chunk}`);
          if (!text || !text.startsWith(c.text)) {
            throw new Error(`${q.id}: chunk ${c.chunk} no longer matches the saved text; re-run the retrieval suite`);
          }
          if (typeof c.rerankScore !== "number") throw new Error(`${q.id}: chunk ${c.chunk} has no rerank score`);
          return {
            rerankScore: c.rerankScore,
            relevant: q.evidence.some((quote) => chunkContainsEvidence(text, quote)),
          };
        }),
      });
    }

    configs[name] = {
      options: saved.options,
      scored: {
        answerable: rows.filter((r) => r.type !== "unanswerable").length,
        unanswerable: rows.filter((r) => r.type === "unanswerable").length,
        chunks: rows.reduce((n, r) => n + r.chunks.length, 0),
        relevantChunks: rows.reduce((n, r) => n + r.chunks.filter((c) => c.relevant).length, 0),
      },
      sweep: sweepThresholds(rows, FILTER_THRESHOLDS),
      questions: rows,
    };
  }

  save(`filter-${variant}`, {
    suite: "filter",
    chunking: variant,
    source: `retrieval-${variant}.json (generated ${retrieval.generatedAt})`,
    appThreshold: DEFAULT_RERANK_THRESHOLD,
    generatedAt: new Date().toISOString(),
    configs,
  });
}

// Report

const pct = (v: number | undefined) => (v === undefined || Number.isNaN(v) ? "–" : `${(v * 100).toFixed(1)}%`);
const num = (v: number | undefined) => (v === undefined || Number.isNaN(v) ? "–" : v.toFixed(3));
const secs = (ms: number | undefined) => (ms === undefined || Number.isNaN(ms) ? "–" : `${(ms / 1000).toFixed(1)} s`);

function table(headers: string[], rows: string[][]): string {
  return [
    `| ${headers.join(" | ")} |`,
    `|${headers.map((_, i) => (i === 0 ? "---" : "---:")).join("|")}|`,
    ...rows.map((r) => `| ${r.join(" | ")} |`),
  ].join("\n");
}

function buildReport() {
  const read = (name: string) => {
    const path = join(RESULTS_DIR, `${name}.json`);
    return existsSync(path) ? JSON.parse(readFileSync(path, "utf8")) : null;
  };
  const files = existsSync(RESULTS_DIR) ? readdirSync(RESULTS_DIR) : [];
  const sections: string[] = [
    "# SourceSage evaluation results",
    "",
    "Generated by `npm run eval -- report` from `eval/results/*.json`. See `eval/README.md` for the method.",
  ];

  for (const file of files.filter((f) => f.startsWith("retrieval-")).sort().reverse()) {
    const data = read(file.replace(".json", ""));
    sections.push(
      "",
      `## Retrieval: ${data.chunking} chunking (${data.questionCount} answerable questions)`,
      "",
      `Chunks: ${Object.entries(data.chunkCounts).map(([d, n]) => `${d} ${n}`).join(", ")}.`,
      "",
      "Searched-unit columns score the chunk that was searched; LLM-context columns score the text the generator receives (the parent window for sentence-window), with its average size.",
      "",
      table(
        ["Config", "Hit@5", "Recall@5", "MRR@5", "Recall factual", "Recall keyword", "Recall multi-hop", "LLM-context recall", "LLM-context multi-hop", "Context size"],
        Object.entries<any>(data.configs).map(([name, c]) => [
          name,
          pct(c.summary.hitAt5),
          pct(c.summary.recallAt5),
          num(c.summary.mrrAt5),
          pct(c.summary.recallByType.factual),
          pct(c.summary.recallByType.keyword),
          pct(c.summary.recallByType.multi_hop),
          pct(c.summary.llmRecallAt5),
          pct(c.summary.llmRecallMultiHop),
          c.summary.meanContextChars ? `${Math.round(c.summary.meanContextChars)} chars` : "–",
        ])
      )
    );
  }

  for (const file of files.filter((f) => f.startsWith("generation-")).sort().reverse()) {
    const data = read(file.replace(".json", ""));
    sections.push(
      "",
      `## Generation: ${data.chunking} chunking (${data.questionCount} questions, judge: ${data.judgeModel})`,
      "",
      table(
        ["Config", "Correctness (all)", "Answerable", "Multi-hop", "Correct abstention", "False refusal", "Faithfulness", "Fully faithful", "Context recall", "Errors"],
        Object.entries<any>(data.configs).map(([name, c]) => [
          c.summary.answered !== undefined && c.summary.answered < data.questionCount
            ? `${name} (partial: ${c.summary.answered}/${data.questionCount})`
            : name,
          pct(c.summary.correctness),
          pct(c.summary.correctnessAnswerable),
          pct(c.summary.correctnessByType.multi_hop),
          pct(c.summary.correctAbstentionRate),
          pct(c.summary.falseRefusalRate),
          pct(c.summary.faithfulness),
          pct(c.summary.fullyFaithfulRate),
          pct(c.summary.contextRecall),
          String(c.summary.errors),
        ])
      )
    );
  }

  for (const file of files.filter((f) => f.startsWith("latency-"))) {
    const data = read(file.replace(".json", ""));
    const stageNames = Array.from(
      new Set(Object.values<any>(data.configs).flatMap((c) => Object.keys(c.stageMeans)))
    );
    sections.push(
      "",
      `## Latency (${data.questions.length} questions, uncached)`,
      "",
      table(
        ["Config", "p50", "p90", ...stageNames.map((s) => `${s} (mean)`)],
        Object.entries<any>(data.configs).map(([name, c]) => [
          name,
          secs(c.p50),
          secs(c.p90),
          ...stageNames.map((s) => secs(c.stageMeans[s])),
        ])
      )
    );
  }

  for (const file of files.filter((f) => f.startsWith("filter-"))) {
    const data = read(file.replace(".json", ""));
    for (const [name, c] of Object.entries<any>(data.configs)) {
      // The retrieval suite skips unanswerable questions, so usually there are none
      const withUnanswerable = c.scored.unanswerable > 0;
      sections.push(
        "",
        `## Relevance filter: rerank-score threshold sweep (${data.chunking} chunking, ${name})`,
        "",
        `From the saved top-5 of ${c.scored.answerable} answerable` +
          (withUnanswerable ? ` and ${c.scored.unanswerable} unanswerable` : "") +
          ` questions: ${c.scored.chunks} chunks, ${c.scored.relevantChunks} of them relevant (contain an evidence quote).` +
          ` The app's \`DEFAULT_RERANK_THRESHOLD\` is ${data.appThreshold}.`,
        "",
        "\"Emptied\": every chunk is dropped, so the pipeline answers \"not found\" without generating. \"Evidence lost\": the question had evidence in the top 5 and every evidence chunk is dropped (emptied or not).",
        "",
        table(
          [
            "Threshold",
            "Relevant kept",
            "Irrelevant dropped",
            "Chunks kept / question",
            "Answerable emptied",
            "Answerable evidence lost",
            ...(withUnanswerable ? ["Unanswerable emptied"] : []),
          ],
          c.sweep.map((r: any) => [
            r.threshold.toFixed(2),
            pct(r.relevantKept),
            pct(r.irrelevantDropped),
            r.meanChunksKept.toFixed(2),
            `${r.emptiedAnswerable}/${c.scored.answerable}`,
            `${r.evidenceLostAnswerable}/${c.scored.answerable}`,
            ...(withUnanswerable ? [`${r.emptiedUnanswerable}/${c.scored.unanswerable}`] : []),
          ])
        )
      );
    }
  }

  writeFileSync(join(EVAL_DIR, "RESULTS.md"), sections.join("\n") + "\n");
  console.log(sections.join("\n"));
}

async function main() {
  const args = parseArgs();
  if (args.suite === "retrieval") await runRetrieval(args);
  else if (args.suite === "generation") await runGeneration(args);
  else if (args.suite === "latency") await runLatency(args);
  else if (args.suite === "filter") await runFilter(args);
  else if (args.suite !== "report") throw new Error(`Unknown suite "${args.suite}"`);
  buildReport();
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
