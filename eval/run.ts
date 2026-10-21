// SourceSage evaluation CLI.
//
//   npm run eval -- retrieval [--chunking=fixed] [--headers]   ranking quality per pipeline config (no generation)
//   npm run eval -- generation                     answer quality, judged by a second LLM
//   npm run eval -- latency                        uncached end-to-end timings on a question subset
//   npm run eval -- report                         rebuild eval/RESULTS.md from eval/results/*.json
//
// Options: --configs=a,b  --questions=id1,id2  --limit=N
//
// Every config calls the same retrieveChunks()/answerQuestion() the app uses,
// with stages switched on or off through PipelineOptions.

import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from "fs";
import { join } from "path";
import {
  ChunkingStrategy,
  DEFAULT_PIPELINE,
  PipelineOptions,
  answerQuestion,
  configureRuntime,
  isNotFoundAnswer,
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
  judgeAnswer,
  loadChunks,
  loadDocuments,
  loadQuestions,
  mean,
  percentile,
  retrievalMetrics,
} from "./lib";

const RESULTS_DIR = join(EVAL_DIR, "results");

const RETRIEVAL_BASE: PipelineOptions = {
  ...DEFAULT_PIPELINE,
  grade: false,
  compress: false,
  groundingCheck: false,
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
  "hybrid+rerank": {
    ...DEFAULT_PIPELINE,
    multiQuery: false,
    grade: false,
    compress: false,
    groundingCheck: false,
  },
  full: DEFAULT_PIPELINE,
  "full+decompose": { ...DEFAULT_PIPELINE, decompose: true },
};

// Seconds to wait between questions in the latency suite so Groq's free-tier
// token-per-minute limit is never hit; otherwise retry back-off would be
// measured as pipeline latency.
const LATENCY_PAUSE_S: Record<string, number> = { naive: 10, "hybrid+rerank": 10, full: 60, "full+decompose": 60 };

type Args = {
  suite: string;
  chunking: ChunkingStrategy;
  contextHeaders: boolean;
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
    chunking: (flags.chunking as ChunkingStrategy) || "sentence-window",
    contextHeaders: "headers" in flags,
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
  const variant = indexVariant(chunking, contextHeaders);
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

function save(name: string, data: unknown) {
  mkdirSync(RESULTS_DIR, { recursive: true });
  writeFileSync(join(RESULTS_DIR, `${name}.json`), JSON.stringify(data, null, 2));
}

function useEvalRuntime(cached: boolean) {
  configureRuntime({
    strict: true,
    maxRetries: 8,
    llmCache: cached ? new FileLLMCache() : undefined,
    apiCache: cached ? apiCache : undefined,
  });
}

// Suites

async function runRetrieval(args: Args) {
  useEvalRuntime(true);
  const questions = selectQuestions(args).filter((q) => q.type !== "unanswerable");
  const { collections, chunkCounts } = await prepareCollections(questions, args);
  const configs: Record<string, unknown> = {};

  for (const [name, options] of selectConfigs(RETRIEVAL_CONFIGS, args)) {
    const rows = [];
    for (const [i, q] of questions.entries()) {
      const { candidates, queries } = await retrieveChunks(collections[q.doc], q.question, options);
      const metrics = retrievalMetrics(q, candidates);
      rows.push({
        id: q.id,
        type: q.type,
        ...metrics,
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
        recallByType: byType(rows, (s) => mean(s.map((r) => r.recall))),
        mrrByType: byType(rows, (s) => mean(s.map((r) => r.reciprocalRank))),
      },
      questions: rows,
    };
  }

  save(`retrieval-${indexVariant(args.chunking, args.contextHeaders)}`, {
    suite: "retrieval",
    chunking: indexVariant(args.chunking, args.contextHeaders),
    chunkCounts,
    questionCount: questions.length,
    generatedAt: new Date().toISOString(),
    configs,
  });
}

async function runGeneration(args: Args) {
  useEvalRuntime(true);
  const questions = selectQuestions(args);
  const { collections } = await prepareCollections(questions, args);
  const configs: Record<string, unknown> = {};

  for (const [name, options] of selectConfigs(GENERATION_CONFIGS, args)) {
    const rows = [];
    const errors = [];

    for (const [i, q] of questions.entries()) {
      try {
        const result = await answerQuestion(collections[q.doc], q.question, options);
        const abstained = isNotFoundAnswer(result.answer);
        const judgement = await judgeAnswer(q, result.answer, result.context);
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
      },
      questions: rows,
      errors,
    };
  }

  save(`generation-${indexVariant(args.chunking, args.contextHeaders)}`, {
    suite: "generation",
    chunking: indexVariant(args.chunking, args.contextHeaders),
    judgeModel: JUDGE_MODEL,
    questionCount: questions.length,
    generatedAt: new Date().toISOString(),
    configs,
  });
}

async function runLatency(args: Args) {
  useEvalRuntime(false); // real network calls only
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

  save(`latency-${indexVariant(args.chunking, args.contextHeaders)}`, {
    suite: "latency",
    chunking: indexVariant(args.chunking, args.contextHeaders),
    questions: questions.map((q) => q.id),
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
      table(
        ["Config", "Hit@5", "Recall@5", "MRR@5", "Recall factual", "Recall keyword", "Recall multi-hop"],
        Object.entries<any>(data.configs).map(([name, c]) => [
          name,
          pct(c.summary.hitAt5),
          pct(c.summary.recallAt5),
          num(c.summary.mrrAt5),
          pct(c.summary.recallByType.factual),
          pct(c.summary.recallByType.keyword),
          pct(c.summary.recallByType.multi_hop),
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
          name,
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

  writeFileSync(join(EVAL_DIR, "RESULTS.md"), sections.join("\n") + "\n");
  console.log(sections.join("\n"));
}

async function main() {
  const args = parseArgs();
  if (args.suite === "retrieval") await runRetrieval(args);
  else if (args.suite === "generation") await runGeneration(args);
  else if (args.suite === "latency") await runLatency(args);
  else if (args.suite !== "report") throw new Error(`Unknown suite "${args.suite}"`);
  buildReport();
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
