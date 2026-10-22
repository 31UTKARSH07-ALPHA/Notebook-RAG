// The full question-answering pipeline, answerQuestion(): condense the
// follow-up, retrieve, generate, check grounding, retry once.

import { DEFAULT_PIPELINE, reportStage } from "./config";
import { capHistory, condenseQuestion } from "./chat";
import { retrieveChunks } from "./retrieval";
import {
  NOT_FOUND_ANSWER,
  formatSources,
  generateGroundedAnswer,
  gradeAnswer,
  isNotFoundAnswer,
} from "./generation";
import type { AnswerHooks, AnswerResult, Grounding, PipelineOptions } from "./types";

// Full query pipeline: condense follow-up -> retrieve -> generate -> check
// grounding -> retry once.
//
// hooks.onStage reports progress as each stage starts. The answer itself is
// deliberately not streamed token by token: it is only final after the
// grounding check, and a draft that fails the check is thrown away and
// regenerated. Streaming tokens would show the user text that may then be
// replaced (including the unsupported claim the grader caught), so callers
// get stage updates while waiting and the answer once it has been checked.
export async function answerQuestion(
  collectionName: string,
  question: string,
  options: PipelineOptions = DEFAULT_PIPELINE,
  hooks: AnswerHooks = {}
): Promise<AnswerResult> {
  const { onStage } = hooks;
  const timingsMs: Record<string, number> = {};
  const started = Date.now();

  // Without history there is nothing to resolve, so no extra LLM call
  let standaloneQuestion = question;
  if (capHistory(hooks.history).length > 0) {
    reportStage(onStage, "condense");
    const start = Date.now();
    standaloneQuestion = await condenseQuestion(question, hooks.history ?? []);
    timingsMs.condense = Date.now() - start;
  }

  const { chunks, candidates, queries } = await retrieveChunks(
    collectionName,
    standaloneQuestion,
    options,
    timingsMs,
    onStage
  );

  const result = (answer: string, grounding: Grounding): AnswerResult => {
    timingsMs.total = Date.now() - started;
    return {
      answer,
      standaloneQuestion,
      sources: formatSources(chunks),
      grounding,
      queries,
      timingsMs,
      context: chunks,
      retrieved: candidates,
    };
  };

  if (chunks.length === 0) {
    return result(NOT_FOUND_ANSWER, {
      verdict: "not_found",
      confidence: 1,
      reasoning: "No retrieved chunk was relevant to the question.",
      retried: false,
    });
  }

  reportStage(onStage, "generate");
  let start = Date.now();
  let answer = await generateGroundedAnswer(chunks, standaloneQuestion);
  timingsMs.generate = Date.now() - start;

  if (isNotFoundAnswer(answer)) {
    return result(answer, {
      verdict: "not_found",
      confidence: 1,
      reasoning: "The model found no answer in the sources.",
      retried: false,
    });
  }

  if (!options.groundingCheck) {
    return result(answer, {
      verdict: "unverified",
      confidence: 0,
      reasoning: "Grounding check disabled.",
      retried: false,
    });
  }

  reportStage(onStage, "groundingCheck");
  start = Date.now();
  let grounding = await gradeAnswer(answer, chunks, standaloneQuestion);
  let retried = false;

  // One retry with the grader's feedback. More retries rarely help and double latency.
  if (grounding.verdict === "not_grounded") {
    retried = true;
    reportStage(onStage, "retry");
    answer = await generateGroundedAnswer(chunks, standaloneQuestion, grounding.reasoning);
    reportStage(onStage, "groundingCheck");
    grounding = await gradeAnswer(answer, chunks, standaloneQuestion);
  }
  timingsMs.groundingCheck = Date.now() - start;

  return result(answer, { ...grounding, retried });
}
