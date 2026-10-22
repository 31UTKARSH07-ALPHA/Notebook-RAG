// Answer generation: the grounded prompt, the answer call, citation and
// refusal handling, the hallucination grader, and formatting sources for the UI.

import { Document } from "@langchain/core/documents";
import { ChatGroq } from "@langchain/groq";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { handleStageFailure } from "./config";
import { getChatModel, messageText } from "./llm";
import type { Grounding, SourceChunk } from "./types";

export const NOT_FOUND_ANSWER = "I could not find that in the uploaded document.";

// Hallucination grader - check if answer is grounded in chunks
export async function gradeAnswer(
  answer: string,
  docs: Document[],
  question: string,
  llm?: ChatGroq
): Promise<Omit<Grounding, "retried">> {
  const model = llm || getChatModel();

  let prompt = `You are a hallucination grader for a RAG system.
Given a question, an answer, and the source chunks used to generate the answer, determine if the answer is fully grounded in the provided sources.

Question: "${question}"

Answer to evaluate: "${answer}"

Source chunks:
`;

  docs.forEach((doc, i) => {
    prompt += `\n\n--- Source ${i + 1} ---\n${doc.pageContent}\n--- End Source ${i + 1} ---\n`;
  });

  prompt += `\n\nIs the answer fully grounded in the sources? Answer with JSON:
{"grounded": true|false, "confidence": 0.0-1.0, "reasoning": "brief explanation"}`;

  try {
    const response = await model.invoke([new HumanMessage(prompt)]);
    const jsonMatch = messageText(response.content).match(/\{[\s\S]*\}/);
    if (jsonMatch) {
      const parsed = JSON.parse(jsonMatch[0]);
      return {
        verdict: parsed.grounded ? "grounded" : "not_grounded",
        confidence: Number(parsed.confidence) || 0,
        reasoning: String(parsed.reasoning || ""),
      };
    }
  } catch (error) {
    handleStageFailure("Grounding check", error);
  }

  return { verdict: "unverified", confidence: 0, reasoning: "Grader output could not be parsed" };
}

// Answer generation

export function formatSources(chunks: Document[]): SourceChunk[] {
  return chunks.map((chunk, index) => ({
    fileName: chunk.metadata.source || undefined,
    chunk: Number(chunk.metadata.chunk || index + 1),
    page:
      Number(
        chunk.metadata.loc?.pageNumber ||
          chunk.metadata.pdf?.pageNumber ||
          0
      ) || undefined,
    // Show child text in UI (concise), not the expanded parent
    text: (chunk.metadata.childText || chunk.pageContent)
      .replace(/\s+/g, " ")
      .slice(0, 420),
    rerankScore: chunk.metadata.rerankScore,
  }));
}

export function buildGroundedPrompt(chunks: Document[], feedback?: string): string {
  const context = chunks
    .map((chunk, index) => {
      const chunkNumber = chunk.metadata.chunk || index + 1;
      const page =
        chunk.metadata.loc?.pageNumber || chunk.metadata.pdf?.pageNumber;
      const file = chunk.metadata.source ? ` | ${chunk.metadata.source}` : "";
      return `SOURCE ${index + 1}${file} | chunk ${chunkNumber}${page ? ` | page ${page}` : ""}\n${chunk.pageContent}`;
    })
    .join("\n\n---\n\n");

  const retryNote = feedback
    ? `\nA previous draft was rejected because it contained claims the sources do not support: ${feedback}
State only facts that appear explicitly in the sources.\n`
    : "";

  return `You are a document question-answering assistant.

Answer using only the sources below. If the sources do not contain the answer, say: "${NOT_FOUND_ANSWER}"

Rules:
- Do not use outside knowledge.
- Be concise but complete.
- Cite the sources you used inline as [1], [2], using the SOURCE numbers below.
- Never invent facts, names, page numbers, or examples not present in the sources.
${retryNote}
Sources:
${context}`;
}

export async function generateGroundedAnswer(
  chunks: Document[],
  question: string,
  feedback?: string
): Promise<string> {
  const model = getChatModel();
  const response = await model.invoke([
    new SystemMessage(buildGroundedPrompt(chunks, feedback)),
    new HumanMessage(question),
  ]);

  return normaliseCitations(messageText(response.content)).trim() || NOT_FOUND_ANSWER;
}

// gpt-oss sometimes ignores the [n] instruction and writes its own markers,
// e.g. "【1†L1】" or "【SOURCE 2 | chunk 7】". Rewrite them to [n].
export function normaliseCitations(answer: string): string {
  return answer.replace(/【\s*(?:SOURCE\s*)?(\d+)[^】]*】/gi, "[$1]");
}

// The model sometimes drops the period or rewords the refusal slightly
export function isNotFoundAnswer(answer: string): boolean {
  return answer.toLowerCase().includes("could not find that in the uploaded document");
}
