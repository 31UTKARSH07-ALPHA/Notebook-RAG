// Post-retrieval: relevance filtering of the retrieved chunks (LLM grading as
// in CRAG, or a free cut-off on the rerank score) and context compression
// down to the relevant sentences.

import { Document } from "@langchain/core/documents";
import { ChatGroq } from "@langchain/groq";
import { HumanMessage } from "@langchain/core/messages";
import { handleStageFailure } from "./config";
import { getChatModel, messageText } from "./llm";
import type { GradeMode, PipelineOptions } from "./types";

// Post-retrieval

// Maps the `grade` option to a mode. Booleans are the pre-mode form of the
// option: true has always meant the LLM grader.
export function resolveGradeMode(grade: PipelineOptions["grade"]): GradeMode {
  if (grade === true) return "llm";
  if (grade === false) return "off";
  return grade;
}

// Rerank scores (rerank-v3.5, 0-1) below this are treated as off-topic.
// Picked offline with `npm run eval -- filter`, which sweeps cut-offs over the
// saved top-5 of the retrieval eval (DECISIONS.md, entry 28). At 0.30 no
// answerable question loses all of its evidence, ~98.5% of evidence chunks
// are kept and ~24% of the others are dropped. The weakest question's best
// evidence chunk scored 0.374, so from 0.40 up questions start losing all
// their evidence. The margin below that is deliberate: the sweep covered
// only two documents.
export const DEFAULT_RERANK_THRESHOLD = 0.3;

// The no-LLM relevance filter: keeps chunks whose Cohere rerank score
// (0-1) is at least the threshold. A chunk without a score (reranking off,
// or the reranker failed and fell back to fusion order) is kept: there is
// no evidence against it, just as a failed LLM grade keeps the chunk.
export function filterByRerankScore(
  docs: Document[],
  threshold: number = DEFAULT_RERANK_THRESHOLD
): Document[] {
  return docs.filter((doc) => {
    const score = doc.metadata.rerankScore;
    return typeof score !== "number" || score >= threshold;
  });
}

export type ChunkGrade = { index: number; relevant: boolean; confidence: number };

// LLM scores every chunk in one call. Chunks missing from the reply (bad JSON)
// are marked relevant so a grader failure never hides real context.
export async function scoreChunks(
  question: string,
  docs: Document[],
  llm?: ChatGroq
): Promise<ChunkGrade[]> {
  if (docs.length === 0) return [];
  const model = llm || getChatModel();

  let prompt = `You are a relevance grader for a RAG system.
Given a question and a document chunk, determine if the chunk is relevant to answering the question.

Question: "${question}"

For each chunk below, output "RELEVANT" or "NOT_RELEVANT" followed by a confidence score (0.0-1.0).

Format your response as JSON:
[
  {"index": 0, "label": "RELEVANT|NOT_RELEVANT", "confidence": 0.95},
  {"index": 1, "label": "RELEVANT|NOT_RELEVANT", "confidence": 0.72},
  ...
]

Document chunks:
`;

  docs.forEach((doc, i) => {
    prompt += `\n\n--- Chunk ${i} ---\n${doc.pageContent}\n--- End Chunk ${i} ---\n`;
  });

  prompt += `\n\nOutput JSON:`;

  let scores: Array<{ index: number; label: string; confidence: number }> = [];
  try {
    const response = await model.invoke([new HumanMessage(prompt)]);
    const jsonMatch = messageText(response.content).match(/\[\s*\{[\s\S]*\}\s*\]/);
    if (jsonMatch) scores = JSON.parse(jsonMatch[0]);
  } catch (error) {
    handleStageFailure("Relevance grading", error);
  }

  return docs.map((_, i) => {
    const score = scores.find((s) => s.index === i);
    if (!score) return { index: i, relevant: true, confidence: 0 };
    return {
      index: i,
      relevant: score.label === "RELEVANT" && score.confidence >= 0.5,
      confidence: score.confidence,
    };
  });
}

export async function gradeChunks(
  question: string,
  docs: Document[],
  llm?: ChatGroq
): Promise<Document[]> {
  const grades = await scoreChunks(question, docs, llm);
  return docs.filter((_, i) => grades[i].relevant);
}

const NO_RELEVANT_CONTENT = "No relevant content found.";

// Extracts only the question-relevant sentences from each chunk (in parallel)
export async function compressContext(
  question: string,
  docs: Document[],
  llm?: ChatGroq
): Promise<Document[]> {
  const model = llm || getChatModel();

  const prompt = `You are a context compressor for a RAG system.
Given a question and a document chunk, extract only the sentences that are relevant to answering the question.

Question: "${question}"

Copy the relevant sentences exactly as written, combined into a single paragraph.
If no sentences are relevant, output "${NO_RELEVANT_CONTENT}"

Chunk to compress:
`;

  const compressed = await Promise.all(
    docs.map(async (doc) => {
      try {
        const response = await model.invoke([
          new HumanMessage(
            prompt +
              `\n\n--- CHUNK START ---\n${doc.pageContent}\n--- CHUNK END ---\n\nOutput compressed context:`
          ),
        ]);
        const content = messageText(response.content).trim();
        if (!content || content.includes(NO_RELEVANT_CONTENT)) return null;

        return new Document({
          pageContent: content,
          metadata: {
            ...doc.metadata,
            compressed: true,
            originalLength: doc.pageContent.length,
          },
        });
      } catch (error) {
        handleStageFailure("Compression", error);
        return doc; // Keep original if compression fails
      }
    })
  );

  const kept = compressed.filter((doc): doc is Document => doc !== null);
  // If the compressor dropped everything, the grader already judged these
  // chunks relevant, so fall back to the uncompressed text.
  return kept.length > 0 ? kept : docs;
}
