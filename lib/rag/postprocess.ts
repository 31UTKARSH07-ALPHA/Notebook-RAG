// Post-retrieval: LLM relevance grading of the retrieved chunks (CRAG) and
// context compression down to the relevant sentences.

import { Document } from "@langchain/core/documents";
import { ChatGroq } from "@langchain/groq";
import { HumanMessage } from "@langchain/core/messages";
import { handleStageFailure } from "./config";
import { getChatModel, messageText } from "./llm";

// Post-retrieval

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
