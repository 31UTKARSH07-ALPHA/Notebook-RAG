import { NextResponse } from "next/server";
import {
  gradeAnswer,
  gradeChunks,
  compressContext,
  formatSources,
} from "@/lib/rag";

export const runtime = "nodejs";
export const maxDuration = 60;

export async function POST(request: Request) {
  try {
    const { question, answer, chunks, mode = "full" } = await request.json();

    if (!question || typeof question !== "string") {
      return NextResponse.json(
        { error: "Question is required." },
        { status: 400 }
      );
    }

    if (!answer || typeof answer !== "string") {
      return NextResponse.json(
        { error: "Answer is required." },
        { status: 400 }
      );
    }

    if (!chunks || !Array.isArray(chunks)) {
      return NextResponse.json(
        { error: "Chunks (array of documents) are required." },
        { status: 400 }
      );
    }

    const results: {
      chunk_scores: Array<{ index: number; relevance: number; reasoning?: string }>;
      answer_scores: {
        grounded: boolean;
        confidence: number;
        reasoning: string;
      };
      compressed_chunks?: Array<{ original: string; compressed: string }>;
    } = {
      chunk_scores: [],
      answer_scores: { grounded: false, confidence: 0, reasoning: "" },
      compressed_chunks: mode === "full" ? [] : undefined,
    };

    // Grade each chunk for relevance
    const gradedChunks = await gradeChunks(question, chunks);
    results.chunk_scores = gradedChunks.map((chunk, index) => ({
      index: index,
      relevance: 0.8, // Placeholder - in real implementation, parse from gradeChunks output
      reasoning: "Chunk passed relevance threshold",
    }));

    // Grade the answer for hallucination
    const answerGrading = await gradeAnswer(answer, chunks, question);
    results.answer_scores = {
      grounded: answerGrading.grounded,
      confidence: answerGrading.confidence,
      reasoning: answerGrading.reasoning,
    };

    // Compress context if full mode
    if (mode === "full") {
      const compressed = await compressContext(question, chunks);
      results.compressed_chunks = compressed.map((doc, i) => ({
        original: chunks[i].pageContent,
        compressed: doc.pageContent,
      }));
    }

    return NextResponse.json({
      status: "success",
      results,
      evaluation_summary: {
        avg_chunk_relevance: results.chunk_scores.reduce((sum, s) => sum + s.relevance, 0) / Math.max(1, results.chunk_scores.length),
        answer_grounded: results.answer_scores.grounded,
        answer_confidence: results.answer_scores.confidence,
      },
    });
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Failed to evaluate.";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
