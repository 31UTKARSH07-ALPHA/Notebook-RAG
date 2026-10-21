import { NextResponse } from "next/server";
import { answerQuestion } from "@/lib/rag";

export const runtime = "nodejs";
export const maxDuration = 60;

export async function POST(request: Request) {
  try {
    const { question, collectionName } = await request.json();

    if (!question || typeof question !== "string") {
      return NextResponse.json(
        { error: "Question is required." },
        { status: 400 },
      );
    }

    if (!collectionName || typeof collectionName !== "string") {
      return NextResponse.json(
        { error: "Upload a document before asking a question." },
        { status: 400 },
      );
    }

    return NextResponse.json(await answerQuestion(collectionName, question));
  } catch (error) {
    console.error("ASK FAILED: ", error);
    const message =
      error instanceof Error ? error.message : "Failed to answer question.";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
