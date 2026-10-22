import { NextResponse } from "next/server";
import { answerQuestion, isNotebookCollection } from "@/lib/rag";

export const runtime = "nodejs";
export const maxDuration = 60;

// Request:  { question, collectionName, history?: [{ role, content }] }
// Response: newline-delimited JSON, one event per line:
//   {"type":"stage","stage":"search"}      as each pipeline stage starts
//   {"type":"result", answer, standaloneQuestion, sources, grounding, queries, timingsMs}
//   {"type":"error","error":"..."}         instead of a result if the pipeline fails
//
// Only progress is streamed, not answer tokens: the answer can be rejected by
// the grounding check and regenerated, so it is sent once it's final (see
// answerQuestion). Bad requests are rejected before streaming starts, so they
// still get a normal JSON body and a 400 status.
export async function POST(request: Request) {
  const body = await request.json().catch(() => null);
  const { question, collectionName, history } = body ?? {};

  if (!question || typeof question !== "string") {
    return NextResponse.json({ error: "Question is required." }, { status: 400 });
  }

  if (!collectionName || typeof collectionName !== "string") {
    return NextResponse.json(
      { error: "Upload a document before asking a question." },
      { status: 400 },
    );
  }

  if (!isNotebookCollection(collectionName)) {
    return NextResponse.json({ error: "Unknown notebook." }, { status: 400 });
  }

  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      let open = true;
      const send = (event: Record<string, unknown>) => {
        if (!open) return;
        try {
          controller.enqueue(encoder.encode(`${JSON.stringify(event)}\n`));
        } catch {
          open = false; // client went away; the pipeline just finishes unobserved
        }
      };

      try {
        // history is validated and capped inside answerQuestion (capHistory)
        const { context, retrieved, ...result } = await answerQuestion(
          collectionName,
          question,
          undefined,
          { history, onStage: (stage) => send({ type: "stage", stage }) },
        );
        // context/retrieved hold full chunk text; the UI only needs the short sources
        send({ type: "result", ...result });
      } catch (error) {
        console.error("ASK FAILED: ", error);
        const message = error instanceof Error ? error.message : "Failed to answer question.";
        send({ type: "error", error: message });
      } finally {
        try {
          controller.close();
        } catch {
          // already closed because the client disconnected
        }
      }
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "application/x-ndjson; charset=utf-8",
      // Stop proxies and compression from buffering the stage events
      "Cache-Control": "no-cache, no-transform",
      "X-Accel-Buffering": "no",
    },
  });
}
