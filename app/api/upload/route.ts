import { randomUUID } from "crypto";
import { NextResponse } from "next/server";
import {
  chunkDocuments,
  createCollectionName,
  deleteCollection,
  deleteDocumentChunks,
  indexChunks,
  isNotebookCollection,
  parseUploadedFile,
} from "@/lib/rag";

export const runtime = "nodejs";
export const maxDuration = 60;

// Uploading without a collectionName starts a new notebook; with one, the
// document is added to that notebook so questions search across all of them.
export async function POST(request: Request) {
  try {
    const formData = await request.formData();
    const file = formData.get("file");
    const existing = formData.get("collectionName");
    const appending = typeof existing === "string" && existing.length > 0;

    if (!(file instanceof File)) {
      return NextResponse.json({ error: "A PDF or text file is required." }, { status: 400 });
    }

    if (appending && !isNotebookCollection(existing)) {
      return NextResponse.json({ error: "Unknown notebook." }, { status: 400 });
    }

    const documentId = randomUUID();
    const documents = await parseUploadedFile(file);
    const chunks = await chunkDocuments(documents);
    for (const chunk of chunks) chunk.metadata.documentId = documentId;

    const collectionName = appending ? existing : createCollectionName();

    try {
      await indexChunks(collectionName, chunks);
    } catch (error) {
      // Don't leave a half-indexed document behind
      if (appending) await deleteDocumentChunks(collectionName, documentId);
      else await deleteCollection(collectionName);
      throw error;
    }

    return NextResponse.json({
      documentId,
      collectionName,
      fileName: file.name,
      chunks: chunks.length
    });
  } catch (error) {
    console.error("UPLOAD FAILED: ", error);
    const message = error instanceof Error ? error.message : "Failed to index document.";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
