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
import { checkUploadLimit } from "@/lib/demo-limits";
import { maybeCleanupExpiredNotebooks, stampNotebookCreatedAt } from "@/lib/notebook-cleanup";
import { MAX_UPLOAD_REQUEST_BYTES, isUploadTooLarge, uploadTooLargeMessage } from "@/lib/upload-limits";

export const runtime = "nodejs";
export const maxDuration = 60;

// Uploading without a collectionName starts a new notebook; with one, the
// document is added to that notebook so questions search across all of them.
export async function POST(request: Request) {
  // Reject oversized bodies before reading them (Vercel itself cuts off at ~4.5 MB)
  const contentLength = Number(request.headers.get("content-length"));
  if (Number.isFinite(contentLength) && contentLength > MAX_UPLOAD_REQUEST_BYTES) {
    return NextResponse.json({ error: uploadTooLargeMessage() }, { status: 413 });
  }

  try {
    const formData = await request.formData();
    const file = formData.get("file");
    const existing = formData.get("collectionName");
    const appending = typeof existing === "string" && existing.length > 0;

    if (!(file instanceof File)) {
      return NextResponse.json({ error: "A PDF or text file is required." }, { status: 400 });
    }

    if (isUploadTooLarge(file.size)) {
      return NextResponse.json({ error: uploadTooLargeMessage(file.size) }, { status: 413 });
    }

    if (appending && !isNotebookCollection(existing)) {
      return NextResponse.json({ error: "Unknown notebook." }, { status: 400 });
    }

    // Demo mode only: per-visitor daily cap (best-effort, see lib/demo-limits.ts)
    const limit = checkUploadLimit(request);
    if (!limit.allowed) {
      return NextResponse.json({ error: limit.message }, { status: 429 });
    }

    // Throttled background deletion of expired notebooks; never blocks or fails this upload
    maybeCleanupExpiredNotebooks(appending ? [existing] : []);

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

    // Creation time for notebook expiry (lib/notebook-cleanup.ts). Best-effort:
    // if it fails, cleanup stamps the notebook when it first sees it.
    if (!appending) {
      await stampNotebookCreatedAt(collectionName).catch((error) =>
        console.warn(`Could not stamp createdAt on ${collectionName}:`, error)
      );
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
