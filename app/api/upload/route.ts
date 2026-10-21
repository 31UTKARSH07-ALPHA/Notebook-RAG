import { NextResponse } from "next/server";
import {
  chunkDocuments,
  createCollectionName,
  deleteCollection,
  indexChunks,
  parseUploadedFile,
} from "@/lib/rag";

export const runtime = "nodejs";
export const maxDuration = 60;

export async function POST(request: Request) {
  try {
    const formData = await request.formData();
    const file = formData.get("file");
    const previousCollectionName = formData.get("previousCollectionName");

    if (!(file instanceof File)) {
      return NextResponse.json({ error: "A PDF or text file is required." }, { status: 400 });
    }

    const documents = await parseUploadedFile(file);
    const chunks = await chunkDocuments(documents);
    const collectionName = createCollectionName();

    try {
      await indexChunks(collectionName, chunks);
    } catch (error) {
      // Don't leave a half-filled collection behind
      await deleteCollection(collectionName);
      throw error;
    }

    // The new document replaces the old one, so drop its collection
    if (typeof previousCollectionName === "string" && previousCollectionName) {
      await deleteCollection(previousCollectionName);
    }

    return NextResponse.json({
      documentId: collectionName,
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
