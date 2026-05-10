import { NextResponse } from "next/server";
import { chunkDocuments, createCollectionName, indexChunks, parseUploadedFile } from "@/lib/rag";

export const runtime = "nodejs";
export const maxDuration = 60;

export async function POST(request: Request) {
  try {
    const formData = await request.formData();
    const file = formData.get("file");

    if (!(file instanceof File)) {
      return NextResponse.json({ error: "A PDF or text file is required." }, { status: 400 });
    }

    const documents = await parseUploadedFile(file);
    const chunks = await chunkDocuments(documents);
    const collectionName = createCollectionName();

    await indexChunks(collectionName, chunks);

    return NextResponse.json({
      documentId: collectionName,
      collectionName,
      fileName: file.name,
      chunks: chunks.length
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Failed to index document.";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
