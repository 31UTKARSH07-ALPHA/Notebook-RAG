import { NextResponse } from "next/server";
import { deleteCollection } from "@/lib/rag";

export const runtime = "nodejs";

// Deletes a notebook's collection. deleteCollection() ignores names without
// the app's prefix, so this can't be used to drop unrelated collections.
export async function DELETE(request: Request) {
  const { collectionName } = await request.json().catch(() => ({}));

  if (!collectionName || typeof collectionName !== "string") {
    return NextResponse.json({ error: "collectionName is required." }, { status: 400 });
  }

  await deleteCollection(collectionName);
  return NextResponse.json({ deleted: true });
}
