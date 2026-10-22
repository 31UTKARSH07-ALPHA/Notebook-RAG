// Upload size limit, shared by the browser (checked before sending) and the
// upload route (checked again, since the client can't be trusted).
//
// Vercel rejects serverless request bodies above ~4.5 MB with a bare 413
// before our code runs, so the app enforces a slightly lower limit itself and
// can explain it. The multipart wrapper adds a little on top of the file.

export const MAX_UPLOAD_BYTES = 4 * 1024 * 1024;
export const MAX_UPLOAD_LABEL = "4 MB";

// Headroom for multipart boundaries and the collectionName field when the
// route checks Content-Length before reading the body.
export const MAX_UPLOAD_REQUEST_BYTES = MAX_UPLOAD_BYTES + 64 * 1024;

export function isUploadTooLarge(sizeBytes: number): boolean {
  return sizeBytes > MAX_UPLOAD_BYTES;
}

export function uploadTooLargeMessage(sizeBytes?: number): string {
  const size =
    typeof sizeBytes === "number" ? ` (this one is ${(sizeBytes / (1024 * 1024)).toFixed(1)} MB)` : "";
  return `Files can be at most ${MAX_UPLOAD_LABEL}${size}. Try a smaller PDF or split it into parts.`;
}
