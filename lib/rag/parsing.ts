// Turns an uploaded PDF or .txt file into one Document per page, with the
// PDF text cleaned up.

import { Document } from "@langchain/core/documents";
import { extractText } from "unpdf";

// Parsing

// Collapses whitespace and rejoins words hyphenated at line breaks
// ("abil- ity" -> "ability"). Real compounds ("long-context") have no space
// after the hyphen, and "best- and worst-case" is a suspended hyphen, so
// and/or/to are kept.
export function cleanPageText(pageText: string): string {
  return pageText
    .replace(/\s+/g, " ")
    .replace(/([a-z])- (?!(?:and|or|to)\b)([a-z])/g, "$1$2")
    .trim();
}

async function parsePdf(buffer: ArrayBuffer): Promise<Document[]> {
  const { text, totalPages } = await extractText(
    buffer,
    { mergePages: false }
  );
  const pages = Array.isArray(text) ? text : [text];

  return pages
    .map((pageText, index) =>
      new Document({
        pageContent: cleanPageText(pageText),
        metadata: {
          type: "pdf",
          source: "",
          loc: { pageNumber: index + 1 },
          totalPages,
        },
      })
    )
    .filter((doc) => doc.pageContent.length > 20);
}

export async function parseUploadedFile(file: File): Promise<Document[]> {
  const fileName = file.name || "uploaded-document";
  const mimeType = file.type;

  if (mimeType === "text/plain" || fileName.toLowerCase().endsWith(".txt")) {
    const text = await file.text();
    return [
      new Document({
        pageContent: text,
        metadata: { source: fileName, type: "text" },
      }),
    ];
  }

  if (
    mimeType === "application/pdf" ||
    fileName.toLowerCase().endsWith(".pdf")
  ) {
    const buffer = await file.arrayBuffer();
    const pages = await parsePdf(buffer);

    if (pages.length === 0) {
      throw new Error(
        "No readable text found in the PDF. It may be a scanned image-only PDF."
      );
    }

    return pages.map(
      (page) =>
        new Document({
          pageContent: page.pageContent,
          metadata: { ...page.metadata, source: fileName },
        })
    );
  }

  throw new Error("Unsupported file type. Upload a PDF or .txt file.");
}
