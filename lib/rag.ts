import { randomUUID } from "crypto";
import { Document } from "@langchain/core/documents";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { QdrantVectorStore } from "@langchain/qdrant";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import PDFParser from "pdf2json";

const CHUNK_SIZE = 1000;
const CHUNK_OVERLAP = 200;
const DEFAULT_PREFIX = "notebooklm";

export type SourceChunk = {
  chunk: number;
  page?: number;
  text: string;
};

function requireEnv(name: string) {
  const value = process.env[name];
  if (!value) {
    throw new Error(`${name} is not configured. Add it to your environment variables.`);
  }
  return value;
}

function getQdrantConfig() {
  return {
    url: requireEnv("QDRANT_URL"),
    apiKey: process.env.QDRANT_API_KEY,
    collectionName: ""
  };
}

export function createCollectionName() {
  const prefix = process.env.QDRANT_COLLECTION_PREFIX || DEFAULT_PREFIX;
  return `${prefix}_${randomUUID().replaceAll("-", "_")}`;
}

export function getEmbeddings() {
  requireEnv("GOOGLE_API_KEY");
  return new GoogleGenerativeAIEmbeddings({
    modelName: "text-embedding-004"
  });
}

export function getChatModel() {
  requireEnv("GOOGLE_API_KEY");
  return new ChatGoogleGenerativeAI({
    model: "gemini-1.5-flash",
    temperature: 0.1,
    maxOutputTokens: 1200
  });
}

function parsePdfBuffer(buffer: Buffer): Promise<Document[]> {
  return new Promise((resolve, reject) => {
    const pdfParser = new (PDFParser as any)(null, 1);

    pdfParser.on("pdfParser_dataError", (err: any) => {
      reject(new Error(err?.parserError || "Failed to parse PDF"));
    });

    pdfParser.on("pdfParser_dataReady", (pdfData: any) => {
      try {
        const pages: Document[] = (pdfData?.Pages || [])
          .map((page: any, index: number) => {
            const text = (page?.Texts || [])
              .map((t: any) =>
                (t?.R || [])
                  .map((r: any) => decodeURIComponent(r?.T || ""))
                  .join("")
              )
              .join(" ")
              .replace(/\s+/g, " ")
              .trim();

            return new Document({
              pageContent: text,
              metadata: {
                type: "pdf",
                loc: { pageNumber: index + 1 }
              }
            });
          })
          .filter((doc: Document) => doc.pageContent.length > 0);

        resolve(pages);
      } catch (e) {
        reject(e);
      }
    });

    pdfParser.parseBuffer(buffer);
  });
}

export async function parseUploadedFile(file: File) {
  const fileName = file.name || "uploaded-document";
  const mimeType = file.type;

  if (mimeType === "text/plain" || fileName.toLowerCase().endsWith(".txt")) {
    const text = await file.text();
    return [
      new Document({
        pageContent: text,
        metadata: { source: fileName, type: "text" }
      })
    ];
  }

  if (mimeType === "application/pdf" || fileName.toLowerCase().endsWith(".pdf")) {
    const buffer = Buffer.from(await file.arrayBuffer());
    const pages = await parsePdfBuffer(buffer);

    if (pages.length === 0) {
      throw new Error("No readable text found in the PDF. It may be a scanned image-only PDF.");
    }

    return pages.map(
      (page) =>
        new Document({
          pageContent: page.pageContent,
          metadata: { ...page.metadata, source: fileName }
        })
    );
  }

  throw new Error("Unsupported file type. Upload a PDF or .txt file.");
}

export async function chunkDocuments(documents: Document[]) {
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: CHUNK_SIZE,
    chunkOverlap: CHUNK_OVERLAP
  });

  const chunks = await splitter.splitDocuments(documents);

  return chunks
    .filter((chunk) => chunk.pageContent.trim().length > 0)
    .map(
      (chunk, index) =>
        new Document({
          pageContent: chunk.pageContent,
          metadata: {
            ...chunk.metadata,
            chunk: index + 1,
            chunking: `recursive-character-${CHUNK_SIZE}-${CHUNK_OVERLAP}`
          }
        })
    );
}

export async function indexChunks(collectionName: string, chunks: Document[]) {
  if (chunks.length === 0) {
    throw new Error("No readable text was found in the uploaded document.");
  }

  const config = getQdrantConfig();
  const embeddings = getEmbeddings();

  await QdrantVectorStore.fromDocuments(chunks, embeddings, {
    url: config.url,
    apiKey: config.apiKey,
    collectionName
  });
}

export async function retrieveChunks(collectionName: string, question: string) {
  const config = getQdrantConfig();
  const vectorStore = await QdrantVectorStore.fromExistingCollection(getEmbeddings(), {
    url: config.url,
    apiKey: config.apiKey,
    collectionName
  });

  return vectorStore.similaritySearch(question, 5);
}

export function formatSources(chunks: Document[]): SourceChunk[] {
  return chunks.map((chunk, index) => ({
    chunk: Number(chunk.metadata.chunk || index + 1),
    page: Number(chunk.metadata.loc?.pageNumber || chunk.metadata.pdf?.pageNumber || 0) || undefined,
    text: chunk.pageContent.replace(/\s+/g, " ").slice(0, 420)
  }));
}

export function buildGroundedPrompt(chunks: Document[]) {
  const context = chunks
    .map((chunk, index) => {
      const chunkNumber = chunk.metadata.chunk || index + 1;
      const page = chunk.metadata.loc?.pageNumber || chunk.metadata.pdf?.pageNumber;
      return `SOURCE ${index + 1} | chunk ${chunkNumber}${page ? ` | page ${page}` : ""}\n${chunk.pageContent}`;
    })
    .join("\n\n---\n\n");

  return `You are a document question-answering assistant.

Answer using only the sources below. If the sources do not contain the answer, say: "I could not find that in the uploaded document."

Rules:
- Do not use outside knowledge.
- Be concise but complete.
- Mention source chunk numbers naturally when useful.
- Never invent facts, names, page numbers, or examples that are not present in the sources.

Sources:
${context}`;
}

export async function generateGroundedAnswer(chunks: Document[], question: string) {
  const model = getChatModel();
  const response = await model.invoke([
    new SystemMessage(buildGroundedPrompt(chunks)),
    new HumanMessage(question)
  ]);

  return typeof response.content === "string"
    ? response.content
    : response.content.map((part) => ("text" in part ? part.text : "")).join("");
}