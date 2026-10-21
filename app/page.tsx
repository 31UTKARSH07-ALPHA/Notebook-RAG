"use client";

import { FormEvent, useMemo, useState } from "react";
import { BookOpen, FileText, Loader2, Send, UploadCloud } from "lucide-react";

type Source = {
  chunk: number;
  page?: number;
  text: string;
  rerankScore?: number;
};

type Grounding = {
  verdict: "grounded" | "not_grounded" | "unverified" | "not_found";
  confidence: number;
  reasoning: string;
  retried: boolean;
};

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  grounding?: Grounding;
  totalMs?: number;
};

const GROUNDING_LABELS: Record<Grounding["verdict"], string> = {
  grounded: "Verified against sources",
  not_grounded: "May contain unsupported claims",
  unverified: "Grounding check unavailable",
  not_found: "Not found in document"
};

type UploadedDocument = {
  documentId: string;
  collectionName: string;
  fileName: string;
  chunks: number;
};

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [document, setDocument] = useState<UploadedDocument | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [question, setQuestion] = useState("");
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const [isAsking, setIsAsking] = useState(false);

  const canAsk = useMemo(
    () => Boolean(document && question.trim() && !isAsking),
    [document, question, isAsking]
  );

  async function uploadDocument(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file) return;

    setIsUploading(true);
    setError("");
    setStatus("Reading, chunking, embedding, and indexing your document...");

    const body = new FormData();
    body.append("file", file);
    if (document) body.append("previousCollectionName", document.collectionName);

    try {
      const response = await fetch("/api/upload", {
        method: "POST",
        body
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Upload failed");
      }

      setDocument(data);
      setMessages([]);
      setStatus(`Indexed ${data.chunks} chunks from ${data.fileName}.`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
      setStatus("");
    } finally {
      setIsUploading(false);
    }
  }

  async function askQuestion(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!canAsk || !document) return;

    const nextQuestion = question.trim();
    setQuestion("");
    setError("");
    setIsAsking(true);
    setMessages((current) => [...current, { role: "user", content: nextQuestion }]);

    try {
      const response = await fetch("/api/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          question: nextQuestion,
          collectionName: document.collectionName
        })
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Could not answer that question");
      }

      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: data.answer,
          sources: data.sources,
          grounding: data.grounding,
          totalMs: data.timingsMs?.total
        }
      ]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not answer that question");
    } finally {
      setIsAsking(false);
    }
  }

  return (
    <main className="shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <BookOpen size={22} />
          </div>
          <div>
            <h1>SourceSage</h1>
            <p>Ask questions grounded in one uploaded file.</p>
          </div>
        </div>

        <form className="upload-box" onSubmit={uploadDocument}>
          <label className="file-input">
            <span>PDF or plain text</span>
            <input
              accept=".pdf,.txt,text/plain,application/pdf"
              disabled={isUploading}
              type="file"
              onChange={(event) => setFile(event.target.files?.[0] ?? null)}
            />
          </label>
          <button className="primary" disabled={!file || isUploading} type="submit">
            {isUploading ? <Loader2 size={18} /> : <UploadCloud size={18} />}
            {isUploading ? "Indexing" : "Upload and index"}
          </button>
          {status ? <p className="status">{status}</p> : null}
          {error ? <p className="error">{error}</p> : null}
        </form>

        <div className="tips" aria-label="Pipeline">
          <div className="tip">
            <strong>Chunking</strong>
            <p className="section-note">2-sentence chunks are searched; the LLM sees them with 2 sentences of context on each side.</p>
          </div>
          <div className="tip">
            <strong>Retrieval</strong>
            <p className="section-note">Question rewrites, hybrid dense + BM25 search fused with RRF, then Cohere reranking.</p>
          </div>
          <div className="tip">
            <strong>Grounding</strong>
            <p className="section-note">Irrelevant chunks are filtered out, and every answer is checked against its sources.</p>
          </div>
        </div>
      </aside>

      <section className="main">
        <header className="topbar">
          <div>
            <h2>Document chat</h2>
            <p className="section-note">Upload a new document whenever you want a fresh notebook.</p>
          </div>
          <div className="pill" title={document?.fileName || "No document indexed"}>
            {document ? `${document.fileName} · ${document.chunks} chunks` : "No document indexed"}
          </div>
        </header>

        <div className="conversation">
          {messages.length === 0 ? (
            <div className="empty">
              <FileText size={42} />
              <h3>{document ? "Ask from the document" : "Upload a document first"}</h3>
              <p>
                {document
                  ? "Questions are answered from the most relevant indexed chunks."
                  : "The app accepts unseen PDFs and text files, then builds a searchable vector index."}
              </p>
            </div>
          ) : (
            messages.map((message, index) => (
              <article className={`message ${message.role}`} key={`${message.role}-${index}`}>
                <div className="bubble">{message.content}</div>
                {message.grounding ? (
                  <div
                    className={`grounding ${message.grounding.verdict}`}
                    title={message.grounding.reasoning}
                  >
                    {GROUNDING_LABELS[message.grounding.verdict]}
                    {message.grounding.retried ? " · regenerated once" : ""}
                    {message.totalMs ? ` · ${(message.totalMs / 1000).toFixed(1)}s` : ""}
                  </div>
                ) : null}
                {message.sources?.length ? (
                  <div className="sources">
                    {message.sources.map((source) => (
                      <div className="source" key={source.chunk}>
                        <div className="source-meta">
                          Chunk {source.chunk}
                          {source.page ? ` · page ${source.page}` : ""}
                          {source.rerankScore !== undefined ? ` · relevance ${source.rerankScore.toFixed(2)}` : ""}
                        </div>
                        {source.text}
                      </div>
                    ))}
                  </div>
                ) : null}
              </article>
            ))
          )}
          {isAsking ? (
            <article className="message assistant">
              <div className="bubble">Retrieving relevant chunks and drafting a grounded answer...</div>
            </article>
          ) : null}
        </div>

        <form className="composer" onSubmit={askQuestion}>
          <textarea
            disabled={!document || isAsking}
            placeholder={document ? "Ask a question about the uploaded document" : "Upload a document to start"}
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
          />
          <button className="icon-button" disabled={!canAsk} title="Send question" type="submit">
            <Send size={19} />
          </button>
        </form>
      </section>
    </main>
  );
}
