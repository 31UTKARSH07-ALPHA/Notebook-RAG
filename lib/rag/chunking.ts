// Splits pages into chunks: fixed-size (the default), sentence-window and
// semantic, plus the sentence splitting they share.

import { Document } from "@langchain/core/documents";
import { embed, getEmbeddingConfig } from "./cohere";

// Sentence-window chunking config
const SENTENCES_PER_CHILD = 2; // small chunk indexed in Qdrant
const CONTEXT_SENTENCES = 2; // sentences added on each side to build the parent sent to the LLM
export const MAX_SENTENCE_CHARS = 500; // longer "sentences" (tables, unpunctuated text) get split on words

// Fixed-size chunking: the default. In the evaluation it retrieved the most
// evidence of the three strategies, with less context than sentence-window and
// no extra upload cost (see DECISIONS.md and eval/RESULTS.md).
export const FIXED_CHUNK_CHARS = 1000;
export const FIXED_OVERLAP_CHARS = 200;

// Semantic chunking: cut where consecutive sentences stop being about the
// same thing, judged by the similarity of their embeddings.
// Similarities below this percentile of the document's own consecutive-pair
// similarities count as a topic shift. A per-document percentile instead of a
// fixed cutoff, because absolute cosine values vary a lot between documents
// (a policy text that repeats "AI risk" everywhere scores high throughout).
// At 25%, roughly one sentence gap in four is a candidate cut, which with
// ~150-character sentences lands near the max size below rather than far under it.
const SEMANTIC_BREAKPOINT_PERCENTILE = 25;
// Same ceiling as "fixed", so the comparison is about where chunks are cut,
// not about chunk size.
const SEMANTIC_MAX_CHUNK_CHARS = 1000;
// Below this a similarity drop doesn't cut: a 1-2 sentence fragment has too
// little context to answer anything on its own.
const SEMANTIC_MIN_CHUNK_CHARS = 300;

export type ChunkingStrategy = "sentence-window" | "fixed" | "semantic";
export const DEFAULT_CHUNKING: ChunkingStrategy = "fixed";

// Chunking

// Splits text without sentence punctuation (tables, lists, headings) into
// word-bounded pieces so one "sentence" can't swallow a whole page.
export function splitLongSentence(sentence: string): string[] {
  if (sentence.length <= MAX_SENTENCE_CHARS) return [sentence];

  const pieces: string[] = [];
  let current = "";
  for (const word of sentence.split(" ")) {
    if (current && current.length + word.length + 1 > MAX_SENTENCE_CHARS) {
      pieces.push(current);
      current = word;
    } else {
      current = current ? `${current} ${word}` : word;
    }
  }
  if (current) pieces.push(current);
  return pieces;
}

export function splitIntoSentences(text: string): string[] {
  // Split on sentence-ending punctuation followed by whitespace + capital letter
  return text
    .replace(/\s+/g, " ")
    .split(/(?<=[.!?])\s+(?=[A-Z"'])/)
    .map((s) => s.trim())
    .filter((s) => s.length > 10)
    .flatMap(splitLongSentence);
}

// ~1000-character chunks with 200 characters of overlap, cut on word
// boundaries. The chunk that is searched is also what the LLM sees.
function chunkFixedSize(documents: Document[]): Document[] {
  const chunks: Document[] = [];
  let chunkIndex = 1;

  for (const doc of documents) {
    const words = doc.pageContent.replace(/\s+/g, " ").trim().split(" ");
    if (!words[0]) continue; // empty page: "".split(" ") is [""], which would make an empty chunk
    let start = 0;
    while (start < words.length) {
      let end = start;
      let length = 0;
      while (end < words.length && length + words[end].length + 1 <= FIXED_CHUNK_CHARS) {
        length += words[end].length + 1;
        end++;
      }
      end = Math.max(end, start + 1); // a single over-long word still advances
      const text = words.slice(start, end).join(" ");
      chunks.push(
        new Document({
          pageContent: text,
          metadata: { ...doc.metadata, chunk: chunkIndex++, chunking: "fixed-1000-200", parentText: text },
        })
      );
      if (end >= words.length) break;

      // Step back ~200 characters for the overlap
      let back = end;
      let overlap = 0;
      while (back > start + 1 && overlap < FIXED_OVERLAP_CHARS) {
        back--;
        overlap += words[back].length + 1;
      }
      start = back;
    }
  }

  return chunks;
}

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA * normB) || 1);
}

// Groups consecutive sentences into chunks, starting a new chunk where the
// embedding similarity between two neighbouring sentences drops below the
// document's breakpoint threshold, or when the chunk would exceed the max
// size. Chunks never cross a page, so each keeps one page number. Like
// "fixed", the chunk that is searched is also what the LLM sees.
//
// Cost: every sentence is embedded once at upload, only to find the cut
// points, and then the chunks are embedded again by indexChunks(). That
// doubles the embedded text and adds one Cohere call per 96 sentences: for
// the 48-page NIST PDF, ~7 sentence calls on top of 2 for the chunks.
async function chunkSemantic(documents: Document[]): Promise<Document[]> {
  const pages = documents.map((doc) => splitIntoSentences(doc.pageContent));
  const vectors = await embed(pages.flat(), "search_document", getEmbeddingConfig());

  // Similarity between each sentence and the next one on the same page
  let offset = 0;
  const pageSimilarities = pages.map((sentences) => {
    const similarities = sentences
      .slice(1)
      .map((_, i) => cosineSimilarity(vectors[offset + i], vectors[offset + i + 1]));
    offset += sentences.length;
    return similarities;
  });

  // Threshold from the whole document's distribution, not per page, so a
  // page that is all one topic isn't forced to cut somewhere anyway
  const allSimilarities = pageSimilarities.flat().sort((a, b) => a - b);
  const threshold = allSimilarities.length
    ? allSimilarities[Math.floor((SEMANTIC_BREAKPOINT_PERCENTILE / 100) * (allSimilarities.length - 1))]
    : 0;

  const chunks: Document[] = [];
  let chunkIndex = 1;

  documents.forEach((doc, pageIndex) => {
    const sentences = pages[pageIndex];
    const groups: string[][] = [];
    let current: string[] = [];
    let length = 0;

    sentences.forEach((sentence, i) => {
      if (current.length > 0) {
        const tooLong = length + sentence.length + 1 > SEMANTIC_MAX_CHUNK_CHARS;
        const topicShift =
          pageSimilarities[pageIndex][i - 1] < threshold && length >= SEMANTIC_MIN_CHUNK_CHARS;
        if (tooLong || topicShift) {
          groups.push(current);
          current = [];
          length = 0;
        }
      }
      current.push(sentence);
      length += sentence.length + (current.length > 1 ? 1 : 0);
    });

    // A short tail at the end of the page joins the previous chunk when it fits
    if (current.length > 0) {
      const previous = groups[groups.length - 1];
      if (
        previous &&
        length < SEMANTIC_MIN_CHUNK_CHARS &&
        previous.join(" ").length + 1 + length <= SEMANTIC_MAX_CHUNK_CHARS
      ) {
        previous.push(...current);
      } else {
        groups.push(current);
      }
    }

    for (const group of groups) {
      const text = group.join(" ");
      chunks.push(
        new Document({
          pageContent: text,
          metadata: {
            ...doc.metadata,
            chunk: chunkIndex++,
            chunking: `semantic-p${SEMANTIC_BREAKPOINT_PERCENTILE}-${SEMANTIC_MIN_CHUNK_CHARS}-${SEMANTIC_MAX_CHUNK_CHARS}`,
            parentText: text,
          },
        })
      );
    }
  });

  return chunks;
}

export async function chunkDocuments(
  documents: Document[],
  strategy: ChunkingStrategy = DEFAULT_CHUNKING
): Promise<Document[]> {
  if (strategy === "fixed") return chunkFixedSize(documents);
  if (strategy === "semantic") return chunkSemantic(documents);

  const allChildren: Document[] = [];
  let globalChunkIndex = 1;

  for (const doc of documents) {
    const sentences = splitIntoSentences(doc.pageContent);
    const totalSentences = sentences.length;

    for (let i = 0; i < totalSentences; i += SENTENCES_PER_CHILD) {
      // Child: small, precise chunk that gets embedded and searched
      const childText = sentences.slice(i, i + SENTENCES_PER_CHILD).join(" ");

      // Parent: the child plus surrounding sentences, sent to the LLM
      const windowStart = Math.max(0, i - CONTEXT_SENTENCES);
      const windowEnd = Math.min(
        totalSentences,
        i + SENTENCES_PER_CHILD + CONTEXT_SENTENCES
      );
      const parentText = sentences.slice(windowStart, windowEnd).join(" ");

      allChildren.push(
        new Document({
          pageContent: childText,
          metadata: {
            ...doc.metadata,
            chunk: globalChunkIndex,
            chunking: `sentence-window-child-${SENTENCES_PER_CHILD}-context-${CONTEXT_SENTENCES}`,
            parentText,
            sentenceStart: i,
            sentenceEnd: i + SENTENCES_PER_CHILD,
          },
        })
      );

      globalChunkIndex++;
    }
  }

  return allChildren.filter((c) => c.pageContent.trim().length > 0);
}
