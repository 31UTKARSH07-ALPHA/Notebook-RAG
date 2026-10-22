import { describe, expect, it } from "vitest";
import { Document } from "@langchain/core/documents";
import { FIXED_CHUNK_CHARS, FIXED_OVERLAP_CHARS, chunkDocuments } from "../lib/rag";

const page = (text: string, pageNumber = 1) =>
  new Document({
    pageContent: text,
    metadata: { source: "report.pdf", type: "pdf", loc: { pageNumber } },
  });

// "Sentence number 0 on page 1." ... each long enough to survive the 10-char filter
const sentences = (count: number, pageNumber = 1) =>
  Array.from({ length: count }, (_, i) => `Sentence number ${i} on page ${pageNumber}.`);

describe('chunkDocuments "sentence-window"', () => {
  const s = sentences(7);
  const chunksPromise = chunkDocuments([page(s.join(" "))], "sentence-window");

  it("makes children of 2 sentences, the last one shorter", async () => {
    const chunks = await chunksPromise;
    expect(chunks.map((c) => c.pageContent)).toEqual([
      `${s[0]} ${s[1]}`,
      `${s[2]} ${s[3]}`,
      `${s[4]} ${s[5]}`,
      s[6],
    ]);
  });

  it("builds the parent from the child plus 2 sentences each side, clamped at page edges", async () => {
    const chunks = await chunksPromise;
    expect(chunks.map((c) => c.metadata.parentText)).toEqual([
      s.slice(0, 4).join(" "), // no sentences before the first child
      s.slice(0, 6).join(" "),
      s.slice(2, 7).join(" "), // only one sentence after
      s.slice(4, 7).join(" "),
    ]);
  });

  it("keeps the page metadata and records the sentence range", async () => {
    const [first, second] = await chunksPromise;
    expect(first.metadata).toMatchObject({
      source: "report.pdf",
      type: "pdf",
      loc: { pageNumber: 1 },
      chunking: "sentence-window-child-2-context-2",
      sentenceStart: 0,
      sentenceEnd: 2,
    });
    expect(second.metadata).toMatchObject({ sentenceStart: 2, sentenceEnd: 4 });
  });

  it("numbers chunks continuously across pages and never mixes pages in a window", async () => {
    const p1 = sentences(3, 1);
    const p2 = sentences(4, 2);
    const chunks = await chunkDocuments([page(p1.join(" "), 1), page(p2.join(" "), 2)], "sentence-window");

    expect(chunks.map((c) => c.metadata.chunk)).toEqual([1, 2, 3, 4]);
    expect(chunks.map((c) => c.metadata.loc.pageNumber)).toEqual([1, 1, 2, 2]);
    expect(chunks[2].metadata.parentText).toBe(p2.join(" "));
  });

  it("returns no chunks for an empty page", async () => {
    expect(await chunkDocuments([page("")], "sentence-window")).toEqual([]);
  });
});

describe('chunkDocuments "fixed"', () => {
  // Unique words of varying length, so overlaps can be located exactly
  const words = Array.from({ length: 900 }, (_, i) => `w${i}${"x".repeat(i % 9)}`);
  const longestWord = Math.max(...words.map((w) => w.length));
  const text = words.join(" ");
  const chunksPromise = chunkDocuments([page(text)], "fixed");

  it(`keeps every chunk within ${FIXED_CHUNK_CHARS} characters`, async () => {
    const chunks = await chunksPromise;
    expect(chunks.length).toBeGreaterThan(3);
    for (const chunk of chunks) expect(chunk.pageContent.length).toBeLessThanOrEqual(FIXED_CHUNK_CHARS);
  });

  it(`overlaps consecutive chunks by about ${FIXED_OVERLAP_CHARS} characters`, async () => {
    const chunks = await chunksPromise;
    for (let i = 1; i < chunks.length; i++) {
      const previous = chunks[i - 1].pageContent;
      const firstWord = chunks[i].pageContent.split(" ")[0];
      const overlap = previous.slice(previous.indexOf(` ${firstWord} `) + 1);

      expect(chunks[i].pageContent.startsWith(overlap)).toBe(true);
      expect(overlap.length).toBeGreaterThanOrEqual(FIXED_OVERLAP_CHARS - 1);
      expect(overlap.length).toBeLessThanOrEqual(FIXED_OVERLAP_CHARS + longestWord);
    }
  });

  it("covers the whole text with no gaps", async () => {
    const chunks = await chunksPromise;
    const rebuilt = chunks[0].pageContent.split(" ");
    for (const chunk of chunks.slice(1)) {
      const chunkWords = chunk.pageContent.split(" ");
      const overlapWords = rebuilt.length - rebuilt.indexOf(chunkWords[0]);
      rebuilt.push(...chunkWords.slice(overlapWords));
    }
    expect(rebuilt).toEqual(words);
  });

  it("uses the chunk as its own parent and keeps metadata", async () => {
    const chunks = await chunksPromise;
    expect(chunks.map((c) => c.metadata.chunk)).toEqual(chunks.map((_, i) => i + 1));
    for (const chunk of chunks) {
      expect(chunk.metadata).toMatchObject({
        source: "report.pdf",
        loc: { pageNumber: 1 },
        chunking: "fixed-1000-200",
        parentText: chunk.pageContent,
      });
    }
  });

  it("returns one chunk for a short page", async () => {
    const chunks = await chunkDocuments([page("  just a   few words  ")], "fixed");
    expect(chunks.map((c) => c.pageContent)).toEqual(["just a few words"]);
  });

  it("makes no chunk for an empty or whitespace-only page", async () => {
    expect(await chunkDocuments([page(""), page(" \n\t ")], "fixed")).toEqual([]);
  });

  it("terminates on a single word longer than the chunk size", async () => {
    const huge = "z".repeat(FIXED_CHUNK_CHARS + 500);
    expect((await chunkDocuments([page(huge)], "fixed")).map((c) => c.pageContent)).toEqual([huge]);

    const chunks = await chunkDocuments([page(`before ${huge} after`)], "fixed");
    expect(chunks.map((c) => c.pageContent)).toEqual(["before", huge, "after"]);
  });
});

describe("chunkDocuments default", () => {
  it("uses fixed-size chunking, the strategy the evaluation picked", async () => {
    const chunks = await chunkDocuments([page("A short page of text that fits in one chunk.")]);
    expect(chunks[0].metadata.chunking).toBe("fixed-1000-200");
  });
});
