import { describe, expect, it } from "vitest";
import { Document } from "@langchain/core/documents";
import { type Question, chunkContainsEvidence, retrievalMetrics, sweepThresholds } from "../eval/lib";

describe("chunkContainsEvidence", () => {
  // 9 words, so at least 5 must be in the chunk when the quote is split
  const quote = "The quick brown fox jumps over the lazy dog.";

  it("matches a quote contained in the chunk", () => {
    expect(chunkContainsEvidence(`Intro text. ${quote} Outro text.`, quote)).toBe(true);
  });

  it("ignores case and punctuation", () => {
    expect(chunkContainsEvidence("THE QUICK, brown-fox jumps over (the) lazy dog!", quote)).toBe(true);
  });

  it("counts a quote split across a chunk boundary when at least half is present", () => {
    // Chunk ends with the first 5 words / starts with the last 5 words
    expect(chunkContainsEvidence("Earlier text. The quick brown fox jumps", quote)).toBe(true);
    expect(chunkContainsEvidence("jumps over the lazy dog. Later text.", quote)).toBe(true);
  });

  it("does not count a split quote when less than half is present", () => {
    expect(chunkContainsEvidence("Earlier text. The quick brown fox", quote)).toBe(false);
    expect(chunkContainsEvidence("over the lazy dog. Later text.", quote)).toBe(false);
  });

  it("does not count a partial quote in the middle of a chunk", () => {
    expect(chunkContainsEvidence("Text. The quick brown fox jumps over. More text.", quote)).toBe(false);
  });
});

describe("retrievalMetrics", () => {
  const question = (evidence: string[]): Question => ({
    id: "q1",
    doc: "doc",
    type: "factual",
    question: "?",
    reference_answer: "",
    evidence,
    page: [],
  });
  // Metrics read the indexed child text, not the wider parent window
  const chunk = (childText: string, parent = "unrelated parent window") =>
    new Document({ pageContent: parent, metadata: { childText } });

  const ranked = [
    chunk("Nothing useful here at all."),
    chunk("The budget was approved in March."),
    chunk("Nothing useful here either."),
    chunk("The project finished in October."),
  ];

  it("scores hit, full recall and the rank of the first relevant chunk", () => {
    const metrics = retrievalMetrics(
      question(["budget was approved in March", "project finished in October"]),
      ranked
    );
    expect(metrics).toEqual({ hit: 1, recall: 1, reciprocalRank: 1 / 2 });
  });

  it("gives partial recall when only some evidence is retrieved", () => {
    const metrics = retrievalMetrics(
      question(["project finished in October", "the team had twelve members"]),
      ranked
    );
    expect(metrics).toEqual({ hit: 1, recall: 0.5, reciprocalRank: 1 / 4 });
  });

  it("scores zero when no evidence is retrieved", () => {
    expect(retrievalMetrics(question(["the team had twelve members"]), ranked)).toEqual({
      hit: 0,
      recall: 0,
      reciprocalRank: 0,
    });
  });

  it("ignores evidence that is only in the parent window", () => {
    const parentOnly = [chunk("Nothing here.", "The team had twelve members.")];
    expect(retrievalMetrics(question(["the team had twelve members"]), parentOnly).hit).toBe(0);
  });
});

describe("sweepThresholds", () => {
  const questions = [
    // Evidence in the best chunk only
    {
      id: "a1",
      type: "factual" as const,
      chunks: [{ rerankScore: 0.9, relevant: true }, { rerankScore: 0.2, relevant: false }],
    },
    // Evidence in a weak chunk, behind a stronger irrelevant one
    {
      id: "a2",
      type: "multi_hop" as const,
      chunks: [{ rerankScore: 0.6, relevant: false }, { rerankScore: 0.35, relevant: true }],
    },
    { id: "u1", type: "unanswerable" as const, chunks: [{ rerankScore: 0.25, relevant: false }] },
  ];
  const [zero, mid, high] = sweepThresholds(questions, [0, 0.3, 0.5]);

  it("keeps everything at threshold 0", () => {
    expect(zero).toMatchObject({
      relevantKept: 1,
      irrelevantDropped: 0,
      meanChunksKept: 5 / 3,
      emptiedAnswerable: 0,
      evidenceLostAnswerable: 0,
      emptiedUnanswerable: 0,
    });
  });

  it("counts kept, dropped and emptied per threshold", () => {
    expect(mid).toMatchObject({
      relevantKept: 1,
      irrelevantDropped: 2 / 3,
      emptiedAnswerable: 0,
      evidenceLostAnswerable: 0,
      emptiedUnanswerable: 1,
    });
  });

  it("counts lost evidence even when other chunks survive", () => {
    // a2 keeps its irrelevant 0.6 chunk: not emptied, but its evidence is gone
    expect(high).toMatchObject({ relevantKept: 0.5, emptiedAnswerable: 0, evidenceLostAnswerable: 1 });
  });
});
