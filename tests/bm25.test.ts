import { describe, expect, it } from "vitest";
import {
  type SparseVector,
  buildDocumentSparseVector,
  buildQuerySparseVector,
  hashToken,
  tokenize,
} from "../lib/rag";

// Weight a document vector gives to one term
const weightOf = (vector: SparseVector, term: string) =>
  vector.values[vector.indices.indexOf(hashToken(term))];

describe("tokenize", () => {
  it("lowercases, strips punctuation and drops 1-character tokens", () => {
    expect(tokenize("What is BM25? A ranking-function, e.g. Okapi.")).toEqual([
      "what",
      "is",
      "bm25",
      "ranking",
      "function",
      "okapi",
    ]);
  });
});

describe("BM25 document vectors", () => {
  it("never repeats an index, even when two tokens hash to the same slot", () => {
    // Found by brute force over 5-letter words: both hash to 292897
    expect(hashToken("aaaaa")).toBe(hashToken("begeb"));

    const vector = buildDocumentSparseVector("aaaaa begeb other words", 4);
    expect(new Set(vector.indices).size).toBe(vector.indices.length);
    expect(vector.indices).toHaveLength(3);
    // The shared slot is weighted as if one term appeared twice
    expect(weightOf(vector, "aaaaa")).toBe(
      weightOf(buildDocumentSparseVector("aaaaa aaaaa other words", 4), "aaaaa")
    );
  });

  it("gives more occurrences a higher but sub-linear weight", () => {
    // Same length (4 tokens) so only term frequency changes
    const once = weightOf(buildDocumentSparseVector("alpha beta gamma delta", 4), "alpha");
    const twice = weightOf(buildDocumentSparseVector("alpha alpha gamma delta", 4), "alpha");
    const thrice = weightOf(buildDocumentSparseVector("alpha alpha alpha delta", 4), "alpha");

    expect(twice).toBeGreaterThan(once);
    expect(thrice).toBeGreaterThan(twice);
    expect(twice).toBeLessThan(2 * once);
    expect(thrice - twice).toBeLessThan(twice - once);
    expect(thrice).toBeLessThan(2.2); // saturates below k1 + 1
  });

  it("gives a longer chunk a lower weight for the same term frequency", () => {
    const short = weightOf(buildDocumentSparseVector("alpha beta", 4), "alpha");
    const long = weightOf(buildDocumentSparseVector("alpha beta gamma delta epsilon zeta", 4), "alpha");
    expect(long).toBeLessThan(short);
  });

  it("weights a term exactly 1 in a chunk of average length", () => {
    expect(weightOf(buildDocumentSparseVector("alpha beta gamma delta", 4), "alpha")).toBeCloseTo(1);
  });
});

describe("BM25 query vectors", () => {
  it("has one index per distinct term, each with value 1", () => {
    const vector = buildQuerySparseVector("What is BM25? bm25 scoring, aaaaa begeb");
    // what, is, bm25, scoring, and one shared slot for the colliding pair
    expect(vector.indices).toHaveLength(5);
    expect(new Set(vector.indices).size).toBe(5);
    expect(vector.values).toEqual([1, 1, 1, 1, 1]);
  });

  it("is empty for a query with no usable tokens", () => {
    expect(buildQuerySparseVector("? a !")).toEqual({ indices: [], values: [] });
  });
});
