import { describe, expect, it } from "vitest";
import { Document } from "@langchain/core/documents";
import { interleaveRankings, orderFusedPoints, parseSubQuestions } from "../lib/rag";

const doc = (id: string) => new Document({ id, pageContent: id, metadata: {} });
const ids = (docs: Document[]) => docs.map((d) => d.id);

describe("parseSubQuestions", () => {
  it("parses numbered and bulleted sub-questions", () => {
    expect(parseSubQuestions("1. What is A?\n2) What is B?", "orig")).toEqual(["What is A?", "What is B?"]);
    expect(parseSubQuestions("- What is A?\n* What is B?", "orig")).toEqual(["What is A?", "What is B?"]);
  });

  it("keeps the original question when the output has fewer than two parts", () => {
    expect(parseSubQuestions("1. What is A and B?", "What is A and B?")).toEqual(["What is A and B?"]);
    expect(parseSubQuestions("Sorry, I can't help with that.", "orig")).toEqual(["orig"]);
  });

  it("drops duplicates and caps at three sub-questions", () => {
    const text = "1. What is A?\n2. What is A?\n3. What is B?\n4. What is C?\n5. What is D?";
    expect(parseSubQuestions(text, "orig")).toEqual(["What is A?", "What is B?", "What is C?"]);
  });
});

describe("interleaveRankings", () => {
  it("takes turns between lists so every sub-question gets a share", () => {
    const merged = interleaveRankings([[doc("a1"), doc("a2"), doc("a3")], [doc("b1"), doc("b2")]], 4);
    expect(ids(merged)).toEqual(["a1", "b1", "a2", "b2"]);
  });

  it("skips a chunk already taken and gives that list's turn to its next chunk", () => {
    const merged = interleaveRankings([[doc("x"), doc("a2")], [doc("x"), doc("b2")]], 3);
    expect(ids(merged)).toEqual(["x", "b2", "a2"]);
  });

  it("returns a single list unchanged up to topK", () => {
    expect(ids(interleaveRankings([[doc("a"), doc("b"), doc("c")]], 2))).toEqual(["a", "b"]);
  });

  it("stops when every list is exhausted", () => {
    expect(ids(interleaveRankings([[doc("a")], []], 5))).toEqual(["a"]);
  });
});

describe("orderFusedPoints", () => {
  it("breaks score ties by id so the order never depends on Qdrant", () => {
    const a = [
      { id: "c", score: 0.5 },
      { id: "a", score: 0.5 },
      { id: "b", score: 0.9 },
    ];
    const b = [a[1], a[2], a[0]];
    expect(orderFusedPoints(a, 3).map((p) => p.id)).toEqual(["b", "a", "c"]);
    expect(orderFusedPoints(b, 3)).toEqual(orderFusedPoints(a, 3));
  });

  it("cuts to the limit after sorting, so ties at the cut-off are decided by id", () => {
    const points = [
      { id: "z", score: 0.3 },
      { id: "y", score: 0.3 },
      { id: "x", score: 0.8 },
    ];
    expect(orderFusedPoints(points, 2).map((p) => p.id)).toEqual(["x", "y"]);
  });
});
