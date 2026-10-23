import { describe, expect, it } from "vitest";
import { parseAnswer, parseInline } from "../lib/answer-format";

describe("parseInline", () => {
  it("turns **bold** into bold runs and keeps the rest as text", () => {
    expect(parseInline("a subset of **500 examples**, each [2].")).toEqual([
      { text: "a subset of ", bold: false },
      { text: "500 examples", bold: true },
      { text: ", each [2].", bold: false },
    ]);
  });

  it("leaves text without markers alone, including a stray asterisk", () => {
    expect(parseInline("costs $6,000 * 2 [1]")).toEqual([{ text: "costs $6,000 * 2 [1]", bold: false }]);
  });
});

describe("parseAnswer", () => {
  it("groups consecutive bullet lines into one list", () => {
    const blocks = parseAnswer("Two reasons:\n- cost [1]\n- time [2]\nThat's all.");
    expect(blocks.map((b) => b.type)).toEqual(["paragraph", "list", "paragraph"]);
    expect(blocks[1]).toMatchObject({ type: "list", ordered: false, items: [[{ text: "cost [1]" }], [{ text: "time [2]" }]] });
  });

  it("recognises numbered lists and keeps them separate from bullets", () => {
    const blocks = parseAnswer("1. first\n2) second\n- other");
    expect(blocks).toMatchObject([
      { type: "list", ordered: true, items: [[{ text: "first" }], [{ text: "second" }]] },
      { type: "list", ordered: false },
    ]);
  });

  it("splits paragraphs on blank lines, keeps single line breaks, and bolds headings", () => {
    const blocks = parseAnswer("### Answer\nLine one\nLine two\n\nNext paragraph");
    expect(blocks).toEqual([
      {
        type: "paragraph",
        lines: [[{ text: "Answer", bold: true }], [{ text: "Line one", bold: false }], [{ text: "Line two", bold: false }]],
      },
      { type: "paragraph", lines: [[{ text: "Next paragraph", bold: false }]] },
    ]);
  });

  it("never produces markup from text that looks like HTML", () => {
    const blocks = parseAnswer("<script>alert(1)</script>");
    expect(blocks).toEqual([{ type: "paragraph", lines: [[{ text: "<script>alert(1)</script>", bold: false }]] }]);
  });
});
