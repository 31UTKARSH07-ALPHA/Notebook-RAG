import { describe, expect, it } from "vitest";
import { Document } from "@langchain/core/documents";
import {
  NOT_FOUND_ANSWER,
  formatSources,
  isNotFoundAnswer,
  normaliseCitations,
} from "../lib/rag";

describe("normaliseCitations", () => {
  it("rewrites gpt-oss style markers to [n]", () => {
    expect(normaliseCitations("Revenue grew 12%【1†L1-L4】.")).toBe("Revenue grew 12%[1].");
    expect(normaliseCitations("See 【SOURCE 2 | chunk 7】 and 【 source 3 】")).toBe("See [2] and [3]");
  });

  it("leaves normal [n] citations and plain text alone", () => {
    const answer = "Revenue grew 12% [1][2]. No markers here.";
    expect(normaliseCitations(answer)).toBe(answer);
  });
});

describe("isNotFoundAnswer", () => {
  it("recognises the refusal exactly and with small rewordings", () => {
    expect(isNotFoundAnswer(NOT_FOUND_ANSWER)).toBe(true);
    expect(isNotFoundAnswer("Sorry, I COULD NOT FIND THAT IN THE UPLOADED DOCUMENT")).toBe(true);
  });

  it("does not flag a real answer", () => {
    expect(isNotFoundAnswer("The report covers 2023 [1].")).toBe(false);
    expect(isNotFoundAnswer("I could not find that in the document.")).toBe(false);
  });
});

describe("formatSources", () => {
  it("prefers the child text and reads page, file name, chunk and rerank score", () => {
    const chunk = new Document({
      pageContent: "The wide parent window sent to the LLM.",
      metadata: {
        source: "report.pdf",
        chunk: 7,
        loc: { pageNumber: 3 },
        childText: "The small\n child   chunk.",
        rerankScore: 0.91,
      },
    });
    expect(formatSources([chunk])).toEqual([
      { fileName: "report.pdf", chunk: 7, page: 3, text: "The small child chunk.", rerankScore: 0.91 },
    ]);
  });

  it("falls back to page text, list position and the pdf page number", () => {
    const sources = formatSources([
      new Document({ pageContent: "first", metadata: {} }),
      new Document({ pageContent: "second", metadata: { pdf: { pageNumber: 5 } } }),
    ]);
    expect(sources).toEqual([
      { fileName: undefined, chunk: 1, page: undefined, text: "first", rerankScore: undefined },
      { fileName: undefined, chunk: 2, page: 5, text: "second", rerankScore: undefined },
    ]);
  });

  it("caps the text at 420 characters", () => {
    const [source] = formatSources([new Document({ pageContent: "x".repeat(1000), metadata: {} })]);
    expect(source.text).toHaveLength(420);
  });
});
