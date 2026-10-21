import { describe, expect, it } from "vitest";
import {
  MAX_SENTENCE_CHARS,
  cleanPageText,
  splitIntoSentences,
  splitLongSentence,
} from "../lib/rag";

describe("cleanPageText", () => {
  it("rejoins a word hyphenated at a line break", () => {
    expect(cleanPageText("The abil- ity to")).toBe("The ability to");
    expect(cleanPageText("the abil-\nity to")).toBe("the ability to");
  });

  it("keeps suspended hyphens before and/or/to", () => {
    expect(cleanPageText("best- and worst-case")).toBe("best- and worst-case");
    expect(cleanPageText("pre- or post-processing")).toBe("pre- or post-processing");
    expect(cleanPageText("up- to date")).toBe("up- to date");
  });

  it("leaves a hyphen after a capital letter alone", () => {
    expect(cleanPageText("AL- iBi")).toBe("AL- iBi");
  });

  it("leaves real compounds with no space after the hyphen alone", () => {
    expect(cleanPageText("long-context models")).toBe("long-context models");
  });

  it("collapses whitespace and trims", () => {
    expect(cleanPageText("  one\n\ntwo\t three  ")).toBe("one two three");
  });
});

describe("splitIntoSentences", () => {
  it("splits on sentence punctuation followed by a capital or quote", () => {
    expect(
      splitIntoSentences('First sentence is here. Second one follows! Is this the third? "Quoted one starts here."')
    ).toEqual([
      "First sentence is here.",
      "Second one follows!",
      "Is this the third?",
      '"Quoted one starts here."',
    ]);
  });

  it("does not split before a lowercase word", () => {
    expect(splitIntoSentences("Values such as e.g. lowercase words continue here.")).toEqual([
      "Values such as e.g. lowercase words continue here.",
    ]);
  });

  it("drops fragments of 10 characters or fewer", () => {
    expect(splitIntoSentences("This sentence is long enough. Page 4. Yes it is.")).toEqual([
      "This sentence is long enough.",
    ]);
  });

  it("splits long unpunctuated text into word-bounded pieces", () => {
    const words = Array.from({ length: 300 }, (_, i) => `cell${i}`);
    const pieces = splitIntoSentences(words.join(" "));

    expect(pieces.length).toBeGreaterThan(1);
    for (const piece of pieces) expect(piece.length).toBeLessThanOrEqual(MAX_SENTENCE_CHARS);
    // No word is cut in half and nothing is lost
    expect(pieces.join(" ").split(" ")).toEqual(words);
  });
});

describe("splitLongSentence", () => {
  it("returns a short sentence unchanged", () => {
    expect(splitLongSentence("A short sentence.")).toEqual(["A short sentence."]);
  });

  it("keeps a single over-long word as one piece", () => {
    const word = "x".repeat(MAX_SENTENCE_CHARS + 100);
    expect(splitLongSentence(word)).toEqual([word]);
  });

  it("fills each piece as far as the limit allows", () => {
    const word = "y".repeat(99); // 5 words + 4 spaces = 499 chars
    const pieces = splitLongSentence(Array(12).fill(word).join(" "));
    expect(pieces.map((p) => p.split(" ").length)).toEqual([5, 5, 2]);
  });
});
