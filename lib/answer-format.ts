// Turns the small subset of Markdown the LLM uses in answers (bold, bullet
// and numbered lists, headings, line breaks) into plain data the UI renders
// as React elements. Building elements instead of injecting HTML means text
// from the model or a document can never become markup.

export type Inline = { text: string; bold: boolean };
export type AnswerBlock =
  | { type: "paragraph"; lines: Inline[][] }
  | { type: "list"; ordered: boolean; items: Inline[][] };

const LIST_ITEM = /^\s*(?:([-*•])|(\d+)[.)])\s+(.*)$/;
const HEADING = /^\s*#{1,6}\s+(.*)$/;

export function parseInline(text: string): Inline[] {
  const parts: Inline[] = [];
  const pattern = /\*\*(.+?)\*\*/g;
  let last = 0;
  for (const match of text.matchAll(pattern)) {
    if (match.index > last) parts.push({ text: text.slice(last, match.index), bold: false });
    parts.push({ text: match[1], bold: true });
    last = match.index + match[0].length;
  }
  if (last < text.length) parts.push({ text: text.slice(last), bold: false });
  return parts;
}

export function parseAnswer(answer: string): AnswerBlock[] {
  const blocks: AnswerBlock[] = [];
  let paragraph: Inline[][] = [];

  const flushParagraph = () => {
    if (paragraph.length) blocks.push({ type: "paragraph", lines: paragraph });
    paragraph = [];
  };

  for (const line of answer.split("\n")) {
    const item = line.match(LIST_ITEM);
    if (item) {
      flushParagraph();
      const ordered = Boolean(item[2]);
      const previous = blocks[blocks.length - 1];
      if (previous?.type === "list" && previous.ordered === ordered) {
        previous.items.push(parseInline(item[3]));
      } else {
        blocks.push({ type: "list", ordered, items: [parseInline(item[3])] });
      }
      continue;
    }

    if (!line.trim()) {
      flushParagraph();
      continue;
    }

    const heading = line.match(HEADING);
    paragraph.push(heading ? [{ text: heading[1], bold: true }] : parseInline(line));
  }

  flushParagraph();
  return blocks;
}
