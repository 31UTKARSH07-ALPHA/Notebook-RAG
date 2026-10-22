// Contextual chunk headers: one LLM call per page writes a short header that
// is prepended to the page's chunks for search (optional, used by the eval).

import { Document } from "@langchain/core/documents";
import { HumanMessage } from "@langchain/core/messages";
import { handleStageFailure } from "./config";
import { getUtilityModel, messageText } from "./llm";

const HEADER_PAGE_CHARS = 6000; // page text sent when writing a context header

// Contextual chunk headers
//
// A 2-sentence chunk like "It can be spent on courses and books." doesn't say
// what "it" is. For each page, a small LLM writes 1-2 sentences placing the
// page in the document (document name, section, topic), and that header is
// prepended to the page's chunks before embedding and BM25. This is a
// page-level version of Anthropic's "contextual retrieval": one call per page
// instead of one call per chunk with the whole document.
export async function addContextHeaders(pages: Document[], chunks: Document[]): Promise<void> {
  const llm = getUtilityModel();
  const opening = (pages[0]?.pageContent ?? "").split(" ").slice(0, 250).join(" ");
  const pageKey = (doc: Document) => doc.metadata.loc?.pageNumber ?? 1;
  const headers = new Map<number, string>();

  // A few pages at a time keeps upload fast without hammering rate limits
  const CONCURRENCY = 3;
  for (let i = 0; i < pages.length; i += CONCURRENCY) {
    await Promise.all(
      pages.slice(i, i + CONCURRENCY).map(async (page) => {
        const prompt = `Here is the beginning of a document:
<opening>
${opening}
</opening>

Here is one page of the same document:
<page>
${page.pageContent.slice(0, HEADER_PAGE_CHARS)}
</page>

Write 1-2 sentences that situate this page within the document, to improve search over its passages: name the document, then the section and the topics this page covers, using the document's own key terms. Output only the sentences.`;
        try {
          const response = await llm.invoke([new HumanMessage(prompt)]);
          const header = messageText(response.content).replace(/\s+/g, " ").trim();
          if (header) headers.set(pageKey(page), header);
        } catch (error) {
          handleStageFailure("Context header", error); // chunks just go without a header
        }
      })
    );
  }

  for (const chunk of chunks) {
    const header = headers.get(pageKey(chunk));
    if (header) chunk.metadata.contextHeader = header;
  }
}
