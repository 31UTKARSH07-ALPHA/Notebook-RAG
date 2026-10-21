# Engineering decisions

This document records the important design decisions in SourceSage: what was wrong before, what was changed, and why. Each entry also lists the trade-off, since every decision costs something.

The first version of this project listed many advanced RAG techniques, but several didn't actually work: the reranker never ran, the keyword search wasn't BM25, the evaluation was fake, and large documents crashed on upload. The fixes are below, roughly in order of importance.

---

## 1. Reranking: a local model that never loaded → Cohere Rerank API

**Before.** `rerank()` tried to load `cross-encoder/st5-base` with `@xenova/transformers` while `env.allowRemoteModels = false`. That model id doesn't exist in a transformers.js-compatible format, and remote downloads were blocked anyway, so loading always failed. The catch block then silently returned `docs.slice(0, topK)`: the first 5 results in their original order. Reranking was advertised but was a no-op on every request.

**Decision.** Use Cohere `rerank-v3.5` through the API (`lib/rag.ts`, `rerank()`). The top 20 fused candidates go in, the top 5 come out, and each keeps its `rerankScore`. If the API fails, the fusion order is used and the failure is logged.

**Why.**
- A cross-encoder reads the question and the chunk *together*, so it can judge whether the chunk actually answers the question. Embedding similarity only says the two texts are about similar things.
- The project already used Cohere for embeddings, so there's no new account or key.
- Running a transformer model inside a Next.js serverless function means a large bundle (onnxruntime), slow cold starts and webpack workarounds. Removing `@xenova/transformers` also removed the `onnxruntime-node` externals and `null-loader` hacks from `next.config.js`.

**Trade-off.** One more network call per question (about 0.4–3 s on a trial key) and a dependency on a paid API. A self-hosted cross-encoder would be cheaper at scale but isn't worth the operational cost for this project.

**Evidence.** For "What extension do I call if I lose my badge?", the correct chunk scored **0.712** and the next-best **0.015**. The reranker separates relevant from irrelevant chunks very clearly.

---

## 2. Keyword search: term frequency only → real BM25

**Before.** `buildSparseVector()` stored `count / totalTokens` for each word. That rewards common words: "the" in a chunk counted as much as "4410". There was no IDF (inverse document frequency, which boosts rare words), even though the comments called it BM25. Also, two different words with the same hash produced duplicate indices in one vector, which Qdrant rejects.

**Decision.** Split BM25 into its two halves and compute each where its data lives:
- **Term-frequency saturation and length normalization** are computed at upload time: `tf·(k1+1) / (tf + k1·(1 − b + b·len/avgLen))`, with k1 = 1.2 and b = 0.75. This only needs the chunk and the average chunk length, both known while indexing.
- **IDF** is applied by Qdrant at query time: the sparse vector is created with `modifier: "idf"`. Qdrant tracks how many chunks contain each word, so it computes IDF from its own counts.
- Counts are summed per hash index, so hash collisions can no longer produce duplicate indices.
- Tokens are split on punctuation instead of having it deleted, so "state-of-the-art" becomes four words rather than "stateoftheart".

**Why.** Dense embeddings are weak on exact strings: names, IDs, numbers, error codes, acronyms. That's the whole reason to run hybrid search, and the keyword side only helps if it actually scores rare words highly.

**Trade-off.** Requires Qdrant ≥ 1.10. Hashing words to 2²⁰ slots can still merge two rare words now and then; a real vocabulary would avoid that, but it would need storing and versioning.

---

## 3. Six Qdrant searches + fusion in JavaScript → one Query API call

**Before.** For each of 3 rewritten queries, the code embedded the query (3 Cohere calls) and ran a dense search and a sparse search (6 Qdrant calls), all one after another. It then fused the results in JavaScript with a hand-written RRF, followed by a "dedup" step that did nothing because RRF output is already unique. That step also read `docScores[payload.id]`, which was always `undefined` because the payload has no `id`.

**Decision.**
- All queries are embedded in **one** Cohere call.
- **One** Qdrant `query()` call uses `prefetch` (a dense and a sparse search for each query, up to 8 ranked lists) with `query: { fusion: "rrf" }`, so Qdrant does the Reciprocal Rank Fusion on the server (`hybridSearch()`).
- The hand-written RRF, the dedup step and the unused `ragFusion()` (which searched *whichever collection happened to be listed first*) were deleted.

**Why.** Fewer round trips and less code to get wrong. The algorithm (RRF with k = 60) is the same; it just runs where the data is.

**Trade-off.** Per-list weights (e.g. trusting dense more than sparse) are harder to customize than in a hand-written loop. If the evaluation ever shows weighting matters, Qdrant's DBSF fusion or a custom fusion step can be added back.

---

## 4. The original question was never searched

**Before.** Retrieval only used the 3 LLM rewrites. If the rewrites drifted (an LLM paraphrasing "4410" as "the security extension"), the user's exact wording was lost.

**Decision.** Always search `[original, ...rewrites]`, de-duplicated. If the rewrite call fails, it returns `[]` and the original question is searched alone. It no longer searches the same question three times.

**Why.** Rewriting should only *add* candidates, never replace the most reliable one.

---

## 5. Large documents crashed on upload → batched embeddings

**Before.** `embedTexts()` sent every chunk in a single Cohere request. Cohere accepts at most **96 texts per call**, and with 2-sentence chunks a 15–20 page PDF easily goes over that, so uploads failed.

**Decision.** `embed()` sends batches of 96. Documents and queries share the same function, but use `input_type: search_document` and `search_query` respectively (Cohere v3 embeds documents and queries differently and expects you to say which).

**Evidence.** A 146-chunk test document indexed successfully. Before the change it would have exceeded the limit.

**Related.** If indexing fails partway, the half-built collection is deleted, and uploading a new document deletes the previous one's collection. Before, every upload left a collection in Qdrant forever. Deletion only works on names with the app's own prefix, so a client can't make the server delete arbitrary collections.

---

## 6. Hallucination check existed but was never called → generate, verify, retry once

**Before.** `gradeAnswer()` was written but never used by `/api/ask`. When its output couldn't be parsed, it returned `grounded: true`, so a broken grader reported answers as verified.

**Decision.** A new `answerQuestion()` runs the full flow:
1. Retrieve, then generate.
2. Grade the answer against the sources.
3. If **not grounded**, regenerate once and put the grader's reasoning in the prompt ("a previous draft included unsupported claims: …"), then grade again.
4. Return the verdict (`grounded` / `not_grounded` / `unverified` / `not_found`) to the UI, which shows it as a badge under the answer.

A grader failure now returns `unverified` instead of a false "grounded".

**Why one retry?** A second attempt with specific feedback fixes most mistakes; further retries rarely help and add latency every time.

**Trade-off.** One extra LLM call on every answer, and two more when a retry happens. The UI shows the verdict instead of hiding an ungrounded answer, so the user decides.

---

## 7. "Not found" was sent to the LLM as if it were a document

**Before.** When the relevance grader rejected every chunk, `retrieveChunks()` returned a fake `Document` containing "I could not find that…". The generator was still called with it, and the UI showed it as a source.

**Decision.** Return an empty list. `answerQuestion()` then answers "not found" immediately: no generation call and no fake sources.

**Evidence.** "What is the company's stock price?" (not in the document) returned "not found" in about 4 s and skipped the generation and grounding steps.

---

## 8. Context compression ran one chunk at a time → in parallel

**Before.** `compressContext()` made one LLM call per chunk in a `for` loop, so 5 chunks meant 5 calls in sequence. When a chunk had no relevant content, the literal text "No relevant content found." was passed to the generator as a source.

**Decision.** Run all compression calls at once with `Promise.all`. Drop chunks that come back empty. If *all* of them come back empty, keep the uncompressed chunks, because the grader already judged them relevant. The prompt now asks the model to *copy* sentences exactly, so compression doesn't reword the text.

**Trade-off.** Sending 5 requests at once can hit Groq rate limits on a free key. Each failure falls back to the original chunk, so the answer still works.

---

## 9. Chunks that swallowed a whole page

**Before.** Sentences were split on `. ! ?` followed by a capital letter. Tables, bullet lists and headings often have no punctuation, so a whole page could become one "sentence", making a huge chunk that the embedding model cuts off.

**Decision.** Any "sentence" over 500 characters is split on word boundaries (`splitLongSentence()`). The parent-window size was renamed from `WINDOW_SIZE = 5` to `CONTEXT_SENTENCES = 2`, because the old name was wrong: the code added 2 sentences on *each side*, which makes 6 sentences, not 5.

---

## 10. The evaluation was fake → removed until it can be done properly

**Before.**
- The eval page sent each question's **expected** answer as if it were the system's answer, with placeholder chunks like `"Test chunk 1"`. Nothing in it ran the real pipeline.
- `/api/evaluate` hard-coded chunk relevance to `0.8`.
- The dataset asked questions about the RAG system itself, with wrong facts (768 dimensions, text-embedding-004).
- The page's styles never loaded, because it used Tailwind classes and Tailwind wasn't installed.

**Decision.** Delete the eval page, both eval routes, the dataset and `@tremor/react`.

**Why.** A dashboard that shows invented numbers is worse than none. Anyone who reads the code finds out, and it damages trust in everything else. A real evaluation is the next milestone:
- a hand-labeled question set on real documents, including unanswerable and multi-part questions
- retrieval metrics (Recall@k, MRR)
- LLM-judge metrics (faithfulness, answer relevancy)
- an ablation table showing what each pipeline stage adds in quality vs. latency

---

## 11. LLM model: `llama-3.3-70b-versatile` → `openai/gpt-oss-120b`, configurable

**Before.** The model name was hard-coded. Groq retired `llama-3.3-70b-versatile`, and every request failed with `model_not_found`.

**Decision.** Default to `openai/gpt-oss-120b`, overridable with `GROQ_MODEL`. `maxTokens` went from 1200 to 4096, because gpt-oss is a reasoning model and its hidden reasoning tokens count toward that limit. The answer prompt now asks for `[1]`-style citations; this model otherwise produced `【SOURCE 1 | chunk 3】`.

**Why configurable.** Hosted model lineups change every few months; a model change shouldn't need a code change.

---

## 12. Docs, config and repo hygiene

| Before | After |
|---|---|
| README described a different app: Gemini, `RecursiveCharacterTextSplitter`, 1000/200 chunks | README matches the code, with a pipeline diagram and the reasoning for each step |
| UI sidebar said "Recursive chunks of 1000 characters with 200 overlap" | Describes the real pipeline |
| Error messages told users to check `GOOGLE_API_KEY` | Point to `COHERE_API_KEY` |
| 11 unused packages (`@langchain/community`, `pdf-parse`, `pdf2json`, `@xenova/transformers`, …) | Removed. 8 runtime dependencies remain. |
| `.env.example` was gitignored, and the local copy held real keys | Committed with placeholders only |
| `.DS_Store`, `tsconfig.tsbuildinfo`, `AGENT_HANDOFF.md` tracked in git | Untracked and gitignored |
| Empty `QDRANT_API_KEY=` was still sent as a key (Qdrant warned about it) | Empty means no key |
| Name: "Notebook RAG" / `notebooklm-rag-clone` | **SourceSage** |

**Security note.** Real Google and Qdrant API keys were committed in the first commit of this repo. Deleting the file later doesn't remove them from git history, so those keys must be treated as leaked: the Qdrant key was rotated, and the Google key (no longer used by the app) should be revoked. Lesson: make `.env*` ignored from the very first commit, and treat any key that was ever committed as leaked.

---

## Current latency (measured)

From the `timingsMs` the API now returns, on a 146-chunk document (Qdrant Cloud, Cohere trial key, Groq):

| Stage | Typical |
|---|---|
| Query rewrite | 0.4–0.7 s |
| Hybrid search (embed + Qdrant) | ~2.3 s |
| Rerank | 0.4–3 s (one cold call took 14 s) |
| Relevance grading | 0.9–1.6 s |
| Compression | 0.4–0.9 s |
| Generation | 0.4–1 s |
| Grounding check | 0.7–1.1 s |
| **Total** | **~8 s** (4 s when the answer is "not found") |

The pipeline is accurate but slow. Not every stage is needed for every question; the planned ablation will measure which stages pay for themselves. Candidates to cut or make conditional: skip query rewriting for short, specific questions, and replace the LLM relevance grader with a rerank-score threshold.
