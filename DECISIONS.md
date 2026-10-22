# Engineering decisions

This document records the important design decisions in SourceSage: what was wrong before, what was changed, and why. Each entry also lists the trade-off, since every decision costs something.

It has two parts:

- **[Part 1](#part-1-fixing-what-was-broken)** fixes the first version. It listed many advanced RAG techniques, but several didn't actually work: the reranker never ran, the keyword search wasn't BM25, the evaluation was fake, and large documents crashed on upload.
- **[Part 2](#part-2-decisions-made-by-measuring)** covers what came after a real evaluation existed: every retrieval choice was measured, and some earlier choices were reversed because the numbers said so.

# Part 1: Fixing what was broken

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

> **Later:** sentence-window chunking is no longer the default. When measured, fixed-size chunks retrieved more of the evidence (see [entry 15](#15-chunking-sentence-window--fixed-size-a-reversal)).

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

## Latency at the end of Part 1

Measured from the `timingsMs` the API returns, on a 146-chunk document (Qdrant Cloud, Cohere trial key, Groq):

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

---

# Part 2: Decisions made by measuring

All numbers below come from `eval/` (method in [eval/README.md](eval/README.md), full tables in [eval/RESULTS.md](eval/RESULTS.md)). There are 56 questions over two public PDFs. Retrieval is scored on the 48 answerable questions; with that many, one question is about 2 percentage points, so only differences well above that drove decisions.

## 13. Build a real evaluation before changing anything else

**Before.** Every pipeline stage had been added because it's a "known RAG technique", and none had been measured. The old eval page graded the expected answer against itself (Part 1, entry 10).

**Decision.** A labeled dataset and a harness (`npm run eval`).

**Dataset:**
- **56 questions** over two documents of different kinds: the NIST AI Risk Management Framework (policy, repetitive vocabulary, codes like `GOVERN 1.7`) and the "Lost in the Middle" paper (numbers, results spread across sections).
- **Question types:** 28 factual (worded differently from the document), 10 exact-keyword, 10 multi-part, and 8 unanswerable.
- **Labels are exact evidence quotes, not chunk IDs**, so one set of labels scores any chunking strategy. A script checked that every quote occurs exactly once in the document.

**Harness:**
- **Same code as the app.** Each config calls the app's own `retrieveChunks()`/`answerQuestion()` with stages switched through `PipelineOptions`, so nothing is re-implemented.
- **Strict mode.** A failed stage is recorded as an error instead of quietly falling back, which would skew the scores.
- **A different judge model.** Answers are graded by `gpt-oss-20b` while `gpt-oss-120b` generates them, so no model grades its own work.
- **A disk cache for every API response.** Re-runs are free and reproducible, which matters on free API tiers.
- **Hash-checked PDFs.** They're downloaded at run time and checked against a SHA-256 hash.

**Trade-off.** Two documents and 56 questions is a small sample. It's enough to catch large effects, not 1–2 point ones.

## 14. Score what the LLM actually sees, not the chunk that was searched

**Problem found while comparing chunking strategies.** Recall was first scored on the *searched* unit. For sentence-window that's a 2-sentence child (median ~290 characters); for fixed chunks it's ~1000 characters. Five big chunks naturally contain more evidence than five small ones, so the comparison was unfair.

**Decision.** Also score recall on the text the generator receives: the parent window for sentence-window, the chunk itself otherwise. Report the average context size alongside it, so recall is always read next to how much text it took.

**Why it matters.** Without this, the chunking decision below could have been an artifact of chunk size. With it, the comparison is about equal amounts of context.

## 15. Chunking: sentence-window → fixed-size (a reversal)

**Before.** Part 1 kept the project's sentence-window chunking: search 2-sentence children, give the LLM 6-sentence parents.

**Measured** (hybrid + rerank, scored on the LLM's context):

| Strategy | Evidence recall | Multi-hop recall | MRR | Context the LLM gets | Upload cost |
|---|---:|---:|---:|---:|---|
| Sentence-window | 86.1% | 73.3% | 0.700 | ~5,600 chars | normal |
| **Fixed ~1000 chars, 200 overlap** | **93.1%** | **76.7%** | 0.864 | ~4,650 chars | normal |
| Semantic (cut where sentence embeddings change topic) | 88.9% | 76.7% | **0.882** | ~3,900 chars | ~2× embedding calls |

**Decision.** Fixed-size chunking is the default (`DEFAULT_CHUNKING`). It finds the most evidence while giving the LLM *less* text than sentence-window. The other two remain available through `chunkDocuments(docs, strategy)`.

**Why sentence-window lost.** A 2-sentence child is a weak search target. Its embedding carries little context, and BM25 has few words to match. The wide parent window helps the LLM, but only after the right child was found.

**Why not semantic.** It ranks the first relevant chunk slightly higher and uses the least context. But it missed more evidence, it embeds every sentence once at upload just to find cut points (about twice the embedding calls), and its threshold (25th percentile of neighbouring-sentence similarity) is one more thing to tune.

**Trade-off.** Fixed chunks can cut through a sentence or table. The 200-character overlap and the evidence-split rule in the metric cover most of that.

## 16. Hybrid search and reranking: keep both, for different reasons

**Measured** on fixed chunks:
- **Hybrid search finds more.** Recall rises from 81.9% (dense only) and 77.4% (BM25 only) to 93.1%.
- **On small sentence-window chunks the split was sharper.** On exact-keyword questions dense found 80% and BM25 100%, and hybrid kept the 100%.
- **The reranker didn't change recall at all** (93.1% → 93.1%), but **it raised MRR from 0.808 to 0.864**. It doesn't find new evidence; it moves the right chunk to the top.

**Decision.** Keep both.
- **Hybrid** is what finds the evidence.
- **The reranker earns its 0.4–3 s through ranking.** "Lost in the Middle" (one of the eval documents) shows that LLMs use information at the start of the context best, so a better first position is worth having even when the top-5 set is the same.

## 17. Multi-query rewriting → query decomposition

**Measured** on fixed chunks, hybrid + rerank:

| Config | Recall | Multi-hop recall | MRR |
|---|---:|---:|---:|
| no rewriting | 93.1% | 76.7% | 0.864 |
| + multi-query (3 LLM rewrites) | 93.1% | 76.7% | 0.864 |
| **+ decomposition** | **96.2%** | **91.7%** | 0.829 |

**Before.** Every question cost an LLM call for 3 rephrasings. On fixed chunks that changed nothing. On sentence-window chunks it helped a little, and only together with the reranker.

**Decision.** `decompose: true, multiQuery: false` in `DEFAULT_PIPELINE`. Multi-part questions were the weakest category in every config (60–77% recall). Decomposition splits them into standalone sub-questions, searches and reranks each one separately, and merges the results by taking turns, so every part gets a share of the top 5. It's the same single LLM call per question, now aimed at the actual weak spot.

**Design details:**
- **Merge by turns, not by score.** Rerank scores from different sub-questions aren't comparable, and one easy part could otherwise take every slot.
- **Single-part questions** pass through unchanged.
- **Rewrites aren't generated per sub-question.** That would cost 2–3 more LLM calls per question.

**Trade-off.**
- **Slightly lower MRR** (0.864 → 0.829): the merge by turns puts one part's best chunk at position 2.
- **More Cohere calls:** a split question makes one search and one rerank per part.
- **One regression:** the LLM wrote the sub-question "What range of configurations does Appendix C say *these* can take?", which can't stand on its own. It's caught by the prompt rule "repeat names instead of using *it*/*they*", but not always.

## 18. Contextual chunk headers: tried, not adopted

**Idea.** A page-level version of Anthropic's "contextual retrieval". A small LLM writes 1–2 sentences per page ("NIST AI RMF, Section 5.1 GOVERN, covers …"), which are prepended to each chunk *for search only*. That's one call per page instead of Anthropic's one call per chunk with the whole document, which wouldn't fit Groq's free-tier 8K-tokens-per-minute limit.

**Measured** on fixed chunks:

| Config | Without headers | With headers |
|---|---:|---:|
| dense recall / MRR | 81.9% / 0.685 | 75.7% / 0.575 |
| hybrid recall / MRR | 93.1% / 0.808 | 93.1% / 0.774 |
| hybrid + rerank + decompose recall / MRR | 96.2% / 0.829 | 97.2% / 0.842 |

**Decision.** Keep it off. `addContextHeaders()` stays as an option, and the eval runs it with `--headers`.

**Why.**
- **It hurts plain vector search.** Every chunk on a page gets the *same* header, so their embeddings move closer together and become harder to tell apart. Anthropic's per-chunk context avoids this because each chunk's context is different.
- **The gain in the full pipeline is one evidence quote,** which is within noise.
- **It costs one LLM call per page at upload,** minutes for a long PDF on a free tier.

**Note.** The headers for this run were generated with `qwen/qwen3.8-27b`, because `gpt-oss-20b` had hit Groq's daily token cap (entry 22).

## 19. Embedding model and vector size

**Before.** `embed-english-v3.0` at 1024 dimensions was hard-coded.

**Decision 1: make the model and size configurable.** `EMBEDDING_MODEL` and `EMBEDDING_DIM` support Cohere's newer `embed-v4.0`, whose Matryoshka embeddings can be cut to 256–1536 dimensions.
- **Each collection records its model and size in Qdrant collection metadata.** Queries and later uploads read it from there, so changing the setting can never mix two vector spaces in one collection.
- **A size mismatch fails with a clear error** instead of returning silently wrong results.
- **One gotcha:** the Qdrant JS client's `createCollection()` drops the `metadata` field, so the raw API call is used.

**Measured** on fixed chunks. Dense and hybrid only; rerank configs were skipped to save the Cohere trial budget.

| Model @ dimensions | Dense recall / MRR | Hybrid recall / MRR | Storage per vector |
|---|---:|---:|---:|
| **`embed-english-v3.0` @1024 (kept)** | 81.9% / 0.685 | **93.1%** / 0.808 | 4 KB |
| `embed-v4.0` @1024 | 85.1% / 0.694 | 92.4% / 0.790 | 4 KB |
| `embed-v4.0` @256 | 86.1% / 0.649 | 91.3% / 0.809 | 1 KB |

**Decision 2: keep v3 @1024 as the default.**
- **On its own, v4 is the better embedding model** (+3–4 points of dense recall).
- **But SourceSage never uses dense search alone.** In hybrid search BM25 already covers what v3 misses, and the difference is under one question.
- **Switching would buy nothing** while invalidating the cached baseline.

**The useful finding is the size trade-off.** At 256 dimensions v4 keeps ~98% of the hybrid recall in a quarter of the storage. That's the setting to choose if vector storage or memory becomes the constraint, for example many large notebooks on Qdrant Cloud's free tier.

## 20. PDF text cleanup: rejoin words split at line breaks

**Problem found while writing evaluation questions.** PDF extraction leaves line-break hyphens: "abil- ity", "an- swering". Neither BM25 nor the embedding model can match "abil-" to "ability".

**Decision.** `cleanPageText()` rejoins `word- word` when both sides are lowercase.

**A bug in the first version of the fix.** It turned "best- and worst-case" into "bestand", because that hyphen is a real *suspended* hyphen. Hyphens followed by *and/or/to* are now kept. This was found by reviewing the evaluation's evidence quotes, and is covered by unit tests.

**Trade-off.** A real compound split at a line end ("key- value") becomes "keyvalue". Without a dictionary there's no way to tell it apart from a split word.

## 21. The judge was wrong about refusals; refusals are scored in code

**Problem found by spot-checking judgments.**
- **A correct refusal scored as wrong.** For an unanswerable question, the system said "I could not find that in the uploaded document", which is correct, and the judge scored it 0.
- **Refusals counted as hallucinations.** The judge listed the refusal sentence itself as an "unsupported claim", so every refusal scored 0% faithfulness.

**Decision.** Plain refusals aren't sent to the judge. A refusal is correct exactly when the question is unanswerable, and it makes no claims to check. This is deterministic and also saves judge tokens.

**Effect** on the naive config: correctness 88.4% → 90.2%, faithfulness 92.4% → 98.5%, correct abstention 87.5% → 100%. Answers with partial refusals ("the document doesn't say X, but …") still go to the judge.

**Lesson.** Read a sample of LLM-judge outputs before trusting the averages.

## 22. Free-tier limits shaped the evaluation

**Limits found the hard way:**
- **Groq:** 8K tokens/minute and **200K tokens/day per model**. The daily cap isn't visible in response headers; it showed up as errors.
- **Cohere trial key:** 10 rerank calls/minute and 1,000 calls/month.

**Decisions:**
- **Cache every API response on disk**, keyed by the exact request. Configs that share stages (e.g. the same query rewrites) reuse results, and re-scoring an experiment costs nothing.
- **Agents and smoke tests ran on 2–3 questions only.** All full measurements were run one at a time, so jobs didn't throttle each other.
- **The full answer-quality comparison is spread over days.** One run of the full pipeline on 56 questions needs ~400K tokens, twice the daily cap. The naive baseline is complete; the rest is pending (see entry 25).

**Lesson.** A pipeline with 5 LLM calls per question is expensive to evaluate as well as to run. That's another reason to remove stages that don't earn their place.

## 23. Experiments must not inherit app defaults

**Problem.** The retrieval configs were built by spreading `DEFAULT_PIPELINE` and overriding a few fields. When decomposition became the default, the context-header run (entry 18) silently ran *with* decomposition under the old config names.

**Decision.**
- **Every eval config now lists all of its options.** Only the config named `full` deliberately tracks the app default.
- **Each config's options are saved in the results file.** That is how the problem was detected. The affected results were relabeled `+decompose` rather than discarded.

## 24. Product features: notebooks, follow-ups, live progress

**Multi-document notebooks.**
- **Before:** one document per session; uploading a new file replaced the old one.
- **Now:** a notebook is one Qdrant collection, and each upload is added to it with a `documentId` tag. If indexing fails partway, only that document's points are deleted. Sources show their file name, and the generator's prompt labels each source with its file.
- **Why one collection, not one per document:** BM25's IDF and RRF fusion work best over one shared index, and it's one query instead of N.

**Follow-up questions.**
- **How it works:** the last 3 exchanges are sent with each question. A small model (`gpt-oss-20b`) rewrites the follow-up into a standalone question, which is used for retrieval, generation and the grounding check.
- **History is never given to the answer generator,** so an earlier (possibly wrong) answer can't become a source.
- **Cost:** one extra LLM call, made only when there is history.

**Live progress instead of token streaming.**
- **How it works:** `/api/ask` streams NDJSON stage events ("Searching…", "Checking grounding…") and then the result.
- **Why not stream tokens:** the grounding check can reject a draft and regenerate it, so streamed text could be replaced in front of the user.

## 25. What isn't measured yet

- **End-to-end answer quality for every config.** Only the naive baseline is complete:
  - 90.2% correctness
  - 98.5% faithfulness
  - 100% correct "not found" on unanswerable questions
  - 8.3% false refusals

  The `gpt-oss-120b` generator is strong enough to answer well from mediocre retrieval. The open question is whether relevance grading, compression and the grounding check (~3 extra LLM calls, ~3–4 s) improve on that enough to justify their cost. Commands:

  ```bash
  npm run eval -- generation --configs=naive,hybrid+rerank,full,full-threshold,full-no-compress,lean-threshold,full-multiquery
  npm run eval -- latency --configs=naive,hybrid+rerank,full,full-threshold,full-no-compress,lean-threshold
  npm run eval -- report
  ```

  `full-threshold`, `full-no-compress` and `lean-threshold` are the cheaper alternatives from entry 28. Configs run in the order listed, so the least important one (`full-multiquery`) comes last.

  Expect this to take more than one day on Groq's free tier; the cache lets it resume where it stopped.
- **Latency is now measured, but only on 6 questions** (uncached, `npm run eval -- latency`):

  | Config | p50 | p90 |
  |---|---:|---:|
  | naive (dense search → generate) | 2.7 s | 3.3 s |
  | full (current default) | 6.5 s | 14.3 s |

  - **The slow p90 is a multi-part question.**
  - **Per-stage means for `full`:** search 1.6 s, rerank 1.4 s, decomposition 0.8 s, relevance grading 1.3 s, compression 1.0 s, generation 0.7 s, grounding check 0.8 s.
  - **The three post-retrieval LLM checks are about half the latency.** That's why the pending answer-quality comparison matters. If `hybrid+rerank` (no grading, compression or grounding check) answers about as well as `full`, the next step is to make those stages conditional, e.g. replace the grader with a rerank-score threshold. Entry 28 adds that threshold and the configs that test each cheaper alternative.

## 26. Running a public demo on free tiers

**Before.** Anyone with the URL could upload and ask without limit. Every question costs ~5 Groq LLM calls and at least 2 Cohere calls (embed + rerank), against 200K Groq tokens per model per day and ~1,000 Cohere trial calls per month (entry 22), so a handful of curious visitors, or one script, could exhaust the quota for everyone. Notebooks were only deleted when the user pressed "New notebook"; closing the tab left the collection in the free Qdrant cluster forever.

**Decision.** A `DEMO_MODE=true` switch, off by default so local development and the evaluation are unaffected:
- **Per-visitor daily caps** keyed by client IP (first `x-forwarded-for` entry, set by Vercel): 3 uploads and 20 questions per UTC day by default, plus a global cap of 25 questions per day across all visitors: one question uses ~7K Groq tokens and the free tier allows 200K per day, which the evaluation also draws on. Over the limit the route returns HTTP 429 with a readable message, before any API call is made, and the UI shows it as a notice and puts the question back in the input box.
- **4 MB upload limit**, checked in the browser before sending and again on the server. Vercel rejects bodies above ~4.5 MB with a bare 413 before the app runs, so the app's own limit sits just below it and can explain itself.
- **Notebooks expire after 24 h** (`NOTEBOOK_TTL_HOURS`). The upload route writes `createdAt` into the collection's Qdrant metadata (merged with the embedding config already there, entry 19). A cleanup runs in the background on upload, at most once an hour per instance. It only considers collections with the app prefix, explicitly skips `_eval_` collections, and stamps any notebook without `createdAt` with the time it first sees it, so it expires one TTL later. It never deletes one immediately.
- **A note under the upload box** states the limits, the expiry, and asks visitors not to upload private documents.

**Why.** It's the smallest change that protects the quota with no new service and no new dependency. The limiter and the expiry rules are pure functions with offline unit tests, including the eval-collection guard.

**Trade-off.**
- **The counters live in memory, so the limits are best-effort.** On Vercel each warm instance keeps its own counts, and they reset on a cold start or redeploy, so a visitor can get more than the cap. Strict limits would need a shared store (e.g. Upstash Redis's free tier with a per-day counter keyed the same way). For a portfolio demo, stopping casual overuse and runaway loops is enough.
- **IP is a coarse identity.** Visitors behind one NAT share a cap, and anyone with many IPs gets many caps. The global question cap bounds the worst case.
- **Cleanup only runs when someone uploads.** With no uploads, expired notebooks wait for the next one. It runs during the upload request (which takes seconds), not after the response, because serverless functions may be frozen once they respond.

## 27. Split the pipeline into modules

**Before.** All of the RAG code lived in one ~1,600-line `lib/rag.ts`: config, Cohere calls, PDF parsing, chunking, BM25, Qdrant storage, context headers, retrieval, grading and compression, generation, chat follow-ups and the pipeline itself. That came from the original single-file design, and it was kept through the fixes above so that each change stayed a small, reviewable diff.

**Problems.**
- **Hard to read.** Finding how one stage works meant scrolling past every other stage.
- **Hard to test in isolation.** Every test imported the whole pipeline, including the Qdrant and Groq clients.
- **Parallel work conflicted.** Changes to unrelated stages (say, chunking and generation) touched the same file and kept producing merge conflicts.

**Decision.** `lib/rag/` with one module per concern: `types`, `config`, `llm`, `cohere`, `parsing`, `chunking`, `bm25`, `store`, `context-headers`, `chat`, `retrieval`, `postprocess`, `generation` and `pipeline` (see the README's project structure). `lib/rag/index.ts` re-exports exactly the old public API, so `@/lib/rag` and `../lib/rag` imports in the app, the eval harness and the tests didn't change. Imports only point "downwards" (`types` and `config` import nothing from the others), so there are no circular imports.

**How it was verified.** The code was moved, not rewritten: only imports and `export` on helpers now shared between modules changed. All 55 unit tests, `tsc` and `next build` pass. The retrieval eval was re-run from the response cache: `hybrid+rerank+decompose` reproduces the saved numbers exactly, and `hybrid` and `hybrid+rerank` have identical hit and recall. The only difference is a small MRR shift on `hybrid`, which the unchanged pre-refactor code shows too between runs: Qdrant doesn't order tied RRF scores deterministically.

**Trade-off.** Fifteen files instead of one, and a reader has to follow imports between them. The header comment on each module says what it owns, to keep that cheap.

## 28. Cheaper alternatives to the post-retrieval LLM checks

**Problem.** After retrieval, the default pipeline makes three more LLM calls: relevance grading (~1.3 s), compression (~1.0 s) and the grounding check (~0.8 s). That's about half of the 6.5 s p50 (entry 25) and most of the Groq tokens per question. Whether they improve the answers is what the pending answer-quality run will show. This entry makes the cheaper alternatives available and measurable before that run, **without changing the default.**

**Decision 1: a relevance filter that needs no LLM.** `PipelineOptions.grade` now takes a mode as well as a boolean:

| `grade` | Filter |
|---|---|
| `true` or `"llm"` | The LLM grader (CRAG), as before. `DEFAULT_PIPELINE` still uses `true`. |
| `"rerank-threshold"` | Drops chunks whose Cohere rerank score is below `rerankThreshold` (default `DEFAULT_RERANK_THRESHOLD` = 0.30). No LLM call; it takes no measurable time. |
| `false` or `"off"` | Keeps every retrieved chunk. |

- **Same "not found" path.** If no chunk passes, the app answers "not found" without generating, exactly as when the LLM grader rejects everything.
- **Fails open.** A chunk with no rerank score (reranking off, or the reranker failed and the app fell back to fusion order) is kept, just as a chunk the LLM grader couldn't grade is kept.
- **Why widen `grade` instead of adding a `gradeMode` field next to it.** One field means there's one source of truth, with no meaningless combinations like `grade: false, gradeMode: "llm"`. Booleans keep their old meaning, so existing callers, eval configs and the options saved in old result files all still read correctly.

**Decision 2: pick the threshold from data already on disk.** `npm run eval -- filter` reads the reranked top-5 that the retrieval suite saved for each answerable question, together with each chunk's rerank score. It then:

1. **Looks up each chunk's full text by its chunk number.** The saved text is cut to 200 characters, too little to find evidence quotes in. Each lookup checks that the saved prefix still matches the chunk.
2. **Labels a chunk relevant iff it contains an evidence quote.**
3. **Sweeps the thresholds.**

It makes no API call, so it costs nothing.

**Why not replay retrieval from the response cache instead.** That was tried first, with a cache-only guard that refused any request not already cached. Only 16 of the 56 questions could be replayed. The reason: Qdrant doesn't order tied RRF scores the same way on every call (entry 27). The same question, searched 4 times in a row, gave 2–4 different candidate orders, and sometimes a different 20th candidate. So the request sent to the reranker, and with it the cache key, changes from run to run.

**Measured** on fixed chunks, `hybrid+rerank+decompose` (the default retrieval): 48 answerable questions, 240 top-5 chunks, 65 of them relevant. `hybrid+rerank` gives nearly identical numbers (full tables in `eval/RESULTS.md`).

| Threshold | Relevant chunks kept | Irrelevant chunks dropped | Chunks kept / question | Answerable questions emptied (wrong "not found") | Answerable questions that lose all evidence |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 100.0% | 0.0% | 5.00 | 0 | 0 |
| 0.10 | 100.0% | 9.1% | 4.67 | 0 | 0 |
| 0.20 | 98.5% | 12.0% | 4.54 | 0 | 0 |
| 0.25 | 98.5% | 17.7% | 4.33 | 0 | 0 |
| **0.30** | **98.5%** | **24.6%** | **4.08** | **0** | **0** |
| 0.35 | 98.5% | 28.0% | 3.96 | 0 | 0 |
| 0.40 | 95.4% | 34.3% | 3.69 | 1 | 1 |
| 0.45 | 93.8% | 41.7% | 3.40 | 1 | 2 |
| 0.50 | 93.8% | 49.1% | 3.13 | 1 | 2 |
| 0.60 | 83.1% | 61.1% | 2.54 | 4 | 5 |

**Threshold: 0.30.**
- **It costs no answerable question its evidence.** The only relevant chunk it drops (score 0.12) belongs to a question that has other evidence in its top 5.
- **It drops about a quarter of the irrelevant chunks.**
- **The first loss comes at 0.40.** The weakest question (`litm-09`) has its best evidence chunk at 0.374.
- **0.35 is equally clean on this data,** but it's too close to that first loss for a sample of two documents.

**What the sweep can't show.**
- **The cut-off is a coarse guard against off-topic chunks, not a replacement for the LLM grader's precision.** Many "irrelevant" chunks are on-topic passages that just don't contain the quote (their median score is 0.51), so no threshold separates them cleanly.
- **Unanswerable questions aren't covered.** The retrieval suite skips them, so it's unknown how often the cut-off empties their top 5 and short-circuits to "not found". On those questions the generator already refuses on its own (naive baseline: 100% correct abstention, entry 25), so the threshold mainly matters for the answerable ones.

**Decision 3: generation configs for the pending run.** Every config lists all of its options (entry 23), with the threshold written as the number 0.3, not as the app constant:

| Config | Relevance filter | Compression | Grounding check | LLM calls after retrieval |
|---|---|---|---|---:|
| `full` (default) | LLM | yes | yes | 4 (+1 retry, +1 recheck) |
| `full-threshold` | rerank ≥ 0.3 | yes | yes | 3 (+2) |
| `full-no-compress` | LLM | no | yes | 3 (+2) |
| `lean-threshold` | rerank ≥ 0.3 | no | no | 1 |

All four use the default retrieval (hybrid + rerank + decomposition), so any difference between them comes from the post-retrieval stages. `naive` and `hybrid+rerank` remain the lower baselines.

**The plan, and what would change the default.** Run the commands in entry 25. With 48 answerable questions, one question is ~2 points, so "the same" means within about 2 questions (~4 points):
- **If `full-threshold` matches `full`** on correctness, false refusals and correct abstention, the default becomes `grade: "rerank-threshold"`. That saves ~1.3 s and one LLM call per question.
- **If `full-no-compress` matches `full`,** compression goes. That saves ~1.0 s and up to 5 parallel calls. Compression can also cut out a sentence the answer needed, so it could even lose.
- **If `lean-threshold` matches `full` on correctness and faithfulness,** the default becomes lean: about 3 s and 3+ LLM calls saved per question. The grounding check stays only if `full` is clearly more faithful (fully-faithful rate higher by more than ~4 points), since its job is catching hallucination rather than raising correctness.
- **Any config that lowers correct abstention on unanswerable questions doesn't become the default.** Saying "not found" correctly is the product's promise.
- **Latency** (`npm run eval -- latency`) confirms the savings on the same 6 questions as before.
- **Watch the Cohere budget.** Because the RRF tie order changes between runs (see above), a generation run can't count on its rerank requests being cached. A config can make up to one rerank call per (sub-)question, about 75 for 56 questions, and the four configs above don't share them reliably. That is more than a trial key's remaining monthly calls. Making the order deterministic, by breaking ties by point id and sending the reranker its documents in a fixed order, would let configs share their rerank calls. That's a separate change, because it also changes every cache key once.

**Trade-off.**
- **One more option to maintain,** and a threshold tuned on two documents. Other document types may score differently under the reranker.
- **The cut-off is tied to `rerank-v3.5`.** Changing the rerank model means re-running the sweep.
- **The cut-off only works when reranking is on.** Without rerank scores it keeps everything.

## 29. Make search results repeatable: break ties by id

**Problem.** Reciprocal Rank Fusion often gives several chunks *exactly* the same score, and Qdrant returns tied points in no fixed order. The same question could therefore produce a different top-20 order on each call. Three things followed from that:
- **Small metric wobble:** plain hybrid MRR came out 0.808, 0.818 or 0.820 across identical runs of identical code.
- **Wasted calls:** the reranker received the same chunks in a different order, which is a different request, so it missed the eval cache. Only 16 of 56 questions replayed from cache, and each re-run spent new Cohere calls.
- **Inconsistency for users:** two people asking the same question could get differently ordered sources.

**Decision.** `hybridSearch()` fetches 10 extra candidates, sorts by score and then by point id (`orderFusedPoints()`), and keeps the top N. Fetching a few extra matters: ties at the cut-off are decided by id, not by whichever tied point Qdrant happened to return.

**Verified.** The same search, run 4 times for each of 3 test questions, now returns the identical order every time. A unit test checks that tie-breaking doesn't depend on input order.

**Trade-off.** Re-ordering the reranker's input changes its cache keys, so cached rerank results from earlier runs aren't reused once. From then on, configs that share retrieval also share their rerank calls.
