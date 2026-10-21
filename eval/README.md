# Evaluation

This folder measures how well SourceSage finds the right passages and how good its answers are, so every pipeline stage has to prove it's worth its cost. Results are in [RESULTS.md](RESULTS.md).

## Dataset

**Documents.** Two public PDFs of different kinds (listed in [`documents.json`](documents.json)). They're downloaded at eval time and checked against a SHA-256 hash, so everyone evaluates the exact same files.

| Document | Kind | Why it's in the set |
|---|---|---|
| NIST AI Risk Management Framework 1.0 (48 pages) | Government policy framework | Long, repetitive vocabulary (every section talks about "AI risks"), and codes like `GOVERN 1.7` that test exact-keyword retrieval |
| "Lost in the Middle" (Liu et al., 18 pages) | Research paper | Numbers, model names and results spread between method and results sections; tables extract as jumbled text |

**Questions.** There are 56 questions in [`questions.json`](questions.json), 28 per document:

| Type | Count | What it tests |
|---|---:|---|
| `factual` | 28 | One fact, but **paraphrased**, so the question doesn't reuse the document's wording. Tests semantic retrieval. |
| `keyword` | 10 | Hinges on an exact identifier, number or name (`MANAGE 3.2`, `2655 queries`, `45.6%`). Tests whether keyword search earns its place. |
| `multi_hop` | 10 | Needs 2–3 facts from different sections or pages. Tests whether retrieval finds *all* the pieces. |
| `unanswerable` | 8 | On-topic and plausible, but the document doesn't contain the answer. Tests whether the system says "not found" instead of making something up. |

Each answerable question is labeled with **evidence quotes**: short exact quotes (6–20 words, within one sentence) that contain the answer. Every quote was checked by a script to occur exactly once in the extracted text. Labeling with quotes rather than chunk IDs means the same labels work for *any* chunking strategy, which is what makes the chunking comparison possible.

The questions were drafted with LLM help and then reviewed by hand, including a check that the unanswerable questions really have no answer in the text.

## Metrics

### Retrieval (`npm run eval -- retrieval`)
Scored on the top 5 chunks, which is what the generator sees. A chunk "contains" an evidence quote if the normalized quote appears in the chunk's indexed text. If a chunk boundary splits a quote in two, the chunk still counts when it holds at least half of the quote.

- **Hit@5:** at least one evidence quote is in the top 5.
- **Recall@5:** the share of a question's evidence quotes found in the top 5. For multi-part questions this is the important one: finding one of two facts isn't enough.
- **MRR@5:** mean reciprocal rank of the first relevant chunk (1.0 = always ranked first). This measures ranking quality, not just whether the chunk was found.

No LLM generation happens in this suite, so it's cheap and deterministic (the only LLM call is query rewriting, which is cached).

### Generation (`npm run eval -- generation`)
Every answer is graded by an **LLM judge** (`openai/gpt-oss-20b`). It's deliberately a different model from the generator (`gpt-oss-120b`), so the generator isn't grading its own work.

- **Correctness** (0 / 0.5 / 1) against the human reference answer. For unanswerable questions, 1 means the system correctly said the information isn't there.
- **Faithfulness:** the judge lists every factual claim in the answer and marks whether the *retrieved context* supports it. Faithfulness = supported claims / total claims. This catches hallucination even when the answer happens to be correct.
- **Correct abstention:** correctness on the unanswerable questions.
- **False refusal:** the share of answerable questions where the system said "not found". This is the cost side of aggressive filtering.
- **Context recall:** the share of evidence quotes that survived into the LLM's final context. It shows whether the relevance grader throws away chunks it shouldn't.

### Latency (`npm run eval -- latency`)
Real, uncached end-to-end timings on a fixed subset of questions (every 6th), with per-stage means. The harness pauses between questions so the measurement never includes Groq rate-limit back-off.

## How the harness works

- **Same code as the app.** Each config calls the app's own `retrieveChunks()` / `answerQuestion()` with stages switched on or off through `PipelineOptions`. There's no separate re-implementation that could drift from production.
- **Strict mode.** In the app, a failing stage (rerank, grading, …) falls back quietly. In the eval, a failure is recorded as an error, because a silent fallback would make a config look better or worse than it is.
- **Caching.** Every embedding, rerank and LLM response is cached on disk (`eval/.cache`, keyed by the exact request). Re-runs are free and reproducible, and configs that share a stage reuse the same result: the query rewrites used by `hybrid+multiquery` are the same ones used by `full`. This matters on free API tiers (Groq: 8K tokens per minute; Cohere trial: 10 reranks per minute and 1,000 calls per month).
- **Indexing.** Each document is indexed once per index variant (chunking strategy, `--headers`, `--embedding`) into its own Qdrant collection and reused on later runs. Without `--embedding` the baseline `embed-english-v3.0@1024` is used, regardless of `EMBEDDING_MODEL` in `.env.local`, and the variant name has no embedding suffix. Before running, the harness checks that every evidence quote can be found in at least one chunk; otherwise that question couldn't be scored.

## Running it

```bash
npm run eval -- retrieval                      # sentence-window chunking
npm run eval -- retrieval --chunking=fixed     # the original 1000/200-character chunking
npm run eval -- retrieval --embedding=embed-v4.0@512   # another embedding model / vector size
npm run eval -- generation
npm run eval -- latency
npm run eval -- report                         # rebuild RESULTS.md

# Options: --configs=dense,hybrid  --questions=nist-01,litm-05  --limit=5
```

Needs the same `.env.local` as the app.

## Limitations

- **Small sample.** With 48 answerable questions, one question is about 2 percentage points. Differences of a few points between configs are within noise; only larger gaps, or gaps that hold across both documents, should drive decisions.
- **Judge noise.** An LLM judge is not a human. It's most reliable for correctness against a reference answer, and less reliable for claim-level faithfulness. Judgments are saved per question in `results/` so they can be spot-checked.
- **Two documents in English only.** Results may not carry over to very different document types, such as scanned forms or code.
