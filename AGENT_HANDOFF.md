## Phase 1 — Ingestion Pipeline (DONE)
Goal: Index documents into Qdrant with better chunking and hybrid vectors.
 
### What was built
- PDF parsing with unpdf (replaced pdf2json)
- Sentence-window chunking: 2-sentence child chunks indexed in Qdrant
- Parent-child pattern: child retrieved, parent (5-sentence window) sent to LLM
- Gemini text-embedding-004 for dense vectors (768 dims)
- BM25-style sparse vectors built alongside dense in same Qdrant collection
- Raw QdrantClient used (not LangChain wrapper) to support named vectors
- Upsert in batches of 100

### Functions in rag.ts
- parseUploadedFile()     — PDF + txt loading
- splitIntoSentences()    — regex sentence boundary detection
- chunkDocuments()        — sentence-window + parent-child chunking
- buildSparseVector()     — TF-normalized BM25 approximation
- indexChunks()           — creates Qdrant collection, embeds, upserts

### .env required
- GOOGLE_API_KEY
- GROQ_API_KEY
- QDRANT_URL
- QDRANT_API_KEY (optional, Qdrant Cloud only)

---

## Phase 2 — Query Translation + Re-ranking (DONE)
Goal: Smarter retrieval — transform the query before hitting Qdrant, re-rank after.

### What was built
- Multi-query generation: LLM rewrites user question into 3 different phrasings
- Hybrid retrieval (dense + sparse + RRF) for each phrasing
- CrossEncoder re-ranking with local model fallback
- RAG Fusion wrapper function

### Functions in rag.ts
- generateQueryVariants() — LLM rewrites question into 3 variants
- rerank() — CrossEncoder re-ranking (with local model fallback)
- retrieveChunks() — full pipeline: variants → hybrid retrieval → RRF → rerank
- ragFusion() — public API wrapper for retrieveChunks

---

## Phase 3 — Post-retrieval (CRAG + Context Compression) (DONE)
Goal: Filter bad chunks, compress context, reduce hallucinations.

### What was built
- Relevance grader: LLM scores each retrieved chunk
- Fallback strategy: returns "not found in document" if no chunks pass threshold
- Context compression: LLM extracts only relevant sentences from each chunk
- Hallucination grader: LLM checks if answer is grounded in chunks

### Functions in rag.ts
- gradeChunks() — relevance scoring with JSON output
- compressContext() — extract relevant sentences from chunks
- gradeAnswer() — hallucination detection with grounded check
- retrieveChunks() — now includes gradeChunks → fallback → compressContext pipeline

---

## Session Log

### Session 2 (2026-06-22) - Resumption Point
**Current Phase**: Phase 3 — Post-retrieval (CRAG + Context Compression)

**Status**: Phase 1 is complete. Phase 2 and Phase 3 are fully implemented on branch `phase2`.

**Phase 2 Complete**:
- generateQueryVariants() - LLM rewrites user question into 3 different phrasings
- rerank() - CrossEncoder re-ranking (with local model fallback)
- retrieveChunks() - multi-query hybrid retrieval with RRF fusion
- ragFusion() - public API wrapper

**Phase 3 Complete**:
- gradeChunks() — relevance grading with JSON output, filters irrelevant chunks
- compressContext() — extracts relevant sentences from chunks
- gradeAnswer() — hallucination detection with grounded check
- retrieveChunks() — includes gradeChunks → fallback → compressContext pipeline

**What's next**: Verify the complete pipeline works end-to-end and test with real queries.

**Build issues**: There's a version incompatibility error with @langchain/groq and @langchain/core. This may need dependency version fixes.

**Files to check**:
- lib/rag.ts (has Phase 2 and 3 implementation)
- app/api/ask/route.ts (uses retrieveChunks)
- app/api/upload/route.ts (ingestion pipeline)
- .env.local (environment variables configuration)

**Next function to build** (if Phase 3 is complete): Phase 4 functions - eval dataset, RAGAS metrics, LLM-as-judge scoring, eval API endpoint, eval dashboard per ARCHITECTURE.md section on Phase 4.

---

## Resumption Point

**Where to continue**: The complete Phase 3 implementation is ready. The next step is to:
1. Fix dependency issues with @langchain/groq and @langchain/core
2. Test the complete pipeline with real queries
3. If successful, proceed to Phase 4 (Evaluation) which includes:
   - eval-dataset.json test dataset (20-30 question + answer pairs)
   - RAGAS metrics implementation
   - judgeAnswer() function
   - /api/evaluate endpoint
   - eval dashboard page
