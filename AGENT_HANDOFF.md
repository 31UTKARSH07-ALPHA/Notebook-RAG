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

## Session Log

### Session 1 (2026-06-22) - Resumption Point
**Current Phase**: Phase 2 — Query Translation + Re-ranking

**Status**: Phase 1 is complete. The working directory on branch `phase2` has the following Phase 2 functions implemented:
- generateQueryVariants() - LLM rewrites user question into 3 different phrasings
- rerank() - CrossEncoder re-ranking (with local model fallback)
- retrieveChunks() - multi-query hybrid retrieval with RRF fusion

**What's next**: Verify Phase 2 implementation works end-to-end and test with real queries.

**Build issues**: There's a version incompatibility error with @langchain/groq and @langchain/core. This may need dependency version fixes.

**Files to check**:
- lib/rag.ts (has Phase 2 implementation)
- app/api/ask/route.ts (uses retrieveChunks)
- app/api/upload/route.ts (ingestion pipeline)
- .env.local (environment variables configuration)

**Next function to build** (if Phase 2 is complete): Phase 3 functions - gradeChunks(), compressContext(), and hallucination grader per ARCHITECTURE.md section on Phase 3.
