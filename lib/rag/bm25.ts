// BM25 keyword search as Qdrant sparse vectors: tokenizing, hashing terms to
// indices, and building document and query vectors.

// BM25 parameters (standard defaults)
const BM25_K1 = 1.2;
const BM25_B = 0.75;

export type SparseVector = {
  indices: number[];
  values: number[];
};

// BM25 sparse vectors
//
// BM25 = IDF(term) * TF-saturation(term, doc). The TF part only needs the
// chunk and the average chunk length, so it is computed here at index time.
// The IDF part needs corpus statistics, so the collection is created with
// modifier "idf" and Qdrant applies IDF at query time from its own counts.

export function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((t) => t.length > 1);
}

export function hashToken(term: string): number {
  let hash = 0;
  for (let i = 0; i < term.length; i++) {
    hash = (hash * 31 + term.charCodeAt(i)) & 0xfffff;
  }
  return hash;
}

// Counts per hashed index; summing by index means two words that hash to the
// same slot never produce duplicate indices (which Qdrant rejects).
function hashedTermCounts(tokens: string[]): Map<number, number> {
  const counts = new Map<number, number>();
  for (const token of tokens) {
    const index = hashToken(token);
    counts.set(index, (counts.get(index) || 0) + 1);
  }
  return counts;
}

export function buildDocumentSparseVector(text: string, avgDocLength: number): SparseVector {
  const tokens = tokenize(text);
  const lengthNorm = 1 - BM25_B + BM25_B * (tokens.length / avgDocLength);
  const indices: number[] = [];
  const values: number[] = [];

  for (const [index, tf] of hashedTermCounts(tokens)) {
    indices.push(index);
    values.push((tf * (BM25_K1 + 1)) / (tf + BM25_K1 * lengthNorm));
  }

  return { indices, values };
}

export function buildQuerySparseVector(text: string): SparseVector {
  const indices = Array.from(hashedTermCounts(tokenize(text)).keys());
  return { indices, values: indices.map(() => 1) };
}
