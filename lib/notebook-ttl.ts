// Pure rules for expiring notebooks (no I/O, unit-tested in tests/notebook-ttl.test.ts).
// The Qdrant side lives in lib/notebook-cleanup.ts.

const HOUR_MS = 60 * 60 * 1000;

// Evaluation collections are named `<prefix>_eval_<doc>_<variant>` (eval/lib.ts).
// They share the notebook prefix but are built by `npm run eval` and reused
// across runs, so cleanup must never delete them.
export function isEvalCollection(collectionName: string): boolean {
  return collectionName.includes("_eval_");
}

// Only app notebooks are candidates: the caller passes isNotebookCollection()
// (prefix check) and evaluation collections are excluded on top of it.
export function isCleanupCandidate(
  collectionName: string,
  isNotebookCollection: (name: string) => boolean,
): boolean {
  return (
    isNotebookCollection(collectionName) && !isEvalCollection(collectionName)
  );
}

// Parses the createdAt stored in collection metadata. Anything unparseable
// counts as missing, so a bad value can never make a notebook look ancient.
export function parseCreatedAt(value: unknown): number | null {
  if (typeof value !== "string" && typeof value !== "number") return null;
  const time = typeof value === "number" ? value : Date.parse(value);
  return Number.isFinite(time) && time > 0 ? time : null;
}

// Expired only with a known creation time older than the TTL; a creation time
// in the future (clock skew) is never expired.
export function isExpired(
  createdAtMs: number | null,
  nowMs: number,
  ttlHours: number,
): boolean {
  if (createdAtMs === null || !(ttlHours > 0)) return false;
  return nowMs - createdAtMs > ttlHours * HOUR_MS;
}

// At most one cleanup per interval per process.
export function shouldRunCleanup(
  lastRunMs: number | null,
  nowMs: number,
  intervalMs: number,
): boolean {
  return lastRunMs === null || nowMs - lastRunMs >= intervalMs;
}
