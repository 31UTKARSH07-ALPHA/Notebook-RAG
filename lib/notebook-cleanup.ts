// Deletes notebooks older than NOTEBOOK_TTL_HOURS so a public demo doesn't
// fill the free Qdrant cluster with collections nobody deletes (visitors who
// close the tab never press "New notebook").
//
// Creation time: each new notebook gets `createdAt` (ISO string) merged into
// its Qdrant collection metadata right after the upload creates it
// (stampNotebookCreatedAt). Notebooks without it (made before this existed, or
// whose stamp failed) are stamped with the time cleanup first sees them, so
// they expire one TTL later instead of never, and a missing value can't cause
// an immediate deletion.
//
// Runs opportunistically from the upload route, at most once per hour per
// server instance, in the background: errors are logged, never thrown, and
// the upload doesn't wait for it.

import { deleteCollection, getQdrantClient, isNotebookCollection } from "@/lib/rag";
import { readDemoConfig } from "@/lib/demo-limits";
import { isCleanupCandidate, isExpired, parseCreatedAt, shouldRunCleanup } from "@/lib/notebook-ttl";

const CLEANUP_INTERVAL_MS = 60 * 60 * 1000;
const INSPECT_CONCURRENCY = 5;

let lastRunMs: number | null = null;

// Cleanup is on in demo mode, or whenever a TTL is set explicitly.
function cleanupTtlHours(): number | null {
  const config = readDemoConfig();
  return config.enabled || process.env.NOTEBOOK_TTL_HOURS ? config.notebookTtlHours : null;
}

// Merges createdAt into the collection metadata; Qdrant merges metadata keys
// on update, so the stored embedding config is kept.
export async function stampNotebookCreatedAt(collectionName: string, at: Date = new Date()): Promise<void> {
  await getQdrantClient().updateCollection(collectionName, {
    metadata: { createdAt: at.toISOString() },
  });
}

export type CleanupReport = { inspected: number; deleted: string[]; stamped: string[] };

export async function cleanupExpiredNotebooks(
  ttlHours: number,
  options: { exclude?: string[]; now?: number } = {}
): Promise<CleanupReport> {
  const now = options.now ?? Date.now();
  const exclude = new Set(options.exclude ?? []);
  const client = getQdrantClient();
  const { collections } = await client.getCollections();
  const candidates = collections
    .map((c) => c.name)
    .filter((name) => isCleanupCandidate(name, isNotebookCollection) && !exclude.has(name));

  const report: CleanupReport = { inspected: 0, deleted: [], stamped: [] };

  for (let i = 0; i < candidates.length; i += INSPECT_CONCURRENCY) {
    await Promise.all(
      candidates.slice(i, i + INSPECT_CONCURRENCY).map(async (name) => {
        try {
          const { config } = await client.getCollection(name);
          report.inspected++;
          const createdAt = parseCreatedAt(
            (config.metadata as { createdAt?: unknown } | null | undefined)?.createdAt
          );
          if (createdAt === null) {
            await stampNotebookCreatedAt(name, new Date(now));
            report.stamped.push(name);
          } else if (isExpired(createdAt, now, ttlHours)) {
            // Re-checks the prefix itself, so even a bug above can't drop a foreign collection
            await deleteCollection(name);
            report.deleted.push(name);
          }
        } catch (error) {
          // Collection deleted concurrently, transient network error, ...: try again next run
          console.warn(`Notebook cleanup skipped ${name}:`, error);
        }
      })
    );
  }
  return report;
}

// Fire-and-forget, throttled. `exclude` protects the notebook the current
// request is appending to.
export function maybeCleanupExpiredNotebooks(exclude: string[] = []): void {
  const ttlHours = cleanupTtlHours();
  if (ttlHours === null) return;
  const now = Date.now();
  if (!shouldRunCleanup(lastRunMs, now, CLEANUP_INTERVAL_MS)) return;
  lastRunMs = now;

  cleanupExpiredNotebooks(ttlHours, { exclude, now })
    .then((report) => {
      if (report.deleted.length || report.stamped.length) {
        console.info(
          `Notebook cleanup: deleted ${report.deleted.length}, stamped ${report.stamped.length} ` +
            `of ${report.inspected} notebooks`
        );
      }
    })
    .catch((error) => console.warn("Notebook cleanup failed:", error));
}
