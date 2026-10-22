import { describe, expect, it } from "vitest";
import {
  isCleanupCandidate,
  isEvalCollection,
  isExpired,
  parseCreatedAt,
  shouldRunCleanup,
} from "../lib/notebook-ttl";
import { isUploadTooLarge, MAX_UPLOAD_BYTES } from "../lib/upload-limits";

const HOUR = 60 * 60 * 1000;
const isNotebook = (name: string) => name.startsWith("sourcesage_");

describe("eval collection guard", () => {
  it("never selects evaluation collections, even though they share the prefix", () => {
    const evalName = "sourcesage_eval_nist_csf_fixed";
    expect(isNotebook(evalName)).toBe(true);
    expect(isEvalCollection(evalName)).toBe(true);
    expect(isCleanupCandidate(evalName, isNotebook)).toBe(false);
  });

  it("selects app notebooks and nothing without the prefix", () => {
    expect(isCleanupCandidate("sourcesage_0f8e2c1a_4b7d_4e2a_9c1f_123456789abc", isNotebook)).toBe(true);
    expect(isCleanupCandidate("someone_elses_collection", isNotebook)).toBe(false);
    expect(isCleanupCandidate("other_eval_thing", isNotebook)).toBe(false);
  });
});

describe("isExpired", () => {
  const now = Date.parse("2026-10-22T12:00:00Z");

  it("expires notebooks older than the TTL only", () => {
    expect(isExpired(now - 25 * HOUR, now, 24)).toBe(true);
    expect(isExpired(now - 23 * HOUR, now, 24)).toBe(false);
    expect(isExpired(now - 24 * HOUR, now, 24)).toBe(false);
  });

  it("never expires without a creation time, with a future one, or with a bad TTL", () => {
    expect(isExpired(null, now, 24)).toBe(false);
    expect(isExpired(now + HOUR, now, 24)).toBe(false);
    expect(isExpired(now - 100 * HOUR, now, 0)).toBe(false);
    expect(isExpired(now - 100 * HOUR, now, Number.NaN)).toBe(false);
  });
});

describe("parseCreatedAt", () => {
  it("accepts ISO strings and epoch millis, rejects everything else", () => {
    expect(parseCreatedAt("2026-10-22T12:00:00.000Z")).toBe(Date.parse("2026-10-22T12:00:00Z"));
    expect(parseCreatedAt(1_700_000_000_000)).toBe(1_700_000_000_000);
    expect(parseCreatedAt(undefined)).toBeNull();
    expect(parseCreatedAt(null)).toBeNull();
    expect(parseCreatedAt("not a date")).toBeNull();
    expect(parseCreatedAt({})).toBeNull();
  });
});

describe("shouldRunCleanup", () => {
  it("runs first time, then at most once per interval", () => {
    expect(shouldRunCleanup(null, 1000, HOUR)).toBe(true);
    expect(shouldRunCleanup(1000, 1000 + HOUR - 1, HOUR)).toBe(false);
    expect(shouldRunCleanup(1000, 1000 + HOUR, HOUR)).toBe(true);
  });
});

describe("upload size limit", () => {
  it("allows up to 4 MB", () => {
    expect(isUploadTooLarge(MAX_UPLOAD_BYTES)).toBe(false);
    expect(isUploadTooLarge(MAX_UPLOAD_BYTES + 1)).toBe(true);
  });
});
