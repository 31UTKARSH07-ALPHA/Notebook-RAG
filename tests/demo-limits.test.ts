import { describe, expect, it } from "vitest";
import { DailyCounter, DemoLimiter, dayKey, getClientIp, readDemoConfig } from "../lib/demo-limits";

const WHAT = { singular: "question", plural: "questions" };

// A controllable clock
function clock(start: string) {
  let now = Date.parse(start);
  return {
    now: () => now,
    set: (iso: string) => {
      now = Date.parse(iso);
    },
  };
}

describe("DemoLimiter", () => {
  it("allows up to the per-visitor limit, then rejects with a friendly message", () => {
    const limiter = new DemoLimiter(2, null, WHAT, clock("2026-10-22T10:00:00Z").now);
    expect(limiter.consume("1.1.1.1").allowed).toBe(true);
    expect(limiter.consume("1.1.1.1").allowed).toBe(true);
    const third = limiter.consume("1.1.1.1");
    expect(third.allowed).toBe(false);
    if (!third.allowed) expect(third.message).toMatch(/2 questions per visitor per day/);
  });

  it("keeps visitors separate", () => {
    const limiter = new DemoLimiter(1, null, WHAT, clock("2026-10-22T10:00:00Z").now);
    expect(limiter.consume("a").allowed).toBe(true);
    expect(limiter.consume("a").allowed).toBe(false);
    expect(limiter.consume("b").allowed).toBe(true);
    expect(limiter.consume("b").allowed).toBe(false);
  });

  it("resets when the UTC day changes", () => {
    const time = clock("2026-10-22T23:59:00Z");
    const limiter = new DemoLimiter(1, null, WHAT, time.now);
    expect(limiter.consume("a").allowed).toBe(true);
    expect(limiter.consume("a").allowed).toBe(false);
    time.set("2026-10-23T00:00:01Z");
    expect(limiter.consume("a").allowed).toBe(true);
  });

  it("enforces the global cap across visitors", () => {
    const limiter = new DemoLimiter(5, 3, WHAT, clock("2026-10-22T10:00:00Z").now);
    expect(limiter.consume("a").allowed).toBe(true);
    expect(limiter.consume("b").allowed).toBe(true);
    expect(limiter.consume("c").allowed).toBe(true);
    const fourth = limiter.consume("d");
    expect(fourth.allowed).toBe(false);
    if (!fourth.allowed) expect(fourth.message).toMatch(/across all visitors/);
  });

  it("does not count rejected requests against the global cap", () => {
    const limiter = new DemoLimiter(1, 2, WHAT, clock("2026-10-22T10:00:00Z").now);
    expect(limiter.consume("a").allowed).toBe(true);
    // Rejected by the per-visitor cap: must not use up the global budget
    expect(limiter.consume("a").allowed).toBe(false);
    expect(limiter.consume("a").allowed).toBe(false);
    expect(limiter.consume("b").allowed).toBe(true);
    expect(limiter.consume("c").allowed).toBe(false);
  });
});

describe("DailyCounter", () => {
  it("counts per key and clears on a new day", () => {
    const time = clock("2026-10-22T12:00:00Z");
    const counter = new DailyCounter(time.now);
    counter.increment("x");
    counter.increment("x");
    expect(counter.count("x")).toBe(2);
    expect(counter.count("y")).toBe(0);
    time.set("2026-10-23T12:00:00Z");
    expect(counter.count("x")).toBe(0);
  });

  it("uses the UTC calendar day", () => {
    expect(dayKey(Date.parse("2026-10-22T23:59:59Z"))).toBe("2026-10-22");
    expect(dayKey(Date.parse("2026-10-23T00:00:00Z"))).toBe("2026-10-23");
  });
});

describe("readDemoConfig", () => {
  it("is off unless DEMO_MODE=true, with sensible defaults", () => {
    expect(readDemoConfig({}).enabled).toBe(false);
    expect(readDemoConfig({ DEMO_MODE: "1" }).enabled).toBe(false);
    expect(readDemoConfig({ DEMO_MODE: "true" })).toEqual({
      enabled: true,
      maxUploadsPerDay: 3,
      maxQuestionsPerDay: 20,
      maxQuestionsTotalPerDay: 300,
      notebookTtlHours: 24,
    });
  });

  it("reads overrides and ignores invalid values", () => {
    const config = readDemoConfig({
      DEMO_MODE: "TRUE",
      DEMO_MAX_UPLOADS_PER_DAY: "5",
      DEMO_MAX_QUESTIONS_PER_DAY: "abc",
      DEMO_MAX_QUESTIONS_TOTAL_PER_DAY: "-1",
      NOTEBOOK_TTL_HOURS: "6",
    });
    expect(config.maxUploadsPerDay).toBe(5);
    expect(config.maxQuestionsPerDay).toBe(20);
    expect(config.maxQuestionsTotalPerDay).toBe(25);
    expect(config.notebookTtlHours).toBe(6);
  });
});

describe("getClientIp", () => {
  it("uses the first x-forwarded-for entry, then x-real-ip", () => {
    expect(getClientIp(new Headers({ "x-forwarded-for": "203.0.113.7, 10.0.0.1" }))).toBe("203.0.113.7");
    expect(getClientIp(new Headers({ "x-real-ip": "198.51.100.2" }))).toBe("198.51.100.2");
    expect(getClientIp(new Headers())).toBe("unknown");
  });
});
