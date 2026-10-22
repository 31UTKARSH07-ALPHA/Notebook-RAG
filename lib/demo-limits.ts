// Usage limits for a public demo deployment.
//
// A public demo spends the owner's free quotas (Groq tokens per day, Cohere
// trial calls per month), so with DEMO_MODE=true each visitor gets a few
// uploads and questions per day, and all visitors together share a daily
// question cap. Without DEMO_MODE nothing is limited, so local development and
// the evaluation are unaffected.
//
// LIMITATION (honest): the counters live in this process's memory. On
// serverless hosting (Vercel) every warm instance has its own counters and they
// reset on a cold start or redeploy, so the limits are best-effort: they stop
// casual overuse and runaway loops, not a determined abuser. Strict limits need
// a shared durable store (e.g. Upstash Redis with a fixed-window counter keyed
// the same way as below).

export type LimitDecision = { allowed: true } | { allowed: false; message: string };

export type DemoConfig = {
  enabled: boolean;
  maxUploadsPerDay: number;
  maxQuestionsPerDay: number;
  maxQuestionsTotalPerDay: number;
  notebookTtlHours: number;
};

const DEFAULTS = {
  maxUploadsPerDay: 3,
  maxQuestionsPerDay: 20,
  // One question costs ~7K Groq tokens and the free tier allows 200K per day
  // per model, so about 25 questions a day is the real ceiling for everyone.
  maxQuestionsTotalPerDay: 25,
  notebookTtlHours: 24,
};

// Bounds memory if many distinct IPs show up within one day
const MAX_TRACKED_KEYS = 10_000;

function positiveNumber(value: string | undefined, fallback: number): number {
  const parsed = Number(value);
  return value !== undefined && value.trim() !== "" && Number.isFinite(parsed) && parsed > 0
    ? parsed
    : fallback;
}

export function readDemoConfig(env: Record<string, string | undefined> = process.env): DemoConfig {
  return {
    enabled: env.DEMO_MODE?.trim().toLowerCase() === "true",
    maxUploadsPerDay: Math.floor(positiveNumber(env.DEMO_MAX_UPLOADS_PER_DAY, DEFAULTS.maxUploadsPerDay)),
    maxQuestionsPerDay: Math.floor(positiveNumber(env.DEMO_MAX_QUESTIONS_PER_DAY, DEFAULTS.maxQuestionsPerDay)),
    maxQuestionsTotalPerDay: Math.floor(
      positiveNumber(env.DEMO_MAX_QUESTIONS_TOTAL_PER_DAY, DEFAULTS.maxQuestionsTotalPerDay)
    ),
    notebookTtlHours: positiveNumber(env.NOTEBOOK_TTL_HOURS, DEFAULTS.notebookTtlHours),
  };
}

// UTC calendar day, so every counter resets at 00:00 UTC (Groq's daily
// token budget is also a per-day allowance).
export function dayKey(now: number): string {
  return new Date(now).toISOString().slice(0, 10);
}

// Fixed-window counter per key, reset when the UTC day changes.
export class DailyCounter {
  private day = "";
  private counts = new Map<string, number>();

  constructor(private readonly now: () => number = Date.now) {}

  private roll(): void {
    const today = dayKey(this.now());
    if (today !== this.day) {
      this.day = today;
      this.counts.clear();
    }
  }

  count(key: string): number {
    this.roll();
    return this.counts.get(key) ?? 0;
  }

  increment(key: string): void {
    this.roll();
    if (!this.counts.has(key) && this.counts.size >= MAX_TRACKED_KEYS) {
      // Drop the oldest key rather than grow without bound
      const oldest = this.counts.keys().next().value;
      if (oldest !== undefined) this.counts.delete(oldest);
    }
    this.counts.set(key, (this.counts.get(key) ?? 0) + 1);
  }
}

const GLOBAL_KEY = "*";

// Per-visitor daily cap, plus an optional cap across all visitors. A request
// is only counted when it is allowed, so rejected requests don't extend a ban.
export class DemoLimiter {
  private readonly counter: DailyCounter;

  constructor(
    private readonly perKeyLimit: number,
    private readonly globalLimit: number | null,
    private readonly what: { singular: string; plural: string },
    now: () => number = Date.now
  ) {
    this.counter = new DailyCounter(now);
  }

  consume(key: string): LimitDecision {
    if (this.counter.count(key) >= this.perKeyLimit) {
      return {
        allowed: false,
        message:
          `This public demo allows ${this.perKeyLimit} ${this.perKeyLimit === 1 ? this.what.singular : this.what.plural} ` +
          `per visitor per day, and you've used them all. The limit resets at 00:00 UTC, ` +
          `or run SourceSage yourself with your own API keys (see the README).`,
      };
    }
    if (this.globalLimit !== null && this.counter.count(GLOBAL_KEY) >= this.globalLimit) {
      return {
        allowed: false,
        message:
          `The demo has reached its daily ${this.what.singular} budget across all visitors ` +
          `(it runs on free API tiers). Please try again after 00:00 UTC.`,
      };
    }
    this.counter.increment(key);
    if (this.globalLimit !== null) this.counter.increment(GLOBAL_KEY);
    return { allowed: true };
  }
}

// Client IP as seen behind Vercel's proxy. Vercel sets x-forwarded-for itself
// (first entry = the client), so it can't be spoofed there; behind other
// proxies it can be, which is one more reason these limits are best-effort.
export function getClientIp(headers: Headers): string {
  const forwarded = headers.get("x-forwarded-for")?.split(",")[0]?.trim();
  if (forwarded) return forwarded;
  const realIp = headers.get("x-real-ip")?.trim();
  return realIp || "unknown";
}

// Process-wide limiters, created lazily so env changes in tests are picked up
let uploadLimiter: DemoLimiter | null = null;
let questionLimiter: DemoLimiter | null = null;

export function checkUploadLimit(request: Request): LimitDecision {
  const config = readDemoConfig();
  if (!config.enabled) return { allowed: true };
  uploadLimiter ??= new DemoLimiter(config.maxUploadsPerDay, null, {
    singular: "upload",
    plural: "uploads",
  });
  return uploadLimiter.consume(getClientIp(request.headers));
}

export function checkQuestionLimit(request: Request): LimitDecision {
  const config = readDemoConfig();
  if (!config.enabled) return { allowed: true };
  questionLimiter ??= new DemoLimiter(config.maxQuestionsPerDay, config.maxQuestionsTotalPerDay, {
    singular: "question",
    plural: "questions",
  });
  return questionLimiter.consume(getClientIp(request.headers));
}
