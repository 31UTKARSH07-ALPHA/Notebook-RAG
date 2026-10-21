import { defineConfig } from "vitest/config";

// Only the unit tests in tests/. Without this, vitest would also pick up test
// files inside nested git worktrees (.claude/worktrees) or other checkouts.
export default defineConfig({
  test: {
    include: ["tests/**/*.test.ts"],
  },
});
