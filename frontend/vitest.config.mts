import { fileURLToPath } from "node:url";

import { defineConfig } from "vitest/config";

const here = (p: string) => fileURLToPath(new URL(p, import.meta.url));

export default defineConfig({
  resolve: {
    alias: {
      "@": here("./src"),
      // Outside Next's react-server bundle the real package throws on import by design.
      "server-only": here("./test/support/server-only.ts"),
    },
  },
  test: {
    include: ["test/**/*.test.ts"],
    environment: "node",
    // Each file gets its own process: the runner keeps a per-process queue and caches its config.
    pool: "forks",
    isolate: true,
    testTimeout: 20_000,
  },
});
