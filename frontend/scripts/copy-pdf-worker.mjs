// Serve pdf.js's worker from this origin (no CDN): copied from the installed, locked package.
import { copyFileSync, mkdirSync } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const source = require.resolve("pdfjs-dist/build/pdf.worker.min.mjs");
const target = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "public", "pdf.worker.min.mjs");
mkdirSync(path.dirname(target), { recursive: true });
copyFileSync(source, target);
