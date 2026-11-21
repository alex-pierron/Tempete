// Export a self-contained HTML research report to PDF (A4, backgrounds kept) with Playwright.
// Usage: node scripts/hackathon/report_to_pdf.mjs <report.html> [report.pdf]
// Playwright comes from the paper2video tools (override with PLAYWRIGHT_NODE_MODULES).
import { existsSync, symlinkSync, unlinkSync } from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";

const MODULES = process.env.PLAYWRIGHT_NODE_MODULES
  ?? "/Users/nowais/.claude/skills/paper2video/.tools/node_modules";
const SETTLE_MS = 500;

const [, , htmlArg, pdfArg] = process.argv;
if (!htmlArg) {
  console.error("usage: node report_to_pdf.mjs <report.html> [report.pdf]");
  process.exit(2);
}
const html = path.resolve(htmlArg);
const pdf = path.resolve(pdfArg ?? html.replace(/\.html?$/i, ".pdf"));
const require = createRequire(path.join(MODULES, "noop.js"));
const { chromium } = require("playwright");

const browser = await chromium.launch();
try {
  const page = await browser.newPage();
  await page.goto(pathToFileURL(html).href + "?mute=1", { waitUntil: "load" });
  await page.waitForTimeout(SETTLE_MS);
  await page.pdf({ path: pdf, format: "A4", printBackground: true,
                   margin: { top: "12mm", bottom: "12mm", left: "10mm", right: "10mm" } });
  console.log(`PDF: ${pdf}`);
} finally {
  await browser.close();
}
