// Export de la présentation en PDF (une diapositive par page, 1600 x 900) avec Playwright.
//
// Usage :
//   node docs/hackathon/presentation/build_pdf.mjs                   # écrit presentation.pdf
//   node docs/hackathon/presentation/build_pdf.mjs --png /tmp/slides # écrit aussi une capture PNG par diapositive
//
// Playwright n'est pas une dépendance du dépôt : on le prend dans l'installation de l'outil paper2video
// (variable PLAYWRIGHT_NODE_MODULES pour un autre emplacement). Un lien symbolique `node_modules` est créé
// à côté de ce script le temps de l'export, puis supprimé : il ne doit jamais être commité.
//
// Le script ne touche pas au réseau : la présentation est un fichier unique, ouvert en file://.

import { existsSync, lstatSync, mkdirSync, readFileSync, symlinkSync, unlinkSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_PLAYWRIGHT_MODULES = "/Users/nowais/.claude/skills/paper2video/.tools/node_modules";
const PLAYWRIGHT_MODULES = process.env.PLAYWRIGHT_NODE_MODULES ?? DEFAULT_PLAYWRIGHT_MODULES;
const HTML_PATH = path.join(HERE, "index.html");
const PDF_PATH = path.join(HERE, "presentation.pdf");
const MODULES_LINK = path.join(HERE, "node_modules");

const SLIDE_WIDTH_PX = 1600;
const SLIDE_HEIGHT_PX = 900;
const SETTLE_MS = 120;
const OVERFLOW_TOLERANCE_PX = 1;
// "mute=1" est exigé par la règle d'équipe pour tout navigateur headless (voix de synthèse des pages).
const QUERY = "?mute=1";

function parseArguments(argv) {
  const pngIndex = argv.indexOf("--png");
  if (pngIndex === -1) return { pngDir: null };
  const pngDir = argv[pngIndex + 1];
  if (!pngDir) throw new Error("--png demande un dossier de sortie");
  return { pngDir };
}

function linkPlaywright() {
  if (existsSync(MODULES_LINK) || isDanglingLink(MODULES_LINK)) return false;
  if (!existsSync(PLAYWRIGHT_MODULES)) {
    throw new Error(`Playwright introuvable : ${PLAYWRIGHT_MODULES} (voir PLAYWRIGHT_NODE_MODULES)`);
  }
  symlinkSync(PLAYWRIGHT_MODULES, MODULES_LINK, "dir");
  return true;
}

function isDanglingLink(target) {
  try {
    return lstatSync(target).isSymbolicLink();
  } catch {
    return false;
  }
}

// Une diapositive déborde quand son contenu dépasse la zone prévue (le CSS coupe sans prévenir).
async function findOverflowingSlides(page) {
  return page.evaluate((tolerance) => {
    const problems = [];
    document.querySelectorAll("section.slide").forEach((slide, index) => {
      const content = slide.querySelector(".content");
      if (!content) return;
      const hidden = content.scrollHeight - content.clientHeight;
      const wide = content.scrollWidth - content.clientWidth;
      if (hidden > tolerance || wide > tolerance) {
        problems.push({ slide: index + 1, hiddenHeightPx: hidden, hiddenWidthPx: wide });
      }
    });
    return problems;
  }, OVERFLOW_TOLERANCE_PX);
}

async function capturePng(page, directory, slideCount) {
  mkdirSync(directory, { recursive: true });
  for (let number = 1; number <= slideCount; number += 1) {
    await page.evaluate((n) => { window.location.hash = `#${n}`; }, number);
    await page.waitForTimeout(SETTLE_MS);
    const file = path.join(directory, `slide-${String(number).padStart(2, "0")}.png`);
    await page.screenshot({ path: file });
  }
}

function countPdfPages(buffer) {
  const matches = buffer.toString("latin1").match(/\/Type\s*\/Page(?![s\w])/g);
  return matches ? matches.length : 0;
}

async function main() {
  const { pngDir } = parseArguments(process.argv.slice(2));
  if (!existsSync(HTML_PATH)) throw new Error(`Présentation introuvable : ${HTML_PATH}`);
  const created = linkPlaywright();
  try {
    const { chromium } = await import(pathToFileURL(path.join(MODULES_LINK, "playwright", "index.mjs")).href);
    const browser = await chromium.launch({ headless: true });
    try {
      const page = await browser.newPage({ viewport: { width: SLIDE_WIDTH_PX, height: SLIDE_HEIGHT_PX } });
      await page.goto(`${pathToFileURL(HTML_PATH).href}${QUERY}#1`, { waitUntil: "load" });
      const slideCount = await page.evaluate(() => document.querySelectorAll("section.slide").length);

      const overflowing = await findOverflowingSlides(page);
      if (overflowing.length > 0) {
        console.error("Diapositives qui débordent :", JSON.stringify(overflowing));
      }
      if (pngDir) {
        await capturePng(page, pngDir, slideCount);
        console.log(`${slideCount} captures écrites dans ${pngDir}`);
      }

      await page.emulateMedia({ media: "print" });
      const pdf = await page.pdf({
        width: `${SLIDE_WIDTH_PX}px`,
        height: `${SLIDE_HEIGHT_PX}px`,
        printBackground: true,
        preferCSSPageSize: true,
      });
      writeFileSync(PDF_PATH, pdf);
      const pages = countPdfPages(readFileSync(PDF_PATH));
      console.log(`${PDF_PATH} : ${pages} pages pour ${slideCount} diapositives`);
      if (pages !== slideCount) process.exitCode = 1;
      if (overflowing.length > 0) process.exitCode = 1;
    } finally {
      await browser.close();
    }
  } finally {
    if (created) unlinkSync(MODULES_LINK);
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
