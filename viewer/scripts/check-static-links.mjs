#!/usr/bin/env node
/**
 * Verify a static snapshot is self-consistent: every internal link in every
 * emitted page resolves to a file that was actually emitted.
 *
 * This is the guard against the one way the static and server modes can
 * drift. Both render from the same templates, so their *content* cannot
 * diverge — but the URLs a template emits and the URLs `lib/static-paths.ts`
 * enumerates are separate pieces of code, and a link to a page nothing
 * enumerated is a 404 that only exists in the snapshot.
 *
 * The snapshot is currently root-hosted: `linkFor*` emits root-absolute
 * paths, so serving it under a sub-path needs base support in `lib/links.ts`
 * first.
 *
 * Usage: node scripts/check-static-links.mjs [dist/client]
 */
import { readdir, readFile, stat } from "node:fs/promises";
import { join, posix } from "node:path";

const root = process.argv[2] ?? "dist/client";

async function walk(dir, out = []) {
  for (const e of await readdir(dir, { withFileTypes: true })) {
    const p = join(dir, e.name);
    if (e.isDirectory()) await walk(p, out);
    else out.push(p);
  }
  return out;
}

/** Does `urlPath` (root-absolute) exist in the output? */
async function exists(urlPath) {
  const rel = decodeURIComponent(urlPath.replace(/^\//, ""));
  for (const candidate of [rel, posix.join(rel, "index.html")]) {
    try {
      if ((await stat(join(root, candidate))).isFile()) return true;
    } catch {
      /* try the next candidate */
    }
  }
  return false;
}

const files = (await walk(root)).filter((f) => f.endsWith(".html"));
if (files.length === 0) {
  console.error(`no HTML found under ${root} — did the static build run?`);
  process.exit(1);
}

const broken = new Map(); // url -> pages referencing it
let checked = 0;

for (const file of files) {
  const html = await readFile(file, "utf8");
  for (const m of html.matchAll(/(?:href|src)="([^"]+)"/g)) {
    const url = m[1];
    // Skip off-site, in-page, and non-navigational URLs.
    if (!url.startsWith("/") || url.startsWith("//")) continue;
    const path = url.split("#")[0].split("?")[0];
    if (!path || path === "/") continue;
    checked++;
    if (!(await exists(path))) {
      broken.set(path, (broken.get(path) ?? []).concat(file));
    }
  }
}

console.log(`checked ${checked} internal links across ${files.length} pages`);
if (broken.size > 0) {
  console.error(`\n${broken.size} link target(s) missing from the snapshot:`);
  for (const [url, pages] of [...broken].sort()) {
    console.error(`  ${url}\n    referenced by ${pages.length} page(s), e.g. ${pages[0]}`);
  }
  process.exit(1);
}
console.log("all internal links resolve");
