// URL enumeration for the static build.
//
// Every public route's `getStaticPaths` comes from here. Enumeration walks the
// graph and blob store — the same `listModules` / `listDocs` / `listExamples`
// calls the sidebar renders from — rather than crawling links out of rendered
// HTML, which cannot reach pages nothing links to.
//
// These functions are only ever called during a static build; in server mode
// Astro ignores `getStaticPaths` on an on-demand route.

import type { BlobStore, GraphDb } from "papyri-ingest";
import { listBundlesFromDb, listModules, listFiles } from "./ir-reader.ts";
import { listDocs, listExamples } from "./nav.ts";
import { qualnameToSlug } from "./slugs.ts";

export interface Backends {
  blobStore: BlobStore;
  graphDb: GraphDb;
}

/** `{ pkg, ver }` for every ingested bundle. Concrete versions only. */
export async function bundleParams({ graphDb }: Backends): Promise<{ pkg: string; ver: string }[]> {
  const bundles = await listBundlesFromDb(graphDb);
  return bundles.map((b) => ({ pkg: b.pkg, ver: b.version }));
}

/** `{ pkg }` for every ingested package. */
export async function packageParams({ graphDb }: Backends): Promise<{ pkg: string }[]> {
  const seen = new Set<string>();
  for (const b of await listBundlesFromDb(graphDb)) seen.add(b.pkg);
  return [...seen].sort().map((pkg) => ({ pkg }));
}

/**
 * Run `fn` for every bundle and flatten, tagging each result with its bundle.
 * Bundles are walked sequentially: a whole-store snapshot can hold many
 * bundles, and the blob store is local, so concurrency buys little and makes
 * the build's memory ceiling unpredictable.
 */
async function forEachBundle<T extends Record<string, string>>(
  backends: Backends,
  fn: (b: Backends, pkg: string, ver: string) => Promise<T[]>
): Promise<({ pkg: string; ver: string } & T)[]> {
  const out: ({ pkg: string; ver: string } & T)[] = [];
  for (const { pkg, ver } of await bundleParams(backends)) {
    for (const rest of await fn(backends, pkg, ver)) {
      out.push({ pkg, ver, ...rest });
    }
  }
  return out;
}

/** Qualname pages: `/project/<pkg>/<ver>/<slug>/`. */
export function qualnameParams(
  backends: Backends
): Promise<{ pkg: string; ver: string; slug: string }[]> {
  return forEachBundle(backends, async ({ blobStore }, pkg, ver) =>
    (await listModules(blobStore, pkg, ver)).map((qa) => ({ slug: qualnameToSlug(qa) }))
  );
}

/**
 * Narrative doc pages: `/project/<pkg>/<ver>/docs/<a>/<b>/`.
 * Doc keys are colon-separated (`reference:ufuncs`); the URL uses '/', so this
 * is the inverse of the page's `docParam.split("/").join(":")`.
 */
export function docParams(
  backends: Backends
): Promise<{ pkg: string; ver: string; doc: string }[]> {
  return forEachBundle(backends, async ({ blobStore }, pkg, ver) =>
    (await listDocs(blobStore, pkg, ver)).map((key) => ({ doc: key.split(":").join("/") }))
  );
}

/** Example pages: `/project/<pkg>/<ver>/examples/<path>/`. */
export function exampleParams(
  backends: Backends
): Promise<{ pkg: string; ver: string; ex: string }[]> {
  return forEachBundle(backends, async ({ blobStore }, pkg, ver) =>
    (await listExamples(blobStore, pkg, ver)).map((ex) => ({ ex }))
  );
}

/**
 * Bundle assets: `/assets/project/<pkg>/<ver>/<file>`. Asset filenames carry
 * qualnames (`fig-papyri.examples:example1-0.png`), so they get the same
 * ':' -> '$' slug as qualnames — matching `linkForAsset`.
 */
export function assetParams(
  backends: Backends
): Promise<{ pkg: string; ver: string; asset: string }[]> {
  return forEachBundle(backends, async ({ blobStore }, pkg, ver) =>
    (await listFiles(blobStore, pkg, ver, "assets")).map((f) => ({ asset: qualnameToSlug(f) }))
  );
}
