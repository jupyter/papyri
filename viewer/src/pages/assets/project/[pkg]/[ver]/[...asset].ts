// SSR endpoint that serves files from a bundle's `assets/` namespace.
//
// URL shape (built by `linkForAsset` in `lib/links.ts`):
//   /assets/project/<pkg>/<ver>/<filename-with-colons-as-dollars>
//
// Reads through the active BlobStore. We map Content-Type by extension
// since the fs backend doesn't infer it.
//
// SECURITY: these bytes are uploader-supplied. Anyone holding an upload
// token for any project controls them, so this route must never let them
// become active content on the viewer's own origin. Three defences:
//
//   1. An extension allow-list. `papyri gen` only ever writes images here
//      (figure/image directives, doctest plots, the project logo), so
//      anything else is a 404 rather than a guessed Content-Type. This is
//      what keeps `.html` / `.js` off the origin entirely.
//   2. `Content-Security-Policy: default-src 'none'; sandbox`. SVG is an
//      image but also a document that can carry <script>; `sandbox` loads
//      it into an opaque origin on direct navigation, so it cannot read
//      the session cookie or call same-origin APIs, and `default-src
//      'none'` stops it fetching anything at all.
//   3. `X-Content-Type-Options: nosniff` plus an explicit
//      Content-Disposition, so the browser cannot re-interpret a
//      declared image as markup.

import { extname } from "node:path";
import type { APIRoute } from "astro";
import { loadAsset } from "../../../../../lib/ir-reader.ts";
import { getBackends } from "../../../../../lib/backends.ts";
import { assetParams } from "../../../../../lib/static-paths.ts";
import { slugToQualname } from "../../../../../lib/slugs.ts";
import { isSafeSegment } from "../../../../../lib/paths.ts";

/**
 * Extensions this route will serve, and the Content-Type each gets. An
 * allow-list, not a lookup table with a fallback: an asset whose extension
 * is absent is not served at all.
 */
const ALLOWED_MIME: Record<string, string> = {
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".gif": "image/gif",
  ".webp": "image/webp",
  ".avif": "image/avif",
  ".svg": "image/svg+xml",
};

const SECURITY_HEADERS: Record<string, string> = {
  "content-security-policy": "default-src 'none'; sandbox",
  "x-content-type-options": "nosniff",
};

// Every URL this route serves, for the static build. Ignored in server mode,
// where routes are rendered on demand.
export async function getStaticPaths() {
  return (await assetParams(await getBackends())).map((params) => ({ params }));
}
/**
 * Reduce an asset filename to something safe to interpolate into a
 * `Content-Disposition` header. Anything outside the safe set — including
 * the CR/LF and `"` that would let a crafted filename inject headers — is
 * replaced with `_`.
 */
function dispositionName(filename: string): string {
  const base = filename.split("/").pop() ?? "asset";
  const safe = base.replace(/[^A-Za-z0-9._-]/g, "_");
  return safe.length > 0 ? safe : "asset";
}

export const prerender = false;

export const GET: APIRoute = async ({ params }) => {
  const { pkg, ver, asset } = params;
  if (!pkg || !ver || !asset) {
    return new Response("Not found", { status: 404 });
  }
  // Validate before these reach `safeJoin` in the blob store, which throws
  // (500) rather than returning null on a traversal attempt.
  if (!isSafeSegment(pkg) || !isSafeSegment(ver)) {
    return new Response("Bad request", { status: 400 });
  }
  // Reverse the URL-safe slug (`$` -> `:`).
  const filename = slugToQualname(asset);
  const mime = ALLOWED_MIME[extname(filename).toLowerCase()];
  if (!mime) {
    return new Response("Not found", { status: 404 });
  }
  const { blobStore } = await getBackends();
  const bytes = await loadAsset(blobStore, pkg, ver, filename);
  if (!bytes) return new Response("Not found", { status: 404 });
  return new Response(bytes as unknown as BodyInit, {
    headers: {
      ...SECURITY_HEADERS,
      "content-type": mime,
      "content-disposition": `inline; filename="${dispositionName(filename)}"`,
    },
  });
};
