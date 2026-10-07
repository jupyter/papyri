// SSR endpoint: delete a single bundle (one pkg@ver) from the store.
//
// Removes the bundle's graph rows and processed blobs; with `raw=1` also the
// raw archive entry (otherwise a later reingest brings it back).
//
// Method: POST ?pkg=<pkg>&ver=<ver>[&raw=1]
// Auth:   admin action — covered by the "/api/delete-bundle" entry in
//         middleware.ts's ADMIN_ONLY_PREFIXES.
// Response: JSON { ok, deletedBlobs, deletedNodes, rawDeleted, elapsed_s }.

import type { APIRoute } from "astro";
import { deleteBundle } from "papyri-ingest";
import { getBackends } from "../../lib/backends.ts";
import { respond } from "../../lib/api-utils.ts";

export const prerender = false;

export const POST: APIRoute = async ({ url }) => {
  const startedAt = Date.now();
  const pkg = url.searchParams.get("pkg");
  const ver = url.searchParams.get("ver");
  if (!pkg || !ver) return respond({ ok: false, error: "pkg and ver are required" }, 400);
  try {
    const result = await deleteBundle(await getBackends(), pkg, ver, {
      raw: url.searchParams.get("raw") === "1",
    });
    return respond({
      ok: true,
      ...result,
      elapsed_s: ((Date.now() - startedAt) / 1000).toFixed(2),
    });
  } catch (err) {
    return respond({ ok: false, error: String(err) }, 500);
  }
};
