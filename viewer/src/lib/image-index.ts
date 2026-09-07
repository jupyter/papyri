// Bundle-wide image index: queries the node_index table for Figure/Image nodes
// and groups by source URL. Pulled out of `pages/[pkg]/[ver]/images/index.astro`
// so the page becomes a flat template over the returned entries.

import type { BlobStore, GraphDb } from "papyri-ingest";
import { collectImages, type FoundImgNode } from "./ir-reader.ts";
import { linkForAsset } from "./links.ts";
import { walkBundle, type PageRef } from "./bundle-walk.ts";

export type { PageRef };

export interface ImgEntry {
  img: FoundImgNode;
  pages: PageRef[];
}

function addHit(map: Map<string, ImgEntry>, found: FoundImgNode, page: PageRef): void {
  const existing = map.get(found.src);
  if (existing) {
    if (!existing.pages.some((p) => p.href === page.href)) {
      existing.pages.push(page);
    }
  } else {
    map.set(found.src, { img: found, pages: [page] });
  }
}

/**
 * Query the node_index table for all Figure and Image nodes, build image entries,
 * and return them sorted by source URL. Falls back to walkBundle scan if the
 * index is empty (for bundles ingested before the node_index migration).
 *
 * `ver` is the actual stored version; `urlVer` (optional, defaults to `ver`)
 * is used for page hrefs — pass "latest" when serving the canonical latest URL.
 */
export async function collectBundleImages(
  graphDb: GraphDb,
  blobStore: BlobStore,
  pkg: string,
  ver: string,
  urlVer?: string
): Promise<ImgEntry[]> {
  const map = new Map<string, ImgEntry>();
  const uv = urlVer ?? ver;
  const t0 = performance.now();
  console.log(`[images] scan start pkg=${pkg} ver=${ver}`);

  // Try to query the node_index table for Figure and Image nodes.
  const figureRows = await graphDb.queryNodeIndex(pkg, ver, "Figure");
  const imageRows = await graphDb.queryNodeIndex(pkg, ver, "Image");

  if (figureRows.length > 0 || imageRows.length > 0) {
    // Index is available; use it instead of scanning blobs.
    for (const row of figureRows) {
      const node = JSON.parse(row.content) as Record<string, unknown>;
      const ref = node.value as
        | { module?: string; version?: string; kind?: string; path?: string }
        | undefined;
      if (ref?.kind === "assets" && ref.module && ref.version && ref.path) {
        const found: FoundImgNode = {
          kind: "Figure",
          src: linkForAsset(ref.module, ref.version, String(ref.path)),
          assetPath: String(ref.path),
        };
        const page: PageRef = {
          label: row.page_qa,
          href: row.page_href.replace(new RegExp(`/${ver}/`), `/${uv}/`),
        };
        addHit(map, found, page);
      }
    }

    for (const row of imageRows) {
      const node = JSON.parse(row.content) as Record<string, unknown>;
      const url = String((node as Record<string, unknown>).url ?? "");
      if (url) {
        const found: FoundImgNode = {
          kind: "Image",
          src: url,
          alt: String((node as Record<string, unknown>).alt ?? ""),
        };
        const page: PageRef = {
          label: row.page_qa,
          href: row.page_href.replace(new RegExp(`/${ver}/`), `/${uv}/`),
        };
        addHit(map, found, page);
      }
    }

    console.log(
      `[images] scan done pkg=${pkg} ver=${ver} unique=${map.size} ` +
        `${(performance.now() - t0).toFixed(1)}ms (indexed)`
    );
    return [...map.values()].sort((a, b) => a.img.src.localeCompare(b.img.src));
  }

  // Fallback: walk blobs for bundles without node_index (pre-migration).
  await walkBundle(
    blobStore,
    pkg,
    ver,
    (doc, page) => {
      for (const found of collectImages(doc)) addHit(map, found, page);
      return true;
    },
    uv
  );

  console.log(
    `[images] scan done pkg=${pkg} ver=${ver} unique=${map.size} ` +
      `${(performance.now() - t0).toFixed(1)}ms (scanned)`
  );
  return [...map.values()].sort((a, b) => a.img.src.localeCompare(b.img.src));
}
