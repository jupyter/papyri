/**
 * deleteBundle — remove one (pkg, ver) from the processed store and,
 * optionally, the raw archive.
 *
 * Other bundles' links *into* the deleted bundle are dropped with its nodes;
 * they reappear once the bundle is re-uploaded (or a reingest of the
 * referring bundles runs).
 */
import type { BlobStore } from "./blob-store.js";
import type { GraphDb } from "./graph-db.js";
import type { RawStore } from "./raw-store.js";

export interface DeleteBundleResult {
  deletedBlobs: number;
  deletedNodes: number;
  rawDeleted: boolean;
}

export async function deleteBundle(
  backends: { graphDb: GraphDb; blobStore: BlobStore; rawStore: RawStore },
  pkg: string,
  ver: string,
  opts: { raw?: boolean } = {},
): Promise<DeleteBundleResult> {
  const { graphDb, blobStore, rawStore } = backends;
  const countRow = await graphDb.get<{ n: number }>(
    "SELECT COUNT(*) AS n FROM nodes WHERE package = ? AND version = ?",
    [pkg, ver],
  );
  const inBundle = "(SELECT id FROM nodes WHERE package = ? AND version = ?)";
  await graphDb.batch([
    // Explicit link deletion: don't rely on FK cascade being enabled.
    { sql: `DELETE FROM links WHERE source IN ${inBundle}`, params: [pkg, ver] },
    { sql: `DELETE FROM links WHERE dest IN ${inBundle}`, params: [pkg, ver] },
    { sql: "DELETE FROM nodes WHERE package = ? AND version = ?", params: [pkg, ver] },
    { sql: "DELETE FROM bundles WHERE module = ? AND version = ?", params: [pkg, ver] },
    { sql: "DELETE FROM node_index WHERE pkg = ? AND ver = ?", params: [pkg, ver] },
  ]);
  const deletedBlobs = await blobStore.deleteBundle(pkg, ver);
  let rawDeleted = false;
  if (opts.raw) {
    rawDeleted = (await rawStore.get(pkg, ver)) !== null;
    await rawStore.delete(pkg, ver);
  }
  return { deletedBlobs, deletedNodes: countRow?.n ?? 0, rawDeleted };
}
