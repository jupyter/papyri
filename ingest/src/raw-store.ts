/**
 * RawStore — append-only archive of uploaded .papyri.gz bundles.
 *
 * Every bundle received by PUT /api/bundle is written here verbatim (the
 * compressed bytes as received off the wire) before ingest runs. This raw
 * archive is the source-of-truth for the processed store: if the ingest
 * schema changes or a pipeline bug is found, the processed store (BlobStore
 * + GraphDb) can be wiped and rebuilt from these raw bytes without requiring
 * maintainers to re-upload.
 *
 * Key: _raw/<pkg>/<ver>.papyri.gz
 * Metadata (sidecar): _raw/<pkg>/<ver>.meta.json
 *
 *   FsRawStore — Node filesystem, files at <ingest-dir>/_raw/<pkg>/<ver>.papyri.gz
 */
import { mkdir, writeFile, readFile, readdir, rm } from "node:fs/promises";
import { join } from "node:path";
import { safeJoin } from "./fs-safe.js";

export interface RawMeta {
  /** ISO 8601 date string of when the bundle was received (server wall-clock time) */
  received_at: string;
}

export interface RawStore {
  /** Archive a raw .papyri.gz bundle (compressed bytes as received off the wire). */
  put(pkg: string, ver: string, bytes: Uint8Array): Promise<void>;
  /** Retrieve a previously archived bundle. Returns null if absent. */
  get(pkg: string, ver: string): Promise<Uint8Array | null>;
  /** Retrieve metadata for a previously archived bundle. Returns null if absent. */
  getMeta(pkg: string, ver: string): Promise<RawMeta | null>;
  /** List all archived (pkg, ver) pairs, sorted by pkg then ver. */
  list(): Promise<Array<{ pkg: string; ver: string }>>;
  /**
   * Drop every archived bundle. Returns the number of (pkg, ver) entries
   * removed. Idempotent — clearing an already-empty archive returns 0.
   */
  clear(): Promise<number>;
  /** Remove one archived bundle (and its metadata). Idempotent. */
  delete(pkg: string, ver: string): Promise<void>;
}

// ---------------------------------------------------------------------------
// Node filesystem
// ---------------------------------------------------------------------------

export class FsRawStore implements RawStore {
  constructor(private readonly root: string) {}

  private fullPath(pkg: string, ver: string): string {
    return safeJoin(this.root, "_raw", pkg, `${ver}.papyri.gz`);
  }

  private metaPath(pkg: string, ver: string): string {
    return safeJoin(this.root, "_raw", pkg, `${ver}.meta.json`);
  }

  async put(pkg: string, ver: string, bytes: Uint8Array): Promise<void> {
    const p = this.fullPath(pkg, ver);
    await mkdir(safeJoin(this.root, "_raw", pkg), { recursive: true });
    await writeFile(p, bytes);
    // Write metadata sidecar with received timestamp (server wall-clock time)
    const meta: RawMeta = { received_at: new Date().toISOString() };
    await writeFile(this.metaPath(pkg, ver), JSON.stringify(meta));
  }

  async get(pkg: string, ver: string): Promise<Uint8Array | null> {
    try {
      const buf = await readFile(this.fullPath(pkg, ver));
      return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code === "ENOENT") return null;
      throw err;
    }
  }

  async getMeta(pkg: string, ver: string): Promise<RawMeta | null> {
    try {
      const content = await readFile(this.metaPath(pkg, ver), "utf-8");
      return JSON.parse(content) as RawMeta;
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code === "ENOENT") return null;
      throw err;
    }
  }

  async list(): Promise<Array<{ pkg: string; ver: string }>> {
    const rawDir = join(this.root, "_raw");
    const results: Array<{ pkg: string; ver: string }> = [];
    let pkgEntries;
    try {
      pkgEntries = await readdir(rawDir, { withFileTypes: true });
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code === "ENOENT") return [];
      throw err;
    }
    for (const pkgEnt of pkgEntries) {
      if (!pkgEnt.isDirectory()) continue;
      const verEntries = await readdir(join(rawDir, pkgEnt.name), { withFileTypes: true });
      for (const verEnt of verEntries) {
        if (!verEnt.isFile() || !verEnt.name.endsWith(".papyri.gz")) continue;
        results.push({ pkg: pkgEnt.name, ver: verEnt.name.slice(0, -".papyri.gz".length) });
      }
    }
    results.sort((a, b) => a.pkg.localeCompare(b.pkg) || a.ver.localeCompare(b.ver));
    return results;
  }

  async delete(pkg: string, ver: string): Promise<void> {
    await rm(this.fullPath(pkg, ver), { force: true });
    await rm(this.metaPath(pkg, ver), { force: true });
  }

  async clear(): Promise<number> {
    const entries = await this.list();
    await rm(join(this.root, "_raw"), { recursive: true, force: true });
    return entries.length;
  }
}
