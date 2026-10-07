/**
 * BlobStore — async key→bytes store used by the Ingester.
 *
 * FsBlobStore — Node filesystem rooted at a directory. Keyed by
 * `<module>/<version>/<kind>/<path>`.
 *
 * Per-bundle metadata (`meta.cbor`) lives at `<module>/<version>/meta.cbor`
 * — outside the (kind,path) addressing scheme, which is why it needs its
 * own helper.
 */
import { mkdir, writeFile, readFile, stat, readdir, rm } from "node:fs/promises";
import { join, dirname, sep } from "node:path";
import type { Key } from "./keys.js";
import { safeJoin } from "./fs-safe.js";

/** Flat path used as the blob key (an fs path-suffix, or an object-store key). */
export function keyToPath(key: Key): string {
  return `${key.module}/${key.version}/${key.kind}/${key.path}`;
}

export interface BlobStore {
  put(key: Key, bytes: Uint8Array): Promise<void>;
  get(key: Key): Promise<Uint8Array | null>;
  has(key: Key): Promise<boolean>;
  /**
   * List every key under *prefix*, recursive (fs walks recursively).
   * Returned strings are full
   * keys including the prefix; sorted lexicographically. Empty array if
   * the prefix is absent.
   *
   * Both backends paginate internally so callers see a single materialised
   * list. Use coarse prefixes (`<pkg>/<ver>/`) to keep page counts bounded.
   */
  list(prefix: string): Promise<string[]>;
  /** Per-bundle meta.cbor (outside the {kind,path} address space). */
  putMeta(module: string, version: string, bytes: Uint8Array): Promise<void>;
  /** Read per-bundle meta.cbor. Null if absent. */
  getMeta(module: string, version: string): Promise<Uint8Array | null>;
  /**
   * Delete every processed blob (and per-bundle meta.cbor) without touching
   * the raw archive (`_raw/` prefix) that lives in the same backend.
   * Returns the number of objects deleted. Idempotent — clearing an
   * already-empty store is a no-op.
   */
  clear(): Promise<number>;
  /**
   * Delete every blob (and meta.cbor) of one bundle, leaving other bundles and
   * the raw archive alone. Returns the number of blobs deleted (excluding
   * meta). Idempotent.
   */
  deleteBundle(module: string, version: string): Promise<number>;
}

// ---------------------------------------------------------------------------
// Node filesystem
// ---------------------------------------------------------------------------

export class FsBlobStore implements BlobStore {
  constructor(private readonly root: string) {}

  private fullPath(key: Key): string {
    return safeJoin(this.root, key.module, key.version, key.kind, key.path);
  }

  async put(key: Key, bytes: Uint8Array): Promise<void> {
    const p = this.fullPath(key);
    await mkdir(dirname(p), { recursive: true });
    await writeFile(p, bytes);
  }

  async get(key: Key): Promise<Uint8Array | null> {
    try {
      const buf = await readFile(this.fullPath(key));
      return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code === "ENOENT") return null;
      throw err;
    }
  }

  async has(key: Key): Promise<boolean> {
    try {
      await stat(this.fullPath(key));
      return true;
    } catch {
      return false;
    }
  }

  async putMeta(module: string, version: string, bytes: Uint8Array): Promise<void> {
    const p = safeJoin(this.root, module, version, "meta.cbor");
    await mkdir(dirname(p), { recursive: true });
    await writeFile(p, bytes);
  }

  async getMeta(module: string, version: string): Promise<Uint8Array | null> {
    try {
      const buf = await readFile(safeJoin(this.root, module, version, "meta.cbor"));
      return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code === "ENOENT") return null;
      throw err;
    }
  }

  async clear(): Promise<number> {
    // Walk top-level entries at the root; remove every module directory
    // (anything that isn't the raw archive or the SQLite DB). The blob
    // namespace is `<module>/<version>/...`, so module dirs are exactly
    // the top-level entries we need to delete. _raw/ and papyri.db* live
    // at the same root and must be preserved.
    let ents;
    try {
      ents = await readdir(this.root, { withFileTypes: true });
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code === "ENOENT") return 0;
      throw err;
    }
    let count = 0;
    for (const e of ents) {
      if (e.name === "_raw") continue;
      if (e.name === "papyri.db" || e.name.startsWith("papyri.db-")) continue;
      await rm(join(this.root, e.name), { recursive: true, force: true });
      count++;
    }
    return count;
  }

  async deleteBundle(module: string, version: string): Promise<number> {
    const dir = safeJoin(this.root, module, version);
    const count = (await this.list(`${module}/${version}/`)).length;
    await rm(dir, { recursive: true, force: true });
    return count;
  }

  async list(prefix: string): Promise<string[]> {
    const startDir = join(this.root, prefix);
    const out: string[] = [];
    async function walk(dir: string): Promise<void> {
      let ents;
      try {
        ents = await readdir(dir, { withFileTypes: true });
      } catch (err) {
        if ((err as NodeJS.ErrnoException).code === "ENOENT") return;
        throw err;
      }
      for (const e of ents) {
        const full = join(dir, e.name);
        if (e.isDirectory()) {
          await walk(full);
        } else if (e.isFile()) {
          out.push(full);
        }
      }
    }
    await walk(startDir);
    // Convert absolute paths back to root-relative keys (POSIX-style).
    const rootSlash = this.root.endsWith(sep) ? this.root : this.root + sep;
    const rels = out.map((p) => p.slice(rootSlash.length).split(sep).join("/"));
    rels.sort();
    return rels;
  }
}
