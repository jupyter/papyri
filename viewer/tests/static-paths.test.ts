import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { mkdtemp, mkdir, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import Database from "better-sqlite3";
import { FsBlobStore, SqliteGraphDb } from "papyri-ingest";
import {
  bundleParams,
  packageParams,
  qualnameParams,
  docParams,
  exampleParams,
  assetParams,
  type Backends,
} from "../src/lib/static-paths.ts";

const BUNDLES_DDL =
  "CREATE TABLE bundles(module TEXT NOT NULL, version TEXT NOT NULL," +
  " bundle_size_bytes INTEGER NOT NULL, ingested_at INTEGER NOT NULL," +
  " PRIMARY KEY(module, version))";

describe("static-paths.ts", () => {
  let dir: string;
  let backends: Backends;

  beforeEach(async () => {
    dir = await mkdtemp(join(tmpdir(), "papyri-viewer-static-"));

    const dbPath = join(dir, "papyri.db");
    const db = new Database(dbPath);
    db.exec(BUNDLES_DDL);
    const ins = db.prepare(
      "INSERT INTO bundles(module,version,bundle_size_bytes,ingested_at) VALUES (?,?,?,?)"
    );
    ins.run("numpy", "1.26.4", 1, 0);
    ins.run("numpy", "2.0.0", 1, 0);
    ins.run("scipy", "1.13.0", 1, 0);
    db.close();

    const blobs = join(dir, "blobs");
    // Only numpy/2.0.0 gets content; the other bundles stay empty so the
    // enumerators are exercised against bundles with nothing to emit.
    async function write(kind: string, name: string) {
      const d = join(blobs, "numpy", "2.0.0", kind);
      await mkdir(d, { recursive: true });
      await writeFile(join(d, name), "x");
    }
    await write("module", "numpy:array");
    await write("module", "numpy.fft:fft");
    await write("docs", "reference:ufuncs");
    await write("examples", "numpy-array-0");
    await write("assets", "fig-numpy.examples:ex1-0.png");

    backends = {
      blobStore: new FsBlobStore(blobs),
      graphDb: new SqliteGraphDb(new Database(dbPath) as Parameters<typeof SqliteGraphDb>[0]),
    };
  });

  afterEach(async () => {
    await rm(dir, { recursive: true, force: true });
  });

  it("bundleParams: every bundle, concrete versions only (no 'latest' alias)", async () => {
    expect(await bundleParams(backends)).toEqual([
      { pkg: "numpy", ver: "1.26.4" },
      { pkg: "numpy", ver: "2.0.0" },
      { pkg: "scipy", ver: "1.13.0" },
    ]);
  });

  it("packageParams: distinct packages", async () => {
    expect(await packageParams(backends)).toEqual([{ pkg: "numpy" }, { pkg: "scipy" }]);
  });

  it("qualnameParams: ':' slugged to '$' to match linkForQualname", async () => {
    expect(await qualnameParams(backends)).toEqual([
      // Sorted on the raw qualname, where '.' sorts before ':'.
      { pkg: "numpy", ver: "2.0.0", slug: "numpy.fft$fft" },
      { pkg: "numpy", ver: "2.0.0", slug: "numpy$array" },
    ]);
  });

  it("docParams: doc-key ':' becomes a URL path separator", async () => {
    // Inverse of the page's `docParam.split("/").join(":")`.
    expect(await docParams(backends)).toEqual([
      { pkg: "numpy", ver: "2.0.0", doc: "reference/ufuncs" },
    ]);
  });

  it("exampleParams: example paths pass through unchanged", async () => {
    expect(await exampleParams(backends)).toEqual([
      { pkg: "numpy", ver: "2.0.0", ex: "numpy-array-0" },
    ]);
  });

  it("assetParams: filenames slugged the same way linkForAsset slugs them", async () => {
    expect(await assetParams(backends)).toEqual([
      { pkg: "numpy", ver: "2.0.0", asset: "fig-numpy.examples$ex1-0.png" },
    ]);
  });
});
