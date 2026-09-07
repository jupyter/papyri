import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { mkdtemp, mkdir, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import Database from "better-sqlite3";
import { FsBlobStore, SqliteGraphDb } from "papyri-ingest";
import { listBundlesFromDb, listModules } from "../src/lib/ir-reader.ts";

function seedBundlesDb(path: string) {
  const db = new Database(path);
  db.exec(
    "CREATE TABLE bundles(module TEXT NOT NULL, version TEXT NOT NULL, " +
      "bundle_size_bytes INTEGER NOT NULL, ingested_at INTEGER NOT NULL, " +
      "PRIMARY KEY(module, version))"
  );
  const ins = db.prepare(
    "INSERT INTO bundles(module, version, bundle_size_bytes, ingested_at) VALUES (?,?,?,?)"
  );
  ins.run("numpy", "1.26.4", 100, 0);
  ins.run("numpy", "2.0.0", 200, 0);
  ins.run("scipy", "1.13.0", 300, 0);
  db.close();
}

describe("fs-facing helpers", () => {
  let dir: string;
  beforeEach(async () => {
    dir = await mkdtemp(join(tmpdir(), "papyri-viewer-fs-"));
  });
  afterEach(async () => {
    await rm(dir, { recursive: true, force: true });
  });

  it("listBundlesFromDb: [] on empty bundles table, discovers rows", async () => {
    const dbPath = join(dir, "empty.db");
    const emptyDb = new Database(dbPath);
    emptyDb.exec(
      "CREATE TABLE bundles(module TEXT NOT NULL, version TEXT NOT NULL, " +
        "bundle_size_bytes INTEGER NOT NULL, ingested_at INTEGER NOT NULL, " +
        "PRIMARY KEY(module, version))"
    );
    emptyDb.close();
    const emptyGraphDb = new SqliteGraphDb(
      new Database(dbPath) as Parameters<typeof SqliteGraphDb>[0]
    );
    expect(await listBundlesFromDb(emptyGraphDb)).toEqual([]);

    const seededPath = join(dir, "seeded.db");
    seedBundlesDb(seededPath);
    const seededGraphDb = new SqliteGraphDb(
      new Database(seededPath) as Parameters<typeof SqliteGraphDb>[0]
    );
    const got = await listBundlesFromDb(seededGraphDb);
    expect(got.map((b) => `${b.pkg}/${b.version}`)).toEqual([
      "numpy/1.26.4",
      "numpy/2.0.0",
      "scipy/1.13.0",
    ]);
  });

  it("listModules: missing module/ -> [], filenames are qualnames verbatim", async () => {
    const store = new FsBlobStore(dir);
    expect(await listModules(store, "pkg", "1.0")).toEqual([]);

    const modDir = join(dir, "pkg", "1.0", "module");
    await mkdir(modDir, { recursive: true });
    await writeFile(join(modDir, "papyri.node_base:Node"), "x");
    // A method genuinely named `cbor`. Stripping a trailing ".cbor" would
    // collapse this onto the class above and lose the page.
    await writeFile(join(modDir, "papyri.node_base:Node.cbor"), "x");
    // Subdirectories are ignored (rel.includes("/")).
    await mkdir(join(modDir, "subdir"));
    await writeFile(join(modDir, "subdir", "ignored"), "x");

    expect(await listModules(store, "pkg", "1.0")).toEqual([
      "papyri.node_base:Node",
      "papyri.node_base:Node.cbor",
    ]);
  });
});
