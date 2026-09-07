/**
 * Not a test: a dev/CI utility run through vitest so it can import
 * `papyri-ingest` (whose `import` condition points at TypeScript source).
 *
 * Seeds an ingest store from a `.papyri` artifact exactly the way
 * `PUT /api/bundle` does — gunzip -> CBOR decode -> `Ingester.ingestBundle` —
 * so a static build has something to render:
 *
 *   PAPYRI_INGEST_DIR=/tmp/store pnpm seed-store
 *
 * It carries a `.test.ts` suffix only because vitest runs test files; `pnpm
 * test` is scoped to `tests/` so this never runs as part of the suite.
 */
import { it } from "vitest";
import { readFileSync, mkdirSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import Database from "better-sqlite3";
import {
  applyMigrations,
  decode,
  Ingester,
  FsBlobStore,
  FsRawStore,
  SqliteGraphDb,
} from "papyri-ingest";

it("seeds an ingest store from a .papyri artifact", async () => {
  const fixture = process.env.PAPYRI_SEED_FIXTURE ?? "tests/fixtures/papyri-0.0.10.papyri";
  const outDir = process.env.PAPYRI_INGEST_DIR;
  if (!outDir) throw new Error("PAPYRI_INGEST_DIR must be set");

  mkdirSync(outDir, { recursive: true });
  const db = new Database(join(outDir, "papyri.db"));
  const sentinel = import.meta.resolve("papyri-ingest/migrations/0001_init.sql");
  applyMigrations(db as never, dirname(fileURLToPath(sentinel)));

  const compressed = readFileSync(fixture);
  const bundle = decode<Parameters<Ingester["ingestBundle"]>[0]>(gunzipSync(compressed));
  const ingester = new Ingester(
    new FsBlobStore(outDir),
    new SqliteGraphDb(db as never),
    new FsRawStore(outDir)
  );
  // The size argument is what writes the `bundles` row the viewer lists from.
  await ingester.ingestBundle(bundle, compressed.byteLength);
  console.log(`seeded ${fixture} -> ${outDir}`);
});
