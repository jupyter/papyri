import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import Database from "better-sqlite3";
import { SqliteGraphDb, type BlobStore } from "papyri-ingest";
import { collectBundleImages } from "../src/lib/image-index.ts";

const MIGRATION =
  "CREATE TABLE node_index (id INTEGER PRIMARY KEY AUTOINCREMENT, pkg TEXT NOT NULL," +
  " ver TEXT NOT NULL, node_type TEXT NOT NULL, content TEXT NOT NULL," +
  " page_href TEXT NOT NULL, page_kind TEXT NOT NULL, page_qa TEXT NOT NULL);";

/** The image index never touches blobs when node_index has rows. */
const unusedBlobStore = {} as BlobStore;

describe("image-index.ts", () => {
  let dir: string;

  beforeEach(async () => {
    dir = await mkdtemp(join(tmpdir(), "papyri-viewer-images-"));
  });
  afterEach(async () => {
    await rm(dir, { recursive: true, force: true });
  });

  function seed(rows: { node_type: string; content: unknown; page_qa: string }[]): SqliteGraphDb {
    const path = join(dir, "papyri.db");
    const db = new Database(path);
    db.exec(MIGRATION);
    const ins = db.prepare(
      "INSERT INTO node_index(pkg,ver,node_type,content,page_href,page_kind,page_qa)" +
        " VALUES (?,?,?,?,?,?,?)"
    );
    for (const r of rows) {
      ins.run(
        "papyri",
        "1.0.0",
        r.node_type,
        JSON.stringify(r.content),
        `/project/papyri/1.0.0/${r.page_qa}/`,
        "api",
        r.page_qa
      );
    }
    db.close();
    return new SqliteGraphDb(new Database(path) as Parameters<typeof SqliteGraphDb>[0]);
  }

  it("Figure src matches the /assets/project/ route the asset endpoint serves", async () => {
    // Regression: the index used to emit `/assets/<pkg>/<ver>/…`, which 404s —
    // `pages/assets/project/[pkg]/[ver]/[...asset].ts` serves `/assets/project/…`.
    const graphDb = seed([
      {
        node_type: "Figure",
        content: {
          value: { module: "papyri", version: "1.0.0", kind: "assets", path: "fig-a:b-0.png" },
        },
        page_qa: "papyri:thing",
      },
    ]);
    const entries = await collectBundleImages(graphDb, unusedBlobStore, "papyri", "1.0.0");
    expect(entries).toHaveLength(1);
    // ':' in the asset filename is slugged to '$' so the URL stays path-safe.
    expect(entries[0].img.src).toBe("/assets/project/papyri/1.0.0/fig-a$b-0.png");
  });

  it("rewrites page hrefs to the URL version (e.g. 'latest')", async () => {
    const graphDb = seed([
      {
        node_type: "Image",
        content: { url: "https://example.com/x.png", alt: "x" },
        page_qa: "papyri:thing",
      },
    ]);
    const entries = await collectBundleImages(
      graphDb,
      unusedBlobStore,
      "papyri",
      "1.0.0",
      "latest"
    );
    expect(entries[0].pages[0].href).toBe("/project/papyri/latest/papyri:thing/");
  });
});
