import { describe, it, expect, beforeAll, beforeEach, afterEach } from "vitest";
import { mkdtemp, mkdir, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Encoder, Tag } from "cbor-x";
import { FsBlobStore } from "papyri-ingest";

const tg = (tag: number, fields: unknown[]) => new Tag(fields, tag);
const enc = new Encoder({ useRecords: false });

// IngestedDoc (4010) fields, in declared order (see ir-reader FIELD_ORDER).
const mkDoc = (qa: string, o: Record<string, unknown> = {}) =>
  tg(4010, [
    {},
    [],
    o.item_file ?? null,
    o.item_line ?? null,
    o.item_type ?? null,
    [],
    null,
    [],
    o.signature ?? null,
    [],
    qa,
    o.arbitrary ?? [],
  ]);

const bytesFull = enc.encode(
  mkDoc("pkg.mod:foo", {
    item_file: "foo.py",
    item_line: 42,
    item_type: "function",
    signature: tg(4029, ["function", [], tg(4031, []), "foo"]),
  })
);
const bytesBar = enc.encode(mkDoc("pkg:bar"));
const bytesUnknown = enc.encode(mkDoc("pkg:qux", { arbitrary: [tg(9999, ["mystery"])] }));

let loadModule: typeof import("../src/lib/ir-reader.ts").loadModule;
beforeAll(async () => {
  ({ loadModule } = await import("../src/lib/ir-reader.ts"));
});

describe("loadModule (CBOR roundtrip)", () => {
  let dir: string;
  beforeEach(async () => {
    dir = await mkdtemp(join(tmpdir(), "papyri-viewer-cbor-"));
  });
  afterEach(async () => {
    await rm(dir, { recursive: true, force: true });
  });

  // Writes a blob at <dir>/pkg/1.0/module/<name> and returns an FsBlobStore
  // rooted at <dir>.
  const writeBlob = async (name: string, bytes: Uint8Array): Promise<FsBlobStore> => {
    const moduleDir = join(dir, "pkg", "1.0", "module");
    await mkdir(moduleDir, { recursive: true });
    await writeFile(join(moduleDir, name), bytes);
    return new FsBlobStore(dir);
  };

  it("decodes an IngestedDoc (tag 4010) with nested tagged children", async () => {
    const store = await writeBlob("pkg.mod:foo", bytesFull);
    const out = await loadModule(store, "pkg", "1.0", "pkg.mod:foo");
    expect(out.__type).toBe("IngestedDoc");
    expect(out.__tag).toBe(4010);
    expect(out.qa).toBe("pkg.mod:foo");
    expect(out.item_file).toBe("foo.py");
    expect(out.item_line).toBe(42);
    const s = out.signature as {
      __type: string;
      target_name: string;
      return_annotation: { __type: string };
    };
    expect(s.__type).toBe("SignatureNode");
    expect(s.target_name).toBe("foo");
    expect(s.return_annotation.__type).toBe("Empty");
  });

  it("reads the blob at the qualname verbatim, with no .cbor guessing", async () => {
    // A qualname ending in ".cbor" is a real method, not a suffixed filename:
    // guessing would make `pkg:bar` resolve to the `pkg:bar.cbor` page.
    const store = await writeBlob("pkg:bar.cbor", bytesBar);
    await expect(loadModule(store, "pkg", "1.0", "pkg:bar")).rejects.toThrow("module not found");
    expect((await loadModule(store, "pkg", "1.0", "pkg:bar.cbor")).qa).toBe("pkg:bar");
  });

  it("wraps unregistered inner tags as UnknownNode", async () => {
    const store = await writeBlob("pkg:qux", bytesUnknown);
    const out = await loadModule(store, "pkg", "1.0", "pkg:qux");
    const arb = out.arbitrary as unknown[];
    expect(arb).toHaveLength(1);
    const inner = arb[0] as { __type: string; __tag: number; value: unknown };
    expect(inner.__type).toBe("unknown");
    expect(inner.__tag).toBe(9999);
    expect(inner.value).toEqual(["mystery"]);
  });
});
