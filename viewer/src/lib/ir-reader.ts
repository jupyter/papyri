// Async readers over the ingest store, backed by `BlobStore`.
//
// The store layout is `<pkg>/<ver>/<kind>/<path>` (mirrored on disk under
// `~/.papyri/ingest`). This file holds the high-level reader helpers used by
// every page; backend selection happens in `viewer/src/lib/backends.ts`.
//
// The on-disk encoding is CBOR with papyri's tag-numbered IR. We delegate
// decoding to `papyri-ingest`'s `decode()`, which uses a private cbor-x
// Decoder + post-walk so the global cbor-x extension registry isn't
// shared with the SSR ingest encoder running in the same process.

import { decode as decodeIR, type BlobStore, type GraphDb } from "papyri-ingest";
import { IR_TYPE_NAMES } from "./ir-types.ts";
import { linkForAsset } from "./links.ts";
import { qualnameToSlug, slugToQualname } from "./slugs.ts";
export { qualnameToSlug, slugToQualname };

// ---------------------------------------------------------------------------
// Bundle / qualname listings
// ---------------------------------------------------------------------------

export interface IngestedBundle {
  pkg: string;
  version: string;
}

/**
 * Compare two version strings; later version sorts first.
 * Best-effort: numeric segments are compared numerically, the rest
 * falls back to localeCompare. Good enough for typical PEP 440 strings;
 * not a full semver/PEP 440 parser.
 */
export function compareVersionsDesc(a: string, b: string): number {
  const pa = a.split(/[.\-+]/);
  const pb = b.split(/[.\-+]/);
  const n = Math.max(pa.length, pb.length);
  for (let i = 0; i < n; i++) {
    const sa = pa[i] ?? "";
    const sb = pb[i] ?? "";
    const na = /^\d+$/.test(sa) ? Number(sa) : NaN;
    const nb = /^\d+$/.test(sb) ? Number(sb) : NaN;
    if (!Number.isNaN(na) && !Number.isNaN(nb)) {
      if (na !== nb) return nb - na;
      continue;
    }
    const cmp = sa.localeCompare(sb);
    if (cmp !== 0) return -cmp;
  }
  return 0;
}

export interface IngestedPackage {
  pkg: string;
  versions: string[];
  latest: string;
}

/** Distinct packages with their versions, latest first. */
export async function listIngestedPackages(graphDb: GraphDb): Promise<IngestedPackage[]> {
  const bundles = await listBundlesFromDb(graphDb);
  const byPkg = new Map<string, string[]>();
  for (const b of bundles) {
    const arr = byPkg.get(b.pkg) ?? [];
    arr.push(b.version);
    byPkg.set(b.pkg, arr);
  }
  const out: IngestedPackage[] = [];
  for (const [pkg, versions] of byPkg) {
    versions.sort(compareVersionsDesc);
    out.push({ pkg, versions, latest: versions[0]! });
  }
  out.sort((a, b) => a.pkg.localeCompare(b.pkg));
  return out;
}

/**
 * Resolve the string "latest" to the actual latest version for `pkg`.
 * Returns the version unchanged for any other value, or null if `pkg` is
 * not found (only possible when ver === "latest").
 */
export async function resolveVersion(
  graphDb: GraphDb,
  pkg: string,
  ver: string
): Promise<string | null> {
  if (ver !== "latest") return ver;
  const packages = await listIngestedPackages(graphDb);
  return packages.find((p) => p.pkg === pkg)?.latest ?? null;
}

/** Distinct (pkg, version) pairs from the bundles table. */
export async function listBundlesFromDb(graphDb: GraphDb): Promise<IngestedBundle[]> {
  const rows = await graphDb.all<{ module: string; version: string }>(
    "SELECT module, version FROM bundles ORDER BY module, version"
  );
  return rows.map((r) => ({ pkg: r.module, version: r.version }));
}

/** Qualnames under `<pkg>/<ver>/module/`. Sorted. */
export async function listModules(
  blobStore: BlobStore,
  pkg: string,
  version: string
): Promise<string[]> {
  const prefix = `${pkg}/${version}/module/`;
  const keys = await blobStore.list(prefix);
  const out: string[] = [];
  for (const k of keys) {
    const rel = k.slice(prefix.length);
    if (!rel || rel.includes("/")) continue;
    out.push(rel);
  }
  out.sort();
  return out;
}

/**
 * Files under `<pkg>/<ver>/<kind>/`, recursive, returned as POSIX-style
 * paths relative to the kind dir (matching the old `listFilesRecursive`
 * shape). Used for docs/, examples/, assets/.
 */
export async function listFiles(
  blobStore: BlobStore,
  pkg: string,
  version: string,
  kind: string
): Promise<string[]> {
  const prefix = `${pkg}/${version}/${kind}/`;
  const keys = await blobStore.list(prefix);
  const rels = keys.map((k) => k.slice(prefix.length)).filter((r) => r.length > 0);
  rels.sort();
  return rels;
}

// ---------------------------------------------------------------------------
// Decoding helpers
// ---------------------------------------------------------------------------

export interface TypedNode {
  __type: string;
  __tag: number;
  [k: string]: unknown;
}

export interface UnknownNode {
  __type: "unknown";
  __tag: number;
  value: unknown;
}

export type IRNode = TypedNode | UnknownNode;

export const ALL_NODE_TYPES: ReadonlySet<string> = new Set(IR_TYPE_NAMES);

export interface SectionNode {
  __type: "Section";
  __tag: 4015;
  children: IRNode[];
  /** Inline content for the section heading (Text, InlineCode, InlineRole,
   * Link, ...). Empty array means the section has no title. Bundles
   * generated before this schema change may serialise `null` or a plain
   * string here; readers should treat both as "no title" / "single-text
   * title" respectively via `sectionTitleText`. */
  title: IRNode[] | string | null;
  level: number;
  target: string | null;
}

export function sectionChildren(s: SectionNode): IRNode[] {
  return Array.isArray(s.children) ? s.children : [];
}

/** Inline nodes that make up a Section heading, normalised across legacy
 * bundle formats. Returns `[]` when the section is title-less. */
export function sectionTitleNodes(s: SectionNode): IRNode[] {
  const t = s.title;
  if (Array.isArray(t)) return t;
  if (typeof t === "string" && t.length > 0) {
    return [{ __type: "Text", __tag: 4043, value: t } as unknown as IRNode];
  }
  return [];
}

/** Plain-text projection of a Section title, used wherever the heading
 * needs to flow into a `<title>`, a slug, or a nav label. */
export function sectionTitleText(s: SectionNode): string {
  const nodes = sectionTitleNodes(s);
  return nodes.map(inlineNodeText).join("");
}

function inlineNodeText(n: IRNode): string {
  if (!n || typeof n !== "object") return "";
  const v = (n as { value?: unknown }).value;
  if (typeof v === "string") return v;
  const kids = (n as { children?: unknown }).children;
  if (Array.isArray(kids)) return kids.map(inlineNodeText).join("");
  return "";
}

export interface SigParamT {
  __type: "SigParam";
  __tag: 4030;
  name: string;
  annotation: string | EmptyNode | null;
  kind: string;
  default: string | EmptyNode | null;
}

export interface EmptyNode {
  __type: "Empty";
  __tag: 4031;
}

export interface ParamRefT {
  __type: "ParamRef";
  __tag: 4071;
  name: string;
}

export interface SignatureNodeT {
  __type: "SignatureNode";
  __tag: 4029;
  kind: string;
  parameters: SigParamT[];
  return_annotation: string | EmptyNode;
  target_name: string;
}

export interface IngestedDoc {
  __type: "IngestedDoc";
  __tag: 4010;
  _content: Record<string, SectionNode>;
  _ordered_sections: string[];
  item_file: string | null;
  item_line: number | null;
  item_type: string | null;
  aliases: string[];
  example_section_data: SectionNode | null;
  see_also: IRNode[];
  signature: SignatureNodeT | null;
  references: string[] | null;
  qa: string;
  arbitrary: SectionNode[];
}

/** Decode raw CBOR bytes (re-export of papyri-ingest's decoder). */
export function decodeCborBytes<T = unknown>(bytes: Uint8Array | Buffer): T {
  return decodeIR<T>(bytes);
}

// ---------------------------------------------------------------------------
// Per-key loaders
// ---------------------------------------------------------------------------

/** Load a module-page IngestedDoc by qualname. Throws if absent or wrong type. */
export async function loadModule(
  blobStore: BlobStore,
  pkg: string,
  version: string,
  qualname: string
): Promise<IngestedDoc> {
  // The blob path is the qualname verbatim (`Ingester.ingestBundle` stages
  // `{ kind: "module", path: qa }`). There is no `.cbor` suffix to fall back
  // to, and guessing one would shadow real qualnames: `papyri.node_base:Node`
  // and the `Node.cbor` method both exist in papyri's own bundle.
  const bytes = await blobStore.get({ module: pkg, version, kind: "module", path: qualname });
  if (!bytes) throw new Error(`module not found: ${pkg}/${version}/${qualname}`);
  const obj = decodeIR<IRNode>(bytes);
  if (obj && (obj as IRNode).__type === "IngestedDoc") {
    return obj as unknown as IngestedDoc;
  }
  throw new Error(`unexpected decode result for ${qualname}: ${typeof obj}`);
}

/** Generic loader: read `<pkg>/<ver>/<kind>/<path>` and CBOR-decode it. */
export async function loadCbor<T = unknown>(
  blobStore: BlobStore,
  pkg: string,
  version: string,
  kind: string,
  path: string
): Promise<T> {
  const bytes = await blobStore.get({ module: pkg, version, kind, path });
  if (!bytes) throw new Error(`blob not found: ${pkg}/${version}/${kind}/${path}`);
  return decodeIR<T>(bytes);
}

/** Read raw asset bytes (no decode). Returns null if absent. */
export async function loadAsset(
  blobStore: BlobStore,
  pkg: string,
  version: string,
  path: string
): Promise<Uint8Array | null> {
  return blobStore.get({ module: pkg, version, kind: "assets", path });
}

// ---------------------------------------------------------------------------
// Generic IR node collection
// ---------------------------------------------------------------------------

export function collectNodes(
  node: unknown,
  types: ReadonlySet<string>,
  out: IRNode[] = []
): IRNode[] {
  if (!node || typeof node !== "object") return out;
  if (Array.isArray(node)) {
    for (const item of node) collectNodes(item, types, out);
    return out;
  }
  const n = node as Record<string, unknown>;
  if (typeof n.__type === "string" && types.has(n.__type)) {
    out.push(n as IRNode);
  }
  for (const val of Object.values(n)) {
    if (val && typeof val === "object") collectNodes(val, types, out);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Image collection
// ---------------------------------------------------------------------------

export type FoundImgNode =
  | { kind: "Figure"; src: string; assetPath: string }
  | { kind: "Image"; src: string; alt: string };

export function collectImages(node: unknown): FoundImgNode[] {
  const out: FoundImgNode[] = [];
  for (const n of collectNodes(node, new Set(["Figure", "Image"]))) {
    if (n.__type === "Figure") {
      const ref = (n as TypedNode).value as
        | { module?: string; version?: string; kind?: string; path?: string }
        | undefined;
      if (ref?.kind === "assets" && ref.module && ref.version && ref.path) {
        const assetPath = String(ref.path);
        out.push({
          kind: "Figure",
          src: linkForAsset(ref.module, ref.version, assetPath),
          assetPath,
        });
      }
    } else if (n.__type === "Image") {
      const url = String((n as TypedNode).url ?? "");
      if (url) out.push({ kind: "Image", src: url, alt: String((n as TypedNode).alt ?? "") });
    }
  }
  return out;
}
