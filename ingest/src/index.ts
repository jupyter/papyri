/**
 * papyri-ingest — public library API
 *
 * Core exports for programmatic use. Consumed by the viewer's
 * PUT /api/bundle ingest endpoint.
 */

export { Ingester, applyMigrations } from "./ingest.js";
export type { IngestOptions, ProgressCallback } from "./ingest.js";

export type { Key } from "./keys.js";

export { FsBlobStore, keyToPath } from "./blob-store.js";
export type { BlobStore } from "./blob-store.js";

export { SqliteGraphDb } from "./graph-db.js";
export type { GraphDb, BatchStmt, GraphRow, NodeIndexRow } from "./graph-db.js";

export { decode, encode, generatedDocToIngested, FIELD_ORDER } from "./encoder.js";
export type { IRNode, TypedNode, UnknownNode } from "./encoder.js";

export { assertBundle, assertSafeUrls } from "./bundle.js";

export { isSafeUrl } from "./url-safety.js";

export { collectForwardRefs, collectForwardRefsFromSection } from "./visitor.js";

export { deleteBundle } from "./delete-bundle.js";
export type { DeleteBundleResult } from "./delete-bundle.js";

export { FsRawStore } from "./raw-store.js";
export type { RawStore, RawMeta } from "./raw-store.js";

export {
  parseObjectsInv,
  registerProject,
  resolveExternalUri,
  storeInventory,
  unloadProject,
} from "./inventory.js";
export type { InventoryObject, ParsedInventory } from "./inventory.js";
