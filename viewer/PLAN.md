# Viewer — web renderer for papyri IR

A **read-only web viewer** that renders documentation from ingested papyri
bundles. The viewer lives in-tree while the IR is in flux; co-locating
producer and consumer lets us iterate across breaking changes in one PR.
Splitting into a separate repo remains an option once the IR schema stabilizes.

> **IR stability contract.** `src/lib/ir-reader.ts` is the designated shock
> absorber — when the IR changes, the fix lands there, not spread across
> components. Treat it as the only place allowed to know the on-disk format.

## Goals

1. Serve browsable HTML for every ingested package/module/qualname.
2. Consume the IR through an abstract storage layer — no Python-side
   rendering, no new intermediate format.
3. Be the rendering + ingest frontend for the hosted service: long-running
   Node server, in-process upload, auth, per-project upload authorization.
4. Stay a thin projection of the IR: the graphstore remains a derived,
   rebuildable cache; IR-format knowledge stays confined to `ir-reader.ts`.

> The original v0 non-goals (no auth, no multi-tenant hosting, no full-text
> search, no database beyond what papyri provides) were all overtaken as the
> viewer became the hosted-service frontend: the separate auth DB,
> per-project upload tokens, the admin panel, and full-text search now
> exist. This section is updated to describe the project as it is, not as
> v0 imagined it — keep it that way rather than letting scope drift
> silently.

## Non-goals

- Running or re-executing examples.
- Authoring, comments, edit-in-browser.
- Changing the IR format from inside the viewer (see Ground rules).

## Features

### Must-have (v0)

- **Package index**: list ingested `(pkg, version)` bundles.
- **Module / page index**: TOC per bundle.
- **Qualname page**: signature, parameters, description, see-also, notes,
  examples.
- **Cross-links**: forward links resolve via the graph; 404 → nearest match.
- **Back-references**: "used by" / "referenced from" section.
- **Math**: KaTeX (server-side at build time).
- **Code highlighting**: Python, text, console.
- **Example blocks**: render captured stdout/plots/HTML assets from the
  bundle's `assets/` dir.
- **Static build**: pre-rendered HTML + assets for hosting behind any static
  file server.

### Nice-to-have (later)

- Prefix / fuzzy search (client-side index built at export time).
- Dark mode toggle.
- Permalink copy for any anchor.
- Per-bundle version picker.
- Diff view between versions of the same qualname.

### Open follow-ups

- **Bundle-walk shared helper — done; ingest-time index — done.** Walk logic
  consolidated in `lib/bundle-walk.ts`; `node_index` table precomputed at ingest
  time by `Ingester._populateNodeIndex()` (migration `0006_node_index.sql`).
  Image-index and node-browser endpoints now query `node_index` instead of
  scanning all blobs; both fall back to `walkBundle` for pre-migration bundles.

## Tech choices

| Area             | Choice                        | Why                                          |
| ---------------- | ----------------------------- | -------------------------------------------- |
| Language         | TypeScript                    | Typed IR = fewer renderer bugs               |
| Runtime          | Node LTS (long-running server)| Local dev + hosted service (VPS) target      |
| Framework        | Astro (SSG + SSR islands)     | Current choice; not locked in permanently    |
| UI components    | React (inside Astro islands)  | Familiar; may revisit with framework choice  |
| Graph client     | abstracted (`GraphDb`)        | SQLite today; abstraction allows a swap later|
| Blob storage     | abstracted (`BlobStore`)      | Filesystem today; abstraction allows a swap  |
| Math             | `katex` (server-side)         | No JS runtime shipped to the client          |
| Syntax highlight | `shiki`                       | Zero-runtime, VS Code grammars               |
| Styling          | Plain CSS + CSS custom props  | No Tailwind yet; keep the surface small      |
| Package manager  | `pnpm`                        | Workspace-ready for the ingest sibling       |
| Lint / format    | ESLint + Prettier             | Standard                                     |
| Tests            | Vitest + Playwright smoke     | Unit for IR reader; e2e for golden pages     |

## Architecture sketch

```
viewer/
├── PLAN.md
├── package.json
├── astro.config.ts
├── src/
│   ├── lib/
│   │   ├── ir-reader.ts     # load bundle, decode blobs, typed IR
│   │   ├── backends.ts      # getBackends() → BlobStore + GraphDb + RawStore
│   │   ├── graph.ts         # graph queries over GraphDb
│   │   └── paths.ts         # discovery, env override
│   ├── components/          # React islands: Signature, Param, SeeAlso, …
│   ├── pages/
│   │   ├── index.astro      # list of (pkg, version) bundles
│   │   ├── [pkg]/[ver]/index.astro
│   │   └── [pkg]/[ver]/[...slug].astro   # qualname pages
│   └── styles/
└── tests/
```

Data flow per request/page:

1. Resolve `(pkg, ver, qualname)` from the URL.
2. `storage` retrieves the matching blob from the bundle.
3. `ir-reader` decodes the blob into typed IR nodes.
4. `graph` resolves forward and back references.
5. Astro page renders IR nodes → JSX. No ad-hoc HTML strings.

## Config

- `PAPYRI_INGEST_DIR` — defaults to `~/.papyri/ingest`. Used by the Node
  filesystem backend.
- `PAPYRI_INGEST_DB` — defaults to `~/.papyri/ingest/papyri.db`. Used by
  the SQLite graph backend.
- `--mode dev | build` via Astro.

## Milestones

All milestones through M8 are complete. The viewer runs as a
long-running Node.js server (`@astrojs/node`, `output: "server"`) on a
VPS; newly uploaded bundles appear without a rebuild.

The pieces below landed during the (now abandoned) Cloudflare Workers
exploration and are kept because they stand on their own:

- [x] **Async storage + graph layer.** `BlobStore` / `GraphDb` /
      `RawStore` abstractions in `papyri-ingest`, built per-request by
      `viewer/src/lib/backends.ts`.
      `viewer/src/lib/{ir-reader,graph,nav,image-index,xref}.ts` are
      async and parameterised on the backend triple; every page calls
      `getBackends()` and passes it down. CrossRef resolution batches
      once per page via `buildXrefResolver(graphDb, doc)` so render
      components stay sync.
- [x] **Bundle upload in-process.** `PUT /api/bundle` gunzips the body
      via `DecompressionStream`, CBOR-decodes it to a `Bundle` Node, and
      hands it to `Ingester.ingestBundle(node)` — no temp dir, no `tar`
      spawn. Write path uses subquery-based link inserts so a single
      `db.batch([…])` is atomic.
- [x] **Static snapshot of the whole store.** `PAPYRI_STATIC=1` flips
      `output` to `"static"`; each public route's `getStaticPaths` returns
      an enumerator from `src/lib/static-paths.ts`, which walks the graph
      and blob store (a crawler cannot reach pages nothing links to).
      Astro renders the *same* templates in both modes, so the two cannot
      diverge in content; the only mode-dependent code is
      `isStaticBuild()` in `src/lib/static.ts`, and every use of it drops
      an affordance that needs a live server rather than changing how
      anything renders. `scripts/check-static-links.mjs` (run in CI)
      fails if any emitted page links to a page no enumerator produced.

      Omitted from a snapshot, by decision: every dynamic island (per-bundle
      and global text search, the node browser), the query-string endpoints
      behind them, the raw-IR link, the version diff form, and all
      auth/admin/upload routes. Class members render eagerly because
      `server:defer` has no server to defer to. `/latest/` is not emitted —
      cards link concrete versions, since aliasing would mean a second full
      copy of the largest bundle.

      Open follow-ups:
      - **Sub-path hosting.** `linkFor*` emits root-absolute paths, so a
        snapshot only works at a domain root. Needs a `withBase()` pass over
        `src/lib/links.ts` (plus the literal `href="/"` / `/favicon.png` /
        island fetch URLs that bypass it today) before `base` can be honoured.
      - **`/latest/` deep links.** Nothing serves them in a snapshot. Cheapest
        fix: a redirect script in `404.astro` that reads a prerendered
        `bundles.json` — needs a host that serves `404.html`.
      - **Build scale.** Measured only on a one-bundle store (571 pages, ~5s).
        A numpy+scipy+matplotlib store is the real test; `BundleLayout` runs
        several SQLite queries per page and eager members double the render
        work for class-heavy bundles.
      - **`file://` browsing** is not achievable: browsers block `fetch()`
        from `file://`, so any data-loading island is dead there. Serve with
        `python -m http.server -d dist/client`.
- [x] **Raw bundle archive.** Every `PUT /api/bundle` archives the
      compressed `.papyri.gz` bytes to `_raw/<pkg>/<ver>.papyri.gz`
      (`<ingest-dir>/_raw/` on the filesystem) before ingest runs.
      `POST /api/reingest` (auth-gated, NDJSON stream) replays the raw
      archive through a fresh ingest — supports `?pkg=` / `?ver=` to
      scope to one bundle. `RawStore` interface + `FsRawStore` live in
      `ingest/src/raw-store.ts`.

### Cloudflare Workers (R2 + D1) — abandoned

The hosted target was a Cloudflare Workers deploy (graph in D1, blobs in
R2). It was dropped: **ingest latency on Workers/R2/D1 was far too
high.** Workers caps subrequests per invocation (50 Free / 1000 Paid),
and ingest does ~3 subrequests per object (`blobStore.has` +
`blobStore.put` + `graphDb.batch`); even a 0.16 MiB astropy bundle 422'd
on Free, and there was no way to ingest a scipy-sized bundle in one
invocation without queue-based chunking. The wall-clock cost of R2/D1
round-trips made it impractical. Hosting is now a long-running Node
process on a VPS, where ingest is a single in-process transaction.

The storage **abstractions are kept** (`BlobStore` / `GraphDb` /
`RawStore`) so a different backend can be slotted in later without
touching the viewer — but only the filesystem + SQLite implementations
exist today, and there is no Cloudflare adapter, `wrangler.toml`, or
`build:cf` target anymore.

## Open questions

- Static export: **un-parked and implemented** (2026-09), whole-store
  rather than the single-project mode the parked note imagined.
  `PAPYRI_STATIC=1 pnpm build:static` prerenders every public page of every
  ingested bundle to `dist/client/`. The SSR server remains the deployment
  target; the snapshot is a second consumer of the same templates, not a
  fork of them. Remaining questions are listed under Milestones below.
- Encoding convergence: if everything moves to a single encoding (CBOR or
  JSON), `ir-reader` gets simpler. Until then it handles both.
- IR-drift policy: pin a "known-good" IR version, or accept best-effort
  rendering and let components no-op on unknown nodes? Probably the latter
  while the IR is still evolving.

## Ground rules

- No Python code here. Everything reads the IR produced by the top-level
  `papyri` package.
- No changes to the IR format from inside `viewer/`. If the viewer needs a
  field the IR doesn't expose, raise it against the top-level plan first.
- Keep dependencies tight. Every new runtime dep needs a one-line
  justification in the PR.
