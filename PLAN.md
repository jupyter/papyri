# Papyri plan

Source of truth for scope and ordered work. Future sessions (human or agent)
should keep it current: delete finished items keep **Open work** actionable, and update **Open questions** as
answers arrive. If this file contradicts `CLAUDE.md`, this file wins on scope.

## Why this project exists

Two specific problems in the Python documentation ecosystem:

**Problem 1 — Sphinx couples building and rendering.**
Updating an HTML template (e.g. for accessibility) requires a full rebuild
from source. Papyri separates IR *generation* (run once by the maintainer)
from *rendering* (stateless, redoable against the saved IR).

**Problem 2 — Documentation is fragmented across domains.**
Every library lives on its own subdomain with no real cross-linking.
Papyri's model (conda-forge style): maintainers publish DocBundles; a single
service ingests many and serves them from one place.

## Target shape

- **`papyri gen`**: run per project, by each library maintainer in their own
  CI or build environment. Produces a self-contained DocBundle on disk. The
  DocBundle at this stage is intentionally left as JSON and lenient on
  errors/completeness, so other tools can operate on it for flexibility.
- **`papyri pack`**: packs the DocBundle into the final IR artifact for
  upload. This packed form is what should be standardized and exchangeable;
  it must be linted and contain no errors.
- **`papyri upload`**: ships the bundle to a viewer instance's
  `/api/bundle` endpoint, which runs the TypeScript ingest pipeline
  server-side to wire bundles into the cross-linked graph.
- **`viewer/`**: TypeScript web renderer. Works locally for development and
  is built with the centralized service in mind — the intended rendering
  frontend for the hosted service, not just a local debug tool. Deployed as a
  long-running Node.js server on a VPS. Milestone tracker: [`viewer/PLAN.md`](viewer/PLAN.md).
- **`ingest/`**: TypeScript `papyri-ingest` package — the canonical
  ingestion engine, invoked by the viewer's upload endpoint.

`papyri gen` is the *reference* producer, not the only intended one. The
bundle format (schema + the invariants below) is the ecosystem contract;
other producers may emerge — e.g. working from Markdown/MyST sources — and
anything that emits a valid, linted bundle is a first-class citizen. Ingest
validates and may reject (see invariants); it does not care who produced
the bundle.

The viewer lives in-tree while the IR is still in flux; co-locating producer
and consumer lets us iterate across breaking changes in one PR. Splitting into
a separate repo remains an option once the IR schema stabilizes.

The boundary between the two halves:

- `~/.papyri/data/<pkg>_<ver>/` — per-bundle IR (`papyri.json`, `toc.json`,
  `module/`, `docs/`, `examples/`, `assets/`). The per-file encoding is an
  implementation detail; do not assume JSON or CBOR exclusively.
- Storage is abstracted: the viewer and ingest pipeline must not assume a
  specific on-disk layout or wire encoding. The current implementation uses
  SQLite for the cross-link graph (`SqliteGraphDb`) and a filesystem store for
  blobs (`FsBlobStore` / `FsRawStore`). The `BlobStore` /
  `GraphDb` / `RawStore` interfaces exist kept so a backend swap stays possible,
  but this is the only implementations today.

## Invariants

These must hold; treat a change that breaks one as a bug, not a trade-off.

**Storage: the graphstore is a derived cache.**
The DocBundle (`.papyri.gz` artifact produced by `papyri gen` and `papyri pack`, archived
verbatim at `_raw/<pkg>/<ver>.papyri.gz`) is the **only** authoritative IR.
Everything in the viewer's graphstore + blob store is a derived projection,
rebuildable via `POST /api/reingest`. Consequence: **what ingest writes into
the graphstore is not required to be the IR** — it may denormalize, precompute,
resolve refs into concrete keys, drop fields the renderer doesn't consume, or
split a node across tables. The only contracts are (1) the raw archive stays
byte-identical to the upload, and (2) the renderer's input shape stays stable
or migrates in lockstep with the renderer. Do not try to preserve
round-trip-to-IR fidelity in the store; the round-trip is via re-ingest from
the raw archive.

**Gen owns all ref classification; ingest only links. Ingest can _fail_ if it disagree and believe the uploaded bundle is incorrect.**
Applies to the IR in the raw archive (not the graphstore's internal form).
- `LocalRef` means *this bundle*, always. Gen converts every relative ref,
  alias, and local name to a `LocalRef` before writing the IR.
- Every cross-bundle reference is a `RefInfo(package, version, kind, path)`.
  The *name* half (`package`, `kind`, `path`) must be fully resolved — no
  fuzzy strings, no unresolved aliases. The *version* field is late-bound:
  `"?"` (resolve to the best available version at serve time) is the expected
  value for the overwhelming majority of refs. An explicit version is an
  opt-in **pin** for the rare doc that genuinely targets one release;
  version-exact refs are *not* a goal, and gen must not bind refs to
  whatever version happens to be installed in the build environment.
- Ingest resolves `LocalRef`s to full keys and records live or dangling
  `RefInfo` links and an optimisation. A two-step ingest (build a ref map first) is an
  optimisation, not a correctness requirement.

**No raw HTML in the IR.**
The IR is semantic: producers express content as IR nodes, never as embedded
HTML/CSS islands. Raw HTML breaks every non-HTML consumer (terminal /
Jupyter rendering), cross-linking, and theming — it is one of the two
failure modes that made building on docutils/MyST output intractable the
first time (the other: links resolved too early; see the ref-classification
invariant). Directives that only produce HTML get unwrapped, dropped by
explicit policy, or handled by a registered IR-producing handler — never
passed through as markup, and never smuggled in as raw directives either
(see the directive invariant below). This applies to *any* producer, not
just `papyri gen`.

**RST substitutions never reach the IR.**
The IR must never contain `SubstitutionDef` or `SubstitutionRef` nodes.
Non-`replace::` substitution types (image, unicode) are warned and dropped;
support can be added per demand.

**No directive reaches the packed IR; none is silently discarded.**
Directives are a source-format construct (an RST-ism — MyST has its own):
they must not leak into the packed artifact, for the same reason
Python-isms must not leak into the wire encoding. Target state: gen keeps
unhandled directives verbatim in the *lenient* bundle directory
(`Directive.from_unprocessed` — inspectable, available to tooling), and
`papyri pack` / `papyri lint` fail while any `Directive` node remains.
(Current code differs, and is *worse* than previously recorded here:
`Directive` is an `UnserializableNode` with `_reject_at_validate`, so
`doc_blob.validate()` raises at gen time — but under the default
`--fail-early=False` that exception lands in the per-object catch and the
whole host object is **silently dropped** from the bundle; only
`--fail-early` produces the hard failure. 2026-07 audit, finding N3; see
the enforcement item below.)
Every directive must be explicitly handled by pack time: by a built-in handler,
a project-registered handler (see the `DirectiveContext` plugin API), or an
explicit maintainer decision to unwrap or drop (via config). Dropping is
legitimate when *chosen* — never as a silent default. The "not silently
discarded" guarantee is enforced at the strict boundary (pack/lint), not by
preserving raw directives in the artifact.

**Encoding boundary.**
The bundle directory (`papyri gen` output) is JSON — intentionally
human-readable, inspectable with `papyri debug`. CBOR starts at `papyri pack`
and is the only encoding in the `.papyri` artifact and the ingest/viewer
layers. Do not write CBOR into the bundle directory or JSON into the artifact.

## Python version

- Minimum: **Python 3.13**. `requires-python = ">=3.13"`.
- CI matrix: `3.14` only. Add newer versions later; don't carry legacy ones.

## Dependency notes

- RST parsing uses `py-tree-sitter-rst` (PyPI) on top of `tree-sitter >= 0.24`.
  Do not reintroduce `tree_sitter_languages` or `tree-sitter-language-pack`.
- Markdown parsing uses `tree-sitter-markdown` (PyPI) on the same
  `tree-sitter` runtime. It ships *two* grammars — `language()` (block) and
  `inline_language()` — and the inline one emits no text nodes, so plain
  prose is the byte gap between named children. `papyri/ts_markdown.py`
  handles both facts; see its module docstring.
- `numpy`, `scipy`, `astropy`, `IPython` in the CI matrix drift frequently;
  pin each matrix entry to a known-good version or xfail with a reason.
- TypeScript is split across two packages (Microsoft's documented TS 7
  migration pattern). `ingest/` compiles and type-checks with the native
  Go-based TypeScript 7 (`typescript7` npm alias → `typescript@7`, provides
  the `tsc` bin). The `typescript` dependency name in both `ingest/` and
  `viewer/` is an alias for `@typescript/typescript6`, which re-exports the
  TS 6 JS compiler API (bin: `tsc6`) — needed because `typescript-eslint`
  and `@astrojs/check` consume the old JS API, which `typescript@7` no
  longer ships. *Follow-up:* when typescript-eslint and Astro support the
  stable native API (expected ~TS 7.1), drop the `@typescript/typescript6`
  alias and move `viewer/` type-checking to TS 7 as well.

---

## Open questions (need a decision before the work is scoped)

- **PyPI republish?** Re-publish under a new version, or keep "install from
  git" only for the foreseeable future? Still open.
- **Pinned-ref semantics at serve time.** When a `RefInfo` pins a version the
  store doesn't hold: hard dangling link, or fall back to the best available
  version with a "pinned to X, showing Y" indicator? (Lean fallback +
  indicator — a pin expresses authorial intent, not a guarantee about what a
  given service instance holds.) Related: does the pin stay an exact version,
  or eventually admit a PEP 440 specifier ("changed in numpy 2.0" is `>=2.0`,
  not `==2.0.1`)? *(2026-07: explicitly deferred until the pin path is
  implemented. Note: no pin pathway exists anywhere in gen today — no role
  syntax, directive option, or config knob — deferred by design; recorded
  so the invariant isn't read as implying one exists.)*
- **Second-producer experiment (MyST- or docutils-based).** An earlier
  attempt to build papyri on top of docutils/MyST failed on two things:
  links resolved too early, and content collapsing into HTML. Both are now
  explicit producer requirements (the ref-classification and no-raw-HTML
  invariants), and the coming schema gives an external producer something
  concrete to target — so a retry becomes viable *as an alternative
  producer* (a Sphinx builder or mystmd plugin emitting bundles for
  Markdown/MyST-source projects), not as papyri's base. Worth attempting
  once the schema exists; the IR → MyST exporter provides the round-trip
  check.
- **Terminal / Jupyter client architecture.** Terminal + JupyterLab rendering
  is deferred, not dead, and it is on the critical path of the IPython
  adoption wedge (`?` showing rich cross-linked docs is the demo nobody else
  can match). When it comes back: thin client of the hosted service's JSON
  API, or reader of a local store? Thin-client avoids reimplementing ingest
  in Python but promotes the viewer's JSON endpoints to a public contract —
  design them accordingly either way. *(2026-07 lean: deferred; offline is
  desirable, but relying on the central service is acceptable if it's
  solid — decide when the work starts.)*
- **Future of `papyri find` / `describe` / `diff` / `debug`.** They work
  against the store the TypeScript pipeline writes, but the viewer is the
  user-facing replacement. Keep, trim, or drop?
- **Per-bundle crossref tables instead of one global `links` table?**
  Today every crossref lives in a single `links(source, dest)` table over a
  global `nodes` table (`ingest/migrations/0000_init.sql`); `getBackrefs`
  (`viewer/src/lib/graph.ts`) joins `links` + two copies of `nodes` filtered by
  `(package, identifier)`. Alternative: a per-bundle table of *outgoing* refs
  (partitioned by bundle → re-ingest/eviction is a drop+rebuild) plus a coarse
  bundle→bundle index, with backrefs computed lazily at view time.
  Trade-offs to weigh before committing:
  - *Cost model.* Global table = one indexed query. Per-bundle =
    `O(N_bundles_pointing_at_pkg)` queries / a `UNION ALL`. For a hot package
    (numpy) the fan-out could exceed the single join, though each per-bundle
    table is small so total rows scanned may still be lower.
  - *Write contention.* Per-bundle tables are append-only during one ingest;
    the global `links` table is write-amplified across concurrent uploads.
  - *Wildcard-version stubs.* `getBackrefs` matches `version = '?'` for
    unresolved cross-package targets; the per-bundle scheme must preserve that
    (store the wildcard, or resolve lazily).
  - *Shape.* Prefer one `refs` table partitioned by `(from_pkg, from_ver)` with
    a covering index over literal `refs_<pkg>_<ver>` tables (DDL churn).
  - Pure denormalization — no IR/raw-archive impact, exactly what the storage
    invariant permits. Revisit when backref latency or write contention is a
    *measured* problem.

## Open work — Gen (Python)

- **Wire markdown narrative sources into `papyri gen`.** `papyri/ts_markdown.py`
  parses CommonMark into the same `list[Section]` IR that `papyri/ts.py`
  produces from RST, but nothing calls it yet. To finish the feature:
  `_scan_narrative_sources` / `collect_narrative_docs` (`papyri/gen.py`) glob
  `**/*.rst` only, and need a suffix dispatch plus a markdown counterpart to
  `_extract_rst_targets` — markdown has no `.. _label:`, so cross-doc anchors
  have to come from heading slugs, and the doc-key derivation (`[:-4]` for
  `.rst`) is suffix-specific. Deliberately kept out of the parser PR: the
  anchor scheme is a design decision, not plumbing.
  Scope note: this covers *limited markdown inclusions* (READMEs,
  changelogs, short narrative pages). Directives have no markdown spelling —
  MyST's `:::{note}` belongs to the second-producer experiment below, not
  here. Raw HTML is dropped with a warning per the no-raw-HTML invariant,
  which is the main fidelity loss on real-world READMEs.

- **Typed qa forms (NewType) instead of ambiguous strings.** At least three
  string shapes travel through gen/tree under the name "qa" and are told
  apart only by convention: dotted module paths (`numpy.ma.core`), full_qual
  object form (`numpy.ma.core:MaskedArray.var` — colon splits module from
  qualname; `utils.FullQual` exists but most signatures take plain `str`),
  and narrative doc keys (`reference:ufuncs` — colon is a *directory*
  separator). The ambiguity has already produced real bugs fixed by ad-hoc
  guards: `resolve_` normalizes `":" → "."` before scope derivation,
  `DirectiveVisiter._resolve` detects "qa is not a Python path under the
  module" to fall back to the package root, and `:doc:` resolution needed an
  is-API-context test to stop deriving phantom doc keys from object qa.
  Introduce distinct NewTypes (or tiny dataclasses) — e.g. `ModulePath`,
  `FullQual` (reuse), `DocKey` — annotate the qa-carrying signatures
  (`resolve_`, `DirectiveVisiter`/`GenVisitor` constructors, diagnostics
  targets, doc-target maps), convert at the boundaries where each form is
  created, and let mypy reject cross-form mixing so those guards become
  type errors instead of runtime heuristics.
- **Make the gen pass order irrelevant.** `gen()` runs API → examples →
  narrative, and cross-pass linking only works in that direction because
  each pass hands the *next* one what it collected (`_known_refs`, the
  narrative label maps from `_scan_narrative_sources`); the sweep branch
  had to swap examples after API docs just so example pages could link
  to API objects. In the long run any pass may reference any other (an
  API docstring pointing at an example, an example at a narrative
  label), so order must not matter. Investigate a two-phase gen: a cheap
  first phase that only *registers targets* for every kind (API qualnames,
  example names, narrative doc keys + labels), then a visiting phase that
  resolves against the complete target set — the `DelayedResolver` idea
  already in `tree.py` (currently dead machinery) is the late-binding
  alternative if a strict two-phase split proves awkward.
- **Role handlers as configured, stateful objects (deferred).** Today a
  `[global.roles]` handler is a bare callable `(value) -> nodes`, with
  per-bundle state smuggled in ad hoc (the visitor `partial`-binds `role=`
  and `warn=` onto `role_unset`; `:ghpull:` reads the visitor's slug).
  Mirror the direction chosen for directives: a role handler could be a
  class constructed once from its config options (init args/kwargs, like
  `[global.directives]` table entries already allow) carrying its own
  state, with a method for the actual dispatch `(value, ctx)` and
  state-mutating hooks the visitor calls as it moves — typically "now
  entering page/object X" — so handlers that need the current location
  (link builders, per-page counters) get it without global lookups. Defer
  until a handler actually needs it; `role_unset`/`role_verbatim`/
  `role_text`/`role_drop` are fine as plain functions.
  Two design preferences to carry into both the role and directive
  plugin APIs when they get built:
  - *One context object, not a growing kwargs list.* Handlers should
    receive a single object exposing attributes (`ctx.qa`, `ctx.doc_root`,
    `ctx.warn`, …) rather than `handler(argument, options, content,
    doc_path=…, doc_root=…, asset_store=…, warn=…)`. Beyond keeping
    signatures stable as fields are added, an attribute-access object can
    be instrumented (a recording proxy) so we can later *measure which
    handler reads which fields* — useful for pruning the context, for
    documenting each handler's real dependencies, and for spotting
    handlers that reach for state they shouldn't.
  - *Maybe let handlers ask for what they need, lazily.* Instead of the
    visitor eagerly computing every context field for every call, a
    handler could `yield` requests for the information it needs (a
    generator/coroutine-style protocol: yield `Need("doc_titles")`, get
    the value sent back, continue) so expensive fields (narrative title
    maps, asset stores, execution namespaces) are built only when some
    handler actually asks. Exploratory — weigh it against the plainer
    attribute-object once there is a real handler with an expensive
    dependency.
- **`DirectiveContext` injection (context, not globals).** Directive handlers
  reach bundle state through ad-hoc closures (`make_image_handler` &c.) and,
  historically, module globals. Define an explicit `DirectiveContext` (at least
  `doc_path`, `asset_store`, `module`, `version`, active `Config`, and the
  `Diagnostics` collector) and pass it as a second argument to every handler:
  `handler(argument, options, content, ctx)`. Handlers that need nothing ignore
  it. This subsumes the remaining "handlers should not read global state" work
  (the `:ghpull:`/`:ghissue:` half is already done) and would let the
  factory-bound `warn` callbacks (malformed-directive diagnostics) become plain
  `ctx.warn`. Config-supplied handlers (`obj_from_qualname`) make the signature
  change the delicate part. **End goal (decided 2026-07): this is the
  foundation of a public registration API** — projects register their own
  IR-producing handlers for their custom directives (via `papyri.toml` /
  entry points): Sphinx's ahead-of-time registration model with the
  HTML-output target fixed. `_SPHINX_ONLY_DIRECTIVES` and the built-in
  handler set are a stopgap until third-party registration exists.
  2026-07 review specifics: real closure usage shows ctx needs, beyond the
  minimum list above, `doc_root` (image/figure/include), `qa` (plot + every
  warn emission), the `execute` flag (plot), and the invoked directive's
  *name* (fixes the old TODO in `directives.py`; lets one handler serve
  several names). Config-registered handlers currently receive *zero*
  bundle state — not even the `warn` callback built-ins get via `partial` —
  so pass ctx at the single dispatch site in `replace_UnprocessedDirective`
  rather than partial-binding at registration. Add error isolation at
  dispatch: a failing user handler should emit a coded diagnostic
  (`E-directive-handler-failed`) and fall back to
  `Directive.from_unprocessed`, not abort gen (which `early_error=True`
  does today). Resolve and validate the handler registry once per gen run —
  today `obj_from_qualname` re-imports/re-instantiates per documented
  object and never checks callability — which is also where entry-point
  registered handlers merge in later. Move the module globals onto ctx:
  `_plot_counter` (non-reproducible `fig-plot-N.png` asset names across
  runs in one process) and `_MISSING_DIRECTIVES`.
- **`ts.py` diagnostics wiring.** The unparseable interpreted-text / hyperlink
  fallbacks in `ts.py` still `log.warning` plainly. Blocked on a design
  wrinkle: `ts.parse()` is `@functools.lru_cache`'d, so diagnostics emitted
  during parsing fire only on a cache *miss*, and `parse()` has no handle to the
  Gen's `Diagnostics`. Correct fix: have the cached parse return its warnings
  alongside the nodes so `parse()` re-emits them on every call — a real refactor
  of the TS visitor. The 2026-07 review upgraded this from cleanup to
  **correctness bug**: the cache returns shared *mutable* node trees with no
  defensive copy, and `TreeReplacer.generic_visit` (`tree.py`) plus the
  include handler (`_resolve_nested_includes`, `directives.py`) mutate
  `children` in place — two documents with byte-identical source text share
  one tree, and the second sees the first's visited state. Also
  `_parse_cached` constructs `TSVisitor(text, "")`, so qa context is lost
  even on cache *misses* and parse warnings always print `in ()`. Extended
  fix shape: the cached function returns an immutable `(sections, warnings)`
  payload; `parse(text, qa, warn=…)` deep-copies (or rebuilds) the tree
  before handing it to mutating visitors and re-emits each warning with qa
  attached, routing to Diagnostics when a warn callback is passed.
  2026-07 audit addition (N8): `TSVisitor.visit_ERROR` returns `[]` with
  **no signal at any log level**, silently deleting the text under any
  tree-sitter ERROR node; the Text-preserving fallback branch its code
  comment describes is unreachable because `getattr` dispatch finds
  `visit_ERROR` first. Fold into this refactor: make ERROR nodes the
  Text-preserving fallback plus a queued warning.
- **Per-reference version pins.** `"?"` is the expected version on almost all
  refs (see the ref-classification invariant); what's missing is the opt-in
  path for a doc to *pin* a specific version when it means one, plus an
  enforcement point that pins are well-formed once cross-package version data
  is threaded through.
- **Enforce the directive invariant.** Four parts, sharpened by the
  2026-07 gen conformance review.
  (a) Invert the strictness boundary first: make `Directive` a normal
  registered, JSON-serializable staging node (drop `_reject_at_validate`
  and the fail-fast docstring in `nodes.py`), then add a leftover-Directive
  scan to `lint_bundle`'s node loop and run it on the pack path — today the
  check is unimplementable because a `Directive` can never be read back
  from disk. A check on bundle *content*, not gen-time error records (the
  reverted gate in the done log gated on stale records; this doesn't).
  (b) Config: teach the `[global.directives]` value parser (the
  `obj_from_qualname` loop in `tree.py`) to accept literal `"drop"` /
  `"unwrap"` — the unwrap primitive already exists (`container_handler`,
  `directives.py`); `"drop"` maps to a diagnostics-emitting drop handler.
  (c) Triage the built-in defaults in `_SPHINX_ONLY_DIRECTIVES`, which
  conflates three cases: truly meta → drop (`highlight`, `currentmodule`,
  `testsetup`/`testcleanup`, the `auto*` family); layout containers →
  unwrap via `container_handler` (`grid*`, `card*`, `tab-set`/`tab-item`,
  `dropdown`, `button-*`) — tabs and dropdowns routinely hold unique prose
  (per-OS install instructions are the classic); handwritten py-domain
  directives (`py:function` &c. and the bare `function`/`class`/… forms)
  carry API documentation that exists nowhere else in the bundle — at
  minimum unwrap (argument as signature line + parsed body), eventually
  real handling. `testcode`/`testoutput` render as visible code blocks in
  Sphinx → map to the existing `code_handler`, not drop (the set's comment
  currently asserts the opposite).
  (d) No drop is silent: register a `W-dropped-directive` diagnostic and
  emit it for every `_SPHINX_ONLY_DIRECTIVES` hit (today: bare `log.info`),
  and give `raw_handler`/`only_handler` the same `warn=` binding as the
  other free-function handlers so their drops reach Diagnostics too.
- **See-also refs ship placeholder RefInfo into the packed IR.** `doc.py`
  emits `RefInfo("current-module", "current-version", "to-resolve", name)`
  and gen replaces it only for same-bundle targets; every cross-package
  see-also keeps the fake literals through pack, and the viewer
  special-cases them (`xref.ts`). The `CrossRef` docstring also promises an
  "ingest relink pass" that does not exist anywhere in `ingest/src`. Fix:
  classify in gen (emit `RefInfo(pkg, "?", "module", path)` via the import
  solver; a well-defined missing form otherwise), have pack/lint reject
  `kind="to-resolve"`, rewrite the docstring, then delete the viewer
  special-cases. Clearest current violation of "no fuzzy strings".
- **`resolve_` emits `RefInfo(None, None, …)`; ingest repairs it.** The
  "missing"/"local" branches return None module/version; the "local" ones
  reach the IR (module=None never matches the LocalRef conversion) and TS
  ingest papers over gen's Nones with `?? "?"` (`visitor.ts`). Gen should
  emit the canonical forms itself (LocalRef for same-bundle, a defined
  missing shape otherwise) so ingest can *fail* on malformed refs instead
  of fixing them, per the invariant. Related doc rot: `pack.py` cites
  `Gen._relink_dangling_local_refs`, which doesn't exist, and `LocalRef`'s
  "guaranteed to exist" docstring is not upheld by any gen-side pass.
- **Figure/asset refs stamp the build-environment version.** Four sites
  (`gen.py` ×2, `directives.py` ×2) emit
  `RefInfo(module, <concrete version>, "assets", name)` for assets that are
  same-bundle by construction, making the bundle digest depend on its own
  version number — exactly what `_ref_to_crossref` avoids for other
  intra-bundle refs. Change `Figure.value` to accept `LocalRef`, emit
  `LocalRef("assets", name)` at all four sites, update the Figure check in
  `pack.py`, and delete the stale "todo: add version number here" comment
  in `tree.py`.
- **Inline roles in csv-table cells.** `csv-table` cells become plain
  `Text` — role markup inside cells (`:kbd:` in docs-tree csv files)
  degrades to inert literal text. Upholds no-raw-HTML but reads noisily;
  run cell content through the inline parser once the `DirectiveContext`
  work lands. (2026-07 audit: the `list-table` half of this item was
  refuted — list-table cell content goes through `parse()` and roles
  resolve normally.)
- **Raw-markup passthroughs.** Three places copy unparsed RST source into
  content nodes, against the no-raw-markup direction: `autosummary` renders
  its own directive markup as a visible `Code` block
  (`_block_verbatim_helper` — drop it like the rest of the `auto*` family,
  or give it a real LocalRef-list handler, then delete the helper);
  grid/simple RST tables become verbatim-source `Code` blocks (`ts.py` —
  parse into the existing table nodes, or at minimum emit a coded
  diagnostic so the degradation is tracked); `|x| replace::` substitution
  bodies are spliced in as raw source `Text` (roles like
  ``:class:`numpy.ndarray``` inside a substitution bypass inline parsing
  and ref classification — run them through the inline parser).
- **Builtin ref resolution at gen time.** Ship a Python-builtins bundle shim (a
  minimal DocBundle registering every builtin as a `RefInfo`); `papyri gen`
  emits builtin refs as ordinary cross-refs and ingest resolves them against the
  shim like any package — no special-casing in the resolver. (The intersphinx
  inventory already covers stdlib links via CPython's `objects.inv`; the shim is
  the gen-time alternative for builtins specifically.)
- **Remaining pack/lint unification.** The core of the 2026-07 review
  finding is closed: `make_artifact_from_dir` runs the lint checks
  (substitution nodes and `DocstringSentinel` placeholders always fail;
  missing Figure assets, leftover `InlineRole` residue, and
  `Unimplemented` placeholder counts warn by default, fail under
  `--strict`). Still open from that finding: share `_assert_safe_urls`
  with `papyri lint` (today it runs only on the pack path); give `papyri
  lint` a `--strict` flag and fix its help text (it advertises a
  dangling-LocalRef check that non-strict `read_bundle_dir` only warns
  about); and a heuristic raw-HTML scan over string leaves (warn;
  `--strict` error) so the no-raw-HTML invariant is enforced at the
  boundary for *any* producer, not only by gen's handler table.
- **Route every skipped object/page through Diagnostics (2026-07 audit
  N3).** Per-object exceptions in `collect_api_docs` (ErrorCollector
  unexpected errors, the post-processing catch) and per-page skips in
  `collect_narrative_docs` discard the object/page with only a
  `log.warning` — no coded diagnostic, so `error_on_warning` never gates
  them and gen exits 0 with content missing. Verified triggers: any
  admonition with a `:class:` option (`assert not options` in
  `admonition_helper`), and an unhandled directive under the default
  `--fail-early=False` (see the corrected directive-invariant note).
  Add `E-object-dropped` / `E-page-dropped` codes (error default),
  emit at every skip site, and fix `admonition_helper` to tolerate
  options via `warn`.
- **Toctree/narrative silent-skip chain (2026-07 audit N5).** A narrative
  page that fails to parse vanishes together with its navigation entry on
  console noise alone: gen skips it (`log.warning`), `toc.py` then drops
  the dangling toctree entry with a bare `print("skip Path", …)` — so
  pack's hard `_check_toc_refs` never sees the breakage. `:glob:` toctree
  entries are skipped un-expanded with no signal at all, and
  absolute-path (`/…`) entries are dropped with a `print`. Convert the
  `print`s to diagnostics, add `W-narrative-page-skipped` (error default)
  and `W-toctree-entry-dropped`.
- **Placeholder English prose ships in packed artifacts (2026-07 audit
  N6).** "No Docstrings", "This module has no documentation", and
  `<No Title …>` toc titles are magic strings injected by gen with no
  diagnostic and no lint shape-check — verified present in real packed
  bundles. Replace with structured representations (empty summary +
  flag; doc-key fallback titles), emit a diagnostic per undocumented
  object / untitled doc, and lint the known placeholder shapes during
  the transition.
- **`papyri upload` packs lenient by default (2026-07 audit N10).** The
  publish path calls `make_artifact_from_dir(strict=False)`: dangling
  LocalRefs, orphan docs, and the warn-tier lint issues ship to the
  viewer un-gated, against "the packed form must be linted and contain
  no errors". Decide: default `--strict` on upload (with an explicit
  opt-out), or at minimum a flag + loud nudge.
- **Audit minors (2026-07, N-misc).** `literalinclude` drops file content
  at `log.info` (not the `warn` binding); `rubric`/`topic`/`container`/
  figure-caption silently discard bodies that don't parse as a leading
  Section; the `:external+inv:` intersphinx prefix is parsed into
  `InlineRole.inventory` then ignored — an explicitly-external ref is
  resolved against the local bundle, overriding author intent;
  `jedi_failure_mode="log"` replaces an example's token stream with
  literal "jedi failed" tokens (and `W-doctest-syntax` injects
  "fail"/"fail" tokens) that pass pack unchecked;
  `TextSignatureParsingFailed` → `pass` ships an object without a
  signature silently; `DelayedResolver.add_reference` is dead machinery
  (never called) and its module-global duplicate-target assert can turn
  into a silent page skip. Fold each into the nearest structural item
  (DirectiveContext, ts.py wiring, diagnostics routing) as they land.
- **Upstream the numpydoc leniencies, then delete them.** The See Also
  backtick normalization and section-heading normalization in
  `numpydoc_compat.py` are recorded-and-diagnosed rewrites
  (`W-see-also-syntax` / `W-section-heading-normalized`), but the real
  fix is upstream: either teach numpydoc the backticked See Also form
  (send the patch, leave a pointer here), or fix the source projects'
  docstrings (numpy.polynomial et al.). Once either lands, delete the
  corresponding rewrite.
- **Inline images in phrasing content (image substitutions).** matplotlib's
  docstrings define `image`-type substitutions (`.. |m30| image:: …` used in
  marker/mathtext tables); gen warns and drops them because `Image` is
  FlowContent only — a `SubstitutionRef` inside a `Paragraph` has no legal
  replacement. Supporting them means admitting `Image` into
  `StaticPhrasingContent` (nodes.py union + ir-types/ir-schema + renderer) and
  routing the substitution body through the image handler. IR schema change —
  batch with the next schema-touching PR.
- **numpydoc section fragments that tree-sitter cannot re-parse (#361).**
  `numpy.ma.core:MaskedArray.resize` stays excluded: the full docstring parses
  fine, but the numpydoc-section fragment gen re-parses trips
  TreeSitterParseError. Fix is in how gen splits/re-parses section content.
- **`:orphan:` flag in the IR.** Orphan-doc detection currently only *warns*
  because the IR can't tell an intentionally-unlisted page from an accidental
  one. Once gen reads the Sphinx field-list `:orphan:` metadata, promote
  accidental orphans to a hard `pack` error and exclude flagged ones. Then
  decide whether canonical-`index`-root vs. any-root reachability matters.
- **Rewrite `docs/IR.md` — it documents the wrong encoding.** It claims
  CBOR `module/<qualname>.cbor` blobs, a `tree`/`titles` toc shape, and
  that JS consumers of the bundle dir need a CBOR library; gen writes
  all-JSON (`module/<qa>.json` — pack *requires* the suffix) and a
  list-shaped `toc.json`. CLAUDE.md and PLAN.md match the code; IR.md is
  the stale document. Also fix its dead pointers (`DocBundler.write` is at
  ~1220 not ~1540; `GeneratedDoc` lives in `doc.py`, not `gen.py`).
- **Typed manifest struct through pack.** `papyri.json` stays JSON, but the
  manifest is read into `Bundle` via a freeform dict (`_read_meta` in
  `pack.py`). Represent it as a typed struct inside `Bundle` so the round-trip
  is fully typed from `pack.py` onward (mirror in `ingest.ts`'s `PapyriMeta`).
- **Assets and example pages are keyed by basename.** `make_image_handler`
  uses `asset_name = img_path.name` (`directives.py`) into the flat
  `Gen.bdata` dict, and `collect_examples` keys pages on `example.name`
  (`gen.py`). Two files with the same basename in different source
  directories silently overwrite each other: one figure shows the wrong
  image, one example page disappears, no diagnostic either way. Content-
  address the asset name (the doctest path already does this — see
  `_figure_name`) and key example pages on the path relative to the
  examples folder. Sorting the source globs (2026-09) made *which* file
  wins deterministic, but the collision is still lossy.
- **Warning-filter leaks in `tokens.parse_script` and `ingested_doc`.**
  `parse_script` sets `warnings.simplefilter("ignore", UserWarning)` and
  only restores it at the end of the function, but returns early on every
  Jedi cache hit — so after the first cached example `UserWarning` is
  suppressed process-wide for the rest of gen, and the restore sets
  `"default"` rather than the previous state. `ingested_doc` does the same
  at *import* time and also calls `logging.basicConfig`, so merely
  importing the read-side module reconfigures logging for the process. Use
  `warnings.catch_warnings()`; move logging setup to the CLI entry point.
- **`error_collector` swallows `SystemExit`.** The bypass test is
  `exc_type in (BaseException, KeyboardInterrupt)` — identity, not
  `issubclass` — so a `SystemExit` raised while importing or inspecting a
  documented object is recorded as an unexpected error and, under the
  default `early_error=False`, swallowed: gen continues and exits 0.
- **`toc._tree` has no cycle guard.** `assert p != current_path` catches
  only direct self-reference; two docs listing each other recurse to
  `RecursionError`, which nothing on the narrative path catches. `counter`
  records visits but is never consulted. Track an in-progress path set.
- **Small dead/incoherent bits (delete on sight).** `gen.normalise_ref` has
  no callers (the done log's claim that it was deleted is wrong) and
  imports arbitrary modules as an `lru_cache` side effect;
  `DFSCollector.prune` is unreachable and would raise if it ever ran;
  `Section.__bool__` returns `len(children) >= 0`, i.e. always `True`;
  `compress_word`'s `Whitespace` branch is unreachable and discards its
  accumulator; `tree.py`'s `hash(local_refs)` is a statement with no
  effect, and the module-global resolver cache is keyed on
  `hash(known_refs)` alone with no equality check and no eviction. Several
  node classes (`Table`, `TableRow`, `TableCell`, `AdmonitionTitle`,
  `Admonition`, `Blockquote`, `Section.title`) use
  `field(default_factory=tuple)` as a default although they are not
  dataclasses — nothing consumes the `Field`, so the "default" is a live
  `dataclasses.Field` object that trips the serializer's isinstance
  assert. Latent only because every call site passes the argument; use
  `= ()`.
- **`ts.py` drops content silently in two places.** A non-alphabetic tail
  folded into an `interpreted_text` node is deleted unless
  `trailing_suffix.isalpha()` (so `` `None`s2 `` loses characters), and the
  `.. warning::` special case raises `ValueError("... has no content")`
  when the directive carries both options and content — `:name:`/`:class:`
  are legal on every admonition, and under the default
  `--fail-early=False` the exception drops the whole host object.

## Open work — IR schema / encoding (cross-cutting)

Decided (2026-07): the IR gets a machine-readable schema as the single
source of truth, replacing "grep for `@register`" plus the hand-maintained
mirrors in `encoder.ts` / `ir-types.ts` — settle this before a third IR
consumer (the future terminal renderer) exists. Direction accepted in
principle; firm up details when implementation starts:

- **Schema.** One JSON Schema document — a fragment per node type,
  discriminated union on a `"type"` field. `ir-types.ts` is generated from
  it (`json-schema-to-typescript` or similar); the Python node classes are
  either generated or conformance-tested against it. CI runs a golden
  corpus of fixture documents that both languages must round-trip.
- **Wire format — what goes is the private tag registry, not necessarily
  CBOR.** RFC 8949 standardizes the envelope; tags 4000–4444 are a private
  vocabulary a consumer can only learn from an out-of-band map. Firm
  decision: string-keyed maps with a `"type"` discriminator
  (`{"type": "Paragraph", …}`), no custom tags — the IR data model becomes
  JSON-isomorphic and the schema validates the decoded tree, so `pack
  --strict` and ingest share one validator and `ir-reader.ts` simplifies
  (resolves the viewer's "encoding convergence" question). Open until
  implementation: JSON bytes vs *tagless* CBOR bytes for the IR files.
  JSON's edge: unzip-and-grep inspectability, zero-dependency consumers.
  CBOR's edge: RFC 8949 §4.2 deterministic encoding gives canonical bytes
  for the content-identity hash (JSON's counterpart is JCS / RFC 8785);
  size is a wash after gzip. Lean JSON; decide when the schema lands.
- **Artifact container.** The `.papyri` artifact becomes a plain container
  (zip or tar.gz): `manifest.json`, per-doc IR files, assets as raw
  members. Raw-member assets beat in-band byte strings whichever IR
  encoding wins: per-asset extraction and caching without decoding the
  whole IR graph, `papyri unpack` becomes near-trivial, and keeping image
  bytes out of the IR payload is what makes the content-identity hash
  ("hash structure + text, not nondeterministic figures") natural.
- **Tuple vs list (tag 4444) — already done by construction.** The 2026-07
  review verified at the byte level that the tag is never emitted:
  `Node.cbor` converts tuple fields to lists before encoding, and decode
  restores tuples from annotations (`_coerce_field`) — exactly the
  schema-driven coercion this bullet asked for. Remaining work is deletion:
  drop `register(4444)(tuple)` (`nodes.py`), the `TUPLE_TAG` branch in
  `ingest/src/encoder.ts`, and the stale tag-4444 row in `docs/IR.md`
  (which wrongly claims the tag round-trips tuples).
- **Boundary invariant rewrite.** When this lands, restate the "Encoding
  boundary" invariant: the gen-dir vs artifact boundary was never really
  JSON-vs-CBOR — it is *lenient staging output* vs *strict, linted,
  schema-validated artifact*. That is the boundary worth enforcing.
- **Node-shape pre-work (do before freezing schema fragments).** 2026-07
  review findings: rename `SeeAlsoItem.type` — a *data* field that hijacks
  the discriminator slot (it serializes as `{"type": null}` today, so the
  node carries no class identity on the wire); replace
  `GeneratedDoc._content` + `_ordered_sections` with a single ordered
  sections sequence (kills `_OrderedDictProxy`, the underscore wire keys,
  the unreachable `| None` arm, and the one map whose key order is
  semantic); fix `node_serializer` to use the ClassVar-filtered
  `get_type_hints` — the `sections` ClassVar constant currently leaks into
  every JSON doc but not into CBOR, so the two encodings disagree on
  `GeneratedDoc`'s field set; collapse `SigParam.annotation`/`default`'s
  three-valued `str | NoneType | Empty` (Python class names leak onto the
  wire as `{"type": "NoneType"}`; the NoneType arm is unreachable from
  gen); delete `UnimplementedInline` (zero remaining producers); exclude
  `UnserializableNode` subclasses (`Directive`, `UnprocessedDirective`)
  from persisted-node unions, which would poison generated schema
  fragments; tighten `FieldListItem`'s annotations to what its `validate()`
  actually enforces.
- **IR → MyST AST export: yes (decided 2026-07).** A one-way exporter is a
  small tree transform once the JSON encoding lands; schedule it after the
  schema exists. It doubles as a conformance tool if a MyST-based producer
  emerges (round-trip testing between exporter and producer).
- The schema is also what makes third-party *producers* possible (see
  Target shape): a bundle is valid because it validates against the schema
  and passes lint — not because `papyri gen` wrote it.

Old raw archives in the CBOR format are re-generated, not migrated
(pre-production rule: no old data matters).

## Open work — Viewer / ingest

- **"0 errors / N warnings" badge per bundle.** Gen records resolved
  diagnostics under `papyri.json`'s `diagnostics` key, but it's a list of dicts
  and `pack._read_meta` only lifts *scalar* manifest keys into `Bundle.extra`,
  so it never reaches the artifact or the viewer. Either add scalar
  `diagnostic_{error,warning}_count` manifest keys (flow through `extra` →
  `meta.cbor`) or carry the full records as a typed `Bundle` field, then render
  the badge on the bundle index/overview. 2026-07 review wrinkles:
  `_read_meta` drops the `diagnostics` list *silently* and stringifies the
  scalars it does lift (`str(v)` — count keys would arrive as `"3"`, so the
  viewer would have to parse), and `_manifest_dict`'s round-trip docstring
  is wrong today (gen-dir → pack → unpack loses the key). Prefer the
  typed-field route; make `_read_meta` log dropped manifest keys either way.
- **Inline class members (methods & attributes).** Per-page/per-bundle toggle
  that expands each class's members inline (full docstrings, signatures, param
  tables) instead of a summary table of links. Reuses the member qualname blobs
  the class page already fetches (`viewer/src/lib/qualname-page.ts`) — no new
  fetching. `?inline-members=1` query flag (shareable) + a React island +
  optional `localStorage` persistence; default collapsed.
- **Inline module-level functions.** Mirror of the above for functions defined
  directly in a module (`?inline-functions=1`). Large modules (numpy) get very
  long — add "collapse all" + per-function anchors. Both toggles should share
  one rendering component.
- **Bundle staging area.** Upload into an isolated staging zone (no backrefs
  computed, atomically droppable) for PR-doc review and RC review. Staged
  bundles never appear in cross-package "Referenced by" lists or the global
  search index.
  - *Design.* Staging is a namespace, not a separate pipeline: same parsing,
    writing into a `_staging/<pkg>/<ver>/` raw zone and `staging_*` tables (or a
    `staging` flag column). Ingest may resolve *outgoing* refs against the main
    graph (so maintainers see live cross-refs) but does not update the main
    graph's backref tables. Drop = single table/row drop, no cascade. Endpoint
    `POST /api/bundle?staging=1`; visually distinct (persistent banner, excluded
    from the default home list). Later: an explicit `…/promote` that moves the
    raw archive to the main zone and re-ingests. Staged bundles skip the
    "latest backrefs only" dedup (they have none). Same upload auth as normal;
    viewing may optionally require login.
  - *Open.* Whether a staged `GET /[pkg]/[ver]/` shows a warning banner
    (probably yes, reuse the version-status banner). Eviction: TTL /
    auto-eviction is required, not optional — see "Adoption / CI
    integration" below (PR-preview load makes staging storage unbounded).
  - *Files.* `ingest/src/ingest.ts` (flag, skip backref writes),
    `viewer/src/pages/api/bundle/[...path].ts`, graph-layer
    `listStagingBundles`/`dropStagingBundle`,
    `viewer/src/pages/[pkg]/[ver]/index.astro` (banner),
    `viewer/src/pages/staging.astro` (admin list).
- **Cross-package Figure/RefInfo version resolution.** TODOs in
  `ingested_doc.py` around version resolution for `Figure`/`RefInfo` across
  packages.
- **Per-bundle → global search.** The manifest is per-bundle; a cross-bundle
  index would enable "find `linspace` across numpy and scipy".
- **Ingest-time precomputation (perf).** Two count queries still run at view
  time: precompute the broken-incoming-refs count into a `bundle_stats` row
  (badge on `/project/[pkg]/[ver]/`), and precompute the latest-linking-version
  backref table (`filterToLatestVersionPerPkg` in `qualname-page.ts`).
- **Promote the shared graph layer into `papyri-ingest` (cross-cutting).**
  `ingest/` and `viewer/` maintain near-copies of graph-layer logic. Once the
  schema stabilises, move the shared bits into the package and have the viewer
  import them rather than re-implement.

- **Ingest has no failure atomicity, and no per-bundle delete.** Three
  linked problems in `ingest/src/ingest.ts`. (a) The `bundles` row that
  marks a bundle as seen is written *last*, after all blob and graph
  writes, so a mid-flight failure leaves nodes and links committed with no
  `bundles` row; the next upload then takes the `freshIngest` path,
  computes `removedRefs = []`, and never deletes the orphans — phantom
  backrefs that no re-upload can clear. (b) Blob writes and graph writes
  are two independent streams (`Promise.all`) and the graph half commits
  in `DB_CHUNK_SIZE` transactions, so `nodes.has_blob = 1` can outlive a
  failed blob write. (c) Ingest is insert-only: an object present in
  version 1.2 and absent from a re-uploaded 1.2 keeps its node row, blob,
  links and backrefs forever (`node_index` is the sole exception). Wanted:
  a per-bundle delete, and a completion marker written first (or one
  transaction). Prerequisite: `Ingester.ingestBundle` currently has **no
  test at all** — the only import of `ingest.ts` anywhere in `ingest/tests`
  is `applyMigrations`.
- **Production opens the graph DB with a different PRAGMA set than
  `openNodeBackends`.** `viewer/src/lib/backends.ts` sets `journal_mode`
  and `synchronous`; the package helper also sets `foreign_keys`,
  `cache_size` and `mmap_size`. SQLite defaults `foreign_keys` to OFF, so
  the `ON DELETE CASCADE` on `links` never fires in the deployment that
  matters. The PRAGMA list should live in `papyri-ingest` and be applied by
  it, not re-typed per caller. (Deliberately not changed in the 2026-09
  pass: turning FKs on may surface latent violations and wants the ingest
  work above alongside it.)
- **`GraphDb` does not abstract the backend.** It is half raw SQL
  (`run`/`get`/`all`/`batch`, with callers hand-writing SQLite-specific
  `ON CONFLICT … excluded.*` and partial-index-dependent queries) and half
  domain methods (`insertNodeIndexRows` / `queryNodeIndex` /
  `deleteNodeIndex`). A second backend would have to reimplement a SQL
  dialect. Either finish the abstraction or, per the pre-production rule,
  delete the interface and use `SqliteGraphDb` directly until a second
  backend is real. Same review: `BlobStore` keys by `Key` while `RawStore`
  keys by positional strings, `BlobStore.clear()`'s doc ("number of
  objects deleted") disagrees with `FsBlobStore`'s implementation (it
  counts top-level directories), and no store defines an error type — six
  call sites sniff `err.code === "ENOENT"` instead.
- **`FsBlobStore.list()` skips `safeJoin`.** Every other method uses it.
  The prefix is built from route params (`ir-reader.ts`, `nav.ts`), so a
  crafted `pkg`/`version` enumerates filenames outside the store root.
- **`FsRawStore.put` writes the authoritative archive non-atomically.** A
  plain `writeFile` over the existing path, plus a second unrelated write
  for the `received_at` sidecar. Per the storage invariant this file is the
  only authoritative IR, so a crash mid-write truncates the one copy
  everything else is supposed to be rebuildable from. Write to a temp name
  in the same directory and `rename`.
- **`_populateNodeIndex` re-walks the whole bundle.** It re-calls
  `generatedDocToIngested` for every doc already converted in phase 2 and
  re-walks each tree with a private `collectNodes` that duplicates
  `visitor.ts`. Fold row collection into the `stage()` loop and export one
  walker. Related: `SqliteGraphDb.batch` re-prepares each statement by SQL
  string with no cache, and `insertNodeIndexRows` sends every row as one
  unbounded transaction while every other write path chunks.
- **`keys.ts`: `keyStr` is ambiguous and `parseKeyStr` is dead.** `keyStr`
  joins four attacker-controlled components with `/`, so a `RefInfo.module`
  containing `/` can collide two distinct refs in the `existingRefs` /
  `forwardRefKeys` dedup sets and drop one. `parseKeyStr` has zero call
  sites and is not re-exported — delete it, and reject `/` in key
  components in `assertBundle`.
- **`node_index.content` stores `JSON.stringify` of a decoded CBOR node.**
  Breaks the "CBOR only, from pack onward" invariant, throws `TypeError` on
  a `BigInt` (cbor-x decodes CBOR integers > 2^53 as BigInt), and mangles
  `Uint8Array`. The throw lands *after* all blob and graph writes have
  committed, i.e. it triggers the orphan mode above.
- **Viewer costs scale badly for a hosted service.** Four items, all
  guest-reachable or always-on: `/api/text-search.json` and
  `/project/*/text-search` walk and CBOR-decode every blob of every ingested
  bundle with only the *hit count* capped, while the structurally identical
  `/api/nodes.json` and `/api/ir-stats.json` are admin-gated for exactly
  that reason; `validate.astro` does an uncapped per-bundle walk plus
  per-ref resolution; `graph.ts`'s `resolveRefs` / `resolveExternalRefs`
  issue one SQL round-trip per reference (its own comment says "naive"), so
  a 200-ref docstring costs 400-600 queries; and `[...slug].astro` emits an
  unconditional `server:defer` island per class member, each rebuilding
  backends and a full xref resolver — PLAN.md specifies this as an opt-in
  `?inline-members=1` toggle, default collapsed, but it ships always-on.
- **`nav.ts`'s `_navCache` is never invalidated.** A process-lifetime `Map`
  with no eviction and no hook from `PUT /api/bundle` or `POST
  /api/reingest`, keyed on the URL version — so `latest` keeps serving the
  nav of whatever version was current when the entry was created, and a
  re-upload does not appear until the server restarts. Contradicts
  "newly uploaded bundles appear without a rebuild".
- **`image-index.ts` builds the asset URL by hand and gets it wrong.**
  ``/assets/${ref.module}/${ref.version}/…`` where the route is
  `/assets/project/<pkg>/<ver>/…`. The `node_index` branch always wins for
  current bundles, so every Figure thumbnail in the image gallery 404s;
  only the pre-migration `walkBundle` fallback produces working links.
  Call `linkForAsset` — this is the duplication `links.ts` exists to
  prevent. Same file and `api/[pkg]/[ver]/nodes.json.ts` interpolate the
  version into a `RegExp` unescaped, so a PEP 440 local version (`0+git…`)
  silently fails to rewrite and a version containing `(` throws.
- **API response and error shapes are inconsistent.**
  `api/bundles.json.ts` and `api/health.json.ts` hand-roll
  `new Response(JSON.stringify(...))` instead of `respond()`, and three
  error shapes are in use across endpoints — `{ ok: false, error }`,
  `{ error }`, and `{ success: false, message }` — so a client cannot
  write one handler. Pick one and route everything through `api-utils.ts`.
- **`qualname-page.ts` duplicates `version-utils.ts`.** `_PRE_RE`,
  `_DEV_RE` and `isPreRelease` are copied with a comment justifying it as
  avoiding "heavier imports"; `version-utils.ts` imports only modules
  `qualname-page.ts` already imports. Import `classifyVersionString`.
  Relatedly `compareVersionsDesc` lives in `ir-reader.ts`, the IR decoder,
  and belongs in `version-utils.ts`.

## Open work — Adoption / CI integration

PR doc previews are the adoption wedge: a project that adds the papyri
GitHub Action gets rendered previews of its own docs on every PR —
single-player value, no other bundles required — and cross-package linking
accrues as projects join for the previews. The build cost (imports, doctest
execution, figure rendering, per-PR) lands on GitHub's free public-repo
minutes; the service pays only for ingest + serve, which the VPS
architecture already makes cheap. This is the structural cost advantage over
central-build services (Read the Docs' largest cost is per-build — notably
per-PR — compute). First adoption target: IPython. Operation/governance:
personal VPS for now; revisit once IPython plus a few packages are live.
Caveat: the free-compute
argument holds for public repos on github.com; private repos and non-GitHub
CI use the token path and pay their own compute.

- **`papyri` GitHub Action.** One copy-pasteable job: install papyri +
  project, `gen`, `pack`, `upload` to a viewer instance. Does not exist yet
  (no `action.yml` anywhere in the repo). The bar is "works on the first try
  in a repo whose tests already pass in CI" — every configuration knob is
  adoption friction. Projects with script-generated doc pages (IPython)
  slot one extra line between gen and pack: an injector script built on
  `papyri.bundle_edit` (see `examples/ipython_inject.py`); the Action
  should make room for such a step.
- **OIDC (trusted-publishing-style) upload auth.** Fork PRs cannot see
  repository secrets, so bearer-token upload silently fails for the most
  common contribution flow, and `pull_request_target` is a known footgun.
  Follow PyPI's trusted-publisher model: `PUT /api/bundle` verifies GitHub's
  OIDC claim (repo, workflow, ref) and maps it to a project via a
  `project → allowed claims` table in the auth DB; per-project tokens stay
  as the non-GitHub fallback. Design this before the token scheme calcifies.
- **Staging eviction is launch-blocking under PR-preview load.** Every push
  to every PR of every enrolled repo uploads a bundle → unbounded storage.
  Needs TTL / auto-eviction, one staging slot per PR (replaced on push,
  dropped on merge/close), and a version naming scheme that can never shadow
  a real release (e.g. `<base-version>+pr<N>.<sha>`). This supersedes the
  "lean explicit delete for v1" note in the staging-area item above.

## Open work — Security / hosting

- **SSRF: intersphinx inventory fetch.** `viewer/src/pages/api/inventory.ts` /
  `ingest/src/inventory.ts` fetch an admin-supplied `inventory_url`/`base_url`
  with no host restriction (`isHttpUrl` permits internal/link-local hosts, e.g.
  `http://169.254.169.254/…`). Admin-gated today (low risk), but a
  metadata-endpoint SSRF vector on the multi-tenant hosted service. Block
  private/link-local ranges and disable redirects to them before hosting.
  2026-09: the IP-literal half is done, the redirect half is not — the
  guard checks only the initial URL and `fetch` still runs with the default
  `redirect: "follow"`, so a 302 to a link-local address is followed and
  parse errors return part of the body in the 422. Use
  `redirect: "manual"` and re-run `isSafeUrl` on each hop.
- **`PAPYRI_AUTH_DB` is not on the persisted volume.** `viewer/Dockerfile`
  sets only `PAPYRI_INGEST_DIR=/data` while `docker-compose.yaml` mounts
  `papyri-data:/data`, so the auth DB falls back to `~/.papyri/auth.db`
  inside the container's writable layer: users, sessions, project
  memberships and minted `papyri_pat_` tokens are lost on every container
  recreate. This is exactly the durability the separate auth DB exists to
  provide. Set `PAPYRI_AUTH_DB=/data/auth.db` (or equivalent) and document
  the upgrade path for existing deployments.
- **Admin endpoints carry no in-handler authorization.** `api/users.ts`,
  `api/projects.ts`, `api/projects/members.ts`, `api/clear.ts`,
  `api/clear-raw.ts`, `api/reingest.ts`, `api/stats.ts`,
  `api/inventory.ts` and the `admin/*.astro` pages all trust
  `middleware.ts`'s `ADMIN_ONLY_PREFIXES` string list, while every
  `api/account/*` handler re-resolves the session itself and documents why
  ("safe regardless of middleware wiring"). One route rename silently
  exposes account creation. Add a `requireAdmin(cookies)` helper in
  `lib/auth.ts` and use it in every admin handler. (`api/stats.ts` also
  still documents itself as "No auth", which is now wrong.)
- **No rate limiting on password login.** `api/auth/login.ts` calls
  `verifyLogin` with no lockout, backoff or failure counter. Argon2id makes
  stuffing slow, but that same cost makes a flood of login POSTs an
  effective CPU-exhaustion vector against a single-process Node server.
- **WebAuthn RP identity falls back to the `Host` header.** `lib/passkey.ts`
  derives `rpID`/`expectedOrigin` from `new URL(request.url).hostname` when
  `PAPYRI_SITE` is unset, which under `@astrojs/node` standalone is
  Host-derived. Impact is bounded (a credential registered under another RP
  ID cannot assert against the real domain), but the RP identity should
  never be attacker-influenced: require `PAPYRI_SITE` whenever passkeys are
  enabled, and note in the env table that it is security-relevant, not just
  cosmetic.
- **Separate domains/processes for upload, admin, and user surfaces.** In a
  hosted deployment, run the upload endpoint, admin panel, and per-user
  management UI as isolated processes on separate subdomains to limit blast
  radius. Design URL structure and routing with this separation in mind so the
  hosted service isn't baked into a monolith. Firm up with the hosting design.
- **Bundle content-identity hash (excluding image bytes).** `papyri upload`
  already skips re-upload when the viewer holds the same SHA-256 over the whole
  `.papyri` artifact (`bundles.content_hash`; `--force` bypasses). That hash
  includes image bytes, so a re-`gen` with churned images triggers a redundant
  (but safe) re-upload. Refinement: a *content-identity* hash over IR structure
  + text + asset *references* but **not** image bytes — autogenerated figures
  (matplotlib, Agg/freetype) are non-deterministic across runs/OSes, so hashing
  them makes every rebuild look changed. Open: keep a separate per-asset hash
  for cache-busting individual images (probably yes, out of the identity hash).
  2026-09 measurement: with the source globs sorted, two consecutive
  `papyri gen examples/papyri.toml --no-infer` runs differ *only* in
  executed-doctest output — matplotlib figure bytes, and `repr()` strings
  carrying memory addresses (`<object object at 0x7f…>`). Filesystem
  ordering is no longer a source of drift, so the identity hash needs to
  exclude executed output as well as image bytes, or gen needs to normalise
  addresses in doctest `out` before storing them.
  - *Follow-up:* make `content_hash` `NOT NULL`. It's nullable only as a
    migration cushion; every write path now computes it. Fold into a future
    migration squash + wipe/re-ingest from the raw archive.

---

## Open work — CLI uniformisation (2026-09 review)

`pack`/`unpack`/`upload`/`lint` share a modern shape (`Annotated`
options, short flags, `error: …` + `typer.Exit(1)`); `gen` and the
graphstore-backed readers do not. Pick `pack.py` as the pattern and
converge:

- **`gen` is the odd command out.** No `Annotated`; `debug`, `dry_run`,
  `api`, `examples` and `narrative` are bare defaults with no help text at
  all; no short flags anywhere; `--no-progress` where the rest use
  `--verbose/-v`; parameter names shadow the builtin `exec` and the sibling
  commands `pack`/`upload`, which is why the body has to re-import them
  under aliases.
- **`gen --fail` is accepted, documented, threaded into `gen_main`, and
  never read.** Delete it (`--fail-early` is the real knob) or wire it.
- **`gen --upload` bypasses the upload resolution chain.** It reads
  `$PAPYRI_UPLOAD_URL` / `$PAPYRI_UPLOAD_TOKEN` itself and passes them as
  *explicit* flags, which outrank everything in `_resolve_upload_params` —
  so a configured `default_target` is silently ignored in favour of
  localhost. Call `upload_func` with neither and add a `--to` passthrough.
  `--pack` likewise hardcodes `strict=False, verbose=False`.
- **Three error-reporting styles.** Clean `error: …` + `typer.Exit(1)`
  (lint, unpack, upload); raw uncaught tracebacks (pack's `_pack_one`,
  gen's `raise SystemExit(str)`, every graphstore-backed command); and
  `sys.exit(str)` (`config_loader`). Exit codes and stderr shape depend on
  which subcommand you happen to run. Also: pack's bulk mode aborts the
  whole loop on the first bad bundle where `upload` accumulates and
  continues.
- **`upload --url` help states the opposite precedence to the code.** Help
  says env var then `--to`; `_resolve_upload_params` evaluates
  `url_flag or target_url or env`, i.e. `--to` wins — which is what the
  docstring and CLAUDE.md say. Fix the help text.
- **The four read commands crash on a fresh machine.**
  `sqlite3.connect` *creates* an empty `papyri.db`; the schema belongs to
  the TS ingester, so the first `store.glob(...)` raises `no such table:
  nodes`. `describe.py`'s careful "Have you run `papyri upload` yet?"
  message is unreachable because the failure happens earlier. Add a
  "store not initialised" check in `GraphStore.__init__`.
- **`GraphStore(link_finder=…)` is dead and `root` is ignored for the DB.**
  `_link_finder` is stored and never read; all four callers pass `{}`. The
  DB path is the module constant `GLOBAL_PATH`, so a different `root`
  yields blobs from one tree and graph rows from another. Delete the
  parameter, derive the DB from `root`, and decide whether the Python
  readers should honour `PAPYRI_INGEST_DIR`/`PAPYRI_INGEST_DB` (today they
  read the wrong store whenever the viewer is configured off-default).
- **`lint`'s advertised LocalRef check never runs.** The help promises
  unresolved-`LocalRef` reporting, but that check lives in
  `_check_local_refs`, which `read_bundle_dir` invokes with `strict=False`
  → `log.warning` only, never in the returned `issues`. `lint` also skips
  `_assert_safe_urls`. Add `--strict` matching `pack -s` and route both
  through it.
- **Three definitions of `~/.papyri/data`** (`cli/pack.py`'s
  `_DEFAULT_DATA_DIR`, `gen.py`'s inline `Path("~/.papyri/data")`, and
  `config.data_dir`, which only `debug.py` imports), and two of the user
  config path (`config.user_config_path` is entirely unused while
  `user_config.py` defines its own unexpanded constant). Import from
  `papyri.config` everywhere.
- **`config_loader` has no real validation at a system boundary.**
  `sys.exit(...)` instead of `typer.Exit`; `conf["global"]` and
  `info.pop("module")` raise bare `KeyError` tracebacks; the
  `assert len(ks) >= 1` checks nothing; `Config(**conf)` turns a typo'd
  TOML key into a raw `TypeError`; and an unknown top-level section is
  dropped silently (this is what made `examples/pandas.toml`'s
  `[pandas.expected_errors]` a no-op — the file was deleted 2026-09 rather
  than fixed, since nothing in CI generated it). Name the offending key and
  raise; keep `assert` for the invariants.
- **`bootstrap` is the only command taking `str` instead of `Path`** (and
  the only one with no `expanduser()`, so `papyri bootstrap ~/x.toml`
  creates a literal `~` directory) **and the only one using raw
  `input()`**, which raises `EOFError` under non-interactive CI. Use
  `typer.prompt`.
- **Small items.** `pyproject.toml`'s `postingest` marker is unused and
  tells you to run two deleted commands (`papyri ingest` + relink); its
  coverage `omit` lists `papyri/examples/*`, a directory that does not
  exist. `scripts/*.py` pin `#!/usr/bin/env python3.13` while CI runs 3.14.
  `cli/about.py` duplicates `--version`/`-V` from a second `__version__`
  source while `upload.py` reads `importlib.metadata`, so the User-Agent
  and `papyri --version` can disagree — and `PAPYRI_VERSION`, documented in
  CLAUDE.md as overriding that User-Agent, is never consulted. `find`
  prints `SKIP …` to stdout where every other diagnostic goes to stderr,
  and `find`/`diff` exit 0 even when nothing matched.
- **`examples/papyri.toml` hardcodes `~/dev/papyri/docs`** for `docs_path`
  and `examples_folder`. Unlike `logo`, those are only `expanduser()`'d,
  not resolved against the config file's directory — so the verification
  step CLAUDE.md tells every contributor to run silently produces a bundle
  with no narrative docs unless their clone is at that exact path. Its
  `github_slug` is also `jupyter/papyri`, so papyri's own self-generated
  bundle emits `:ghpull:`/`:ghissue:` links to the wrong repository.

## Open work — Documentation drift (2026-09 review)

The cross-language IR contract is *verified* — `scripts/check_ir_sync.py`
and `scripts/gen_ir_schema.py` are CI-enforced, and all 55 tags, class
names, field orders and `ir-schema.ts` entries agree between Python and
TS. The prose around it has rotted:

- **Four overlapping trackers describe a layout that no longer exists.**
  `TODO`'s "Open code smells" and "Next up" cite `papyri/crosslink.py`
  (renamed to `ingested_doc.py`) — the TS half of the same item is still
  accurate and open, so only the Python pointers rotted.
  `TODO-renames.md` opens with a "nothing on this list has landed yet"
  banner that is false, asks to delete an `IntermediateNode` that is
  already gone, and proposes renaming CLI commands (`ingest`, `relink`,
  `drop`) that no longer exist. Fold both into PLAN.md or delete them.
- **`docs/IR.md`'s tag registry is wrong.** It lists `4020 Heading` (no
  such class), and `4052 Directive` / `4060 Comment` (real classes,
  deliberately untagged), and omits roughly fifteen tags including `4070
  Bundle` and `4010 IngestedDoc`. It also cites a `DocBundler.write` that
  does not exist and a "PLAN.md Phase 2" that no longer does. Generate the
  table from `TAG_MAP` instead of hand-maintaining it.
- **`docs/IR-NODE-AUDIT.md` still asserts a bug that was fixed.** Its
  central claim — that `tree.py` wraps unhandled directives in a
  *serializable* `Directive`, "the leak path that lets raw directives reach
  disk" — is stale: `Directive` is an `UnserializableNode`. Item 3's
  "`Comment` wastes bytes on disk" is likewise stale (stripped at pack),
  and every line number in the document is off. A reader would re-do closed
  work.
- **`ingest/src/encoder.ts` — the file that *is* the cross-language
  contract — names two source files that do not hold those classes**
  ("`papyri/crosslink.py` (IngestedDoc), `papyri/gen.py` (GeneratedDoc)";
  actually `ingested_doc.py` and `doc.py`).
- **CLAUDE.md's repository layout.** Viewer routes are documented one
  level too shallow (`pages/[pkg]/[ver]/…`; the real routes live under
  `pages/project/[pkg]/[ver]/…`), and nine real modules are missing —
  `papyri/{__main__,_progress,bundle_edit,errors,user_config,utils}.py`,
  `papyri/cli/lint.py`, `ingest/src/{fs-safe,url-safety}.ts` (both
  security guards in the done log), and
  `viewer/src/lib/{auth-db,passkey}.ts` — as are `viewer/README.md`,
  `viewer/DEPLOY.md` and `viewer/TODO.md`. `bundle_edit.py` is called "the
  supported custom step between gen and pack" here yet is invisible there.
- **CLAUDE.md's env table.** `PAPYRI_PORT` (docker-compose) and
  `PAPYRI_COMMIT` (the actual input feeding the documented
  `PAPYRI_BUILD_COMMIT`) are undocumented; `PAPYRI_BUILD_ADAPTER` is
  hard-coded to `"node"` in `buildDefine` and never read from the
  environment at all. `docs/configuration.rst` omits `doctest_optionflags`.
- **PLAN.md's own file paths.** `ingest/migrations/0000_init.sql` (actual
  `0001_init.sql`), `0001_external_inventory.sql` in the done log (actual
  `0003_…`), and `viewer/src/pages/api/bundle/[...path].ts` (actual
  `api/bundle.ts`) — the last inside a scoped open-work item, so it is
  actionable rot.
- **`viewer/PLAN.md` contradicts itself.** "Must-have (v0)" still requires
  a static build while "Open questions" records static export as parked
  (2026-07) and `astro.config.mjs` is `output: "server"`; the tech table
  claims a Playwright smoke suite that is in no `package.json`; the
  architecture sketch says `astro.config.ts`.
- **`Readme.md`** claims Python 3.14+ (`requires-python` is `>=3.13`; 3.14
  is CI only) and describes gen as emitting `module/*.cbor` — the same
  stale encoding claim already flagged for `docs/IR.md`, but in the first
  file a contributor reads.
- **Stale files.** `assets/` holds ~1.4 MB of screenshots referenced by
  nothing; `examples/gallery1.py` is unreferenced;
  `viewer/tests/bundle-ingest.test.ts` tests nothing about bundle ingest
  (it is a second copy of the `isSafeSegment` suite) and should be renamed
  or merged; `viewer/README.md` says Node 20+ where `engines` requires 22.

## Done log

Terse, grep-able record of what exists so future work doesn't re-derive it.
Newest areas first; each line names the key symbol/file.

### Multi-agent review pass (2026-09)

Five parallel reviews (gen core, CLI/storage, ingest, viewer, cross-cutting).
Findings are filed as open work above; what landed in this pass:

- **Bundle-asset route hardened** (`viewer/src/pages/assets/project/[pkg]/[ver]/[...asset].ts`).
  Was a MIME lookup table that deliberately served `.html`/`.htm`/`.js` as
  active content on the viewer's own origin, with no CSP anywhere in the
  app — stored XSS for anyone holding an upload token for any project. Now
  an extension **allow-list** (images only; gen never writes anything else
  into `assets/`), `Content-Security-Policy: default-src 'none'; sandbox`
  so an SVG carrying `<script>` lands in an opaque origin on direct
  navigation, `X-Content-Type-Options: nosniff`, and an explicit
  `Content-Disposition` with a sanitised filename (header-injection safe).
  `pkg`/`ver` now go through `isSafeSegment` first, so a traversal attempt
  is a 400 rather than an uncaught `safeJoin` throw. Tests:
  `viewer/tests/asset-route.test.ts`.
- **Silent data loss in `node_serializer.serialize` fixed.** The final
  branch's first disjunct tested `type.__module__` (the builtin `type`)
  instead of `annotation.__module__`, so it was always false and the check
  degraded to exact-type equality — and there was no terminal `else`, so a
  value matching no branch fell off the end and returned `None`
  implicitly, writing the field to the bundle as JSON `null` with no
  diagnostic. Both fixed; the function now raises. Hint resolution moved to
  `serde.get_type_hints`, which filters `ClassVar` (previously every
  `GeneratedDoc` carried a null `sections` key). `serde.serialize` has the
  same dead `type.__module__` check but does raise at the end, so it loses
  nothing — left alone deliberately; its second dead term
  (`getattr(annotation, "_name", …)`, never set on a Node class) means
  fixing only the module half would change nothing.
- **`GeneratedDoc.content` proxy desync fixed** (`gen.py`). `_content` was
  rebound to a fresh dict after visiting, but `_OrderedDictProxy` captured
  the original mapping by reference at construction — so the
  named-section loop that followed wrote its `dv.visit` results into the
  dict the rebind had just orphaned, and they were discarded. Now updated
  in place, and the redundant second pass (a subset of the same keys) is
  gone along with the "the proxy may be stale" workaround comment it forced.
- **Bundles are reproducible across machines.** `gen`'s narrative
  (`**/*.rst`) and example (`**/*.py`) scans iterated `Path.glob` in
  `os.scandir` order, which decides which duplicate `:ref:` label and which
  external target wins and becomes the narrative-doc/toc order — so the
  same source tree produced different bundles, and different
  `content_hash`, on different filesystems. Both sorted. `pack`'s
  validation walks are sorted too, so the *first* reported problem in a
  malformed bundle is deterministic. (The remaining basename-collision
  bug is filed under Gen above.)
- **`PAPYRI_INGEST_DB` is honoured.** `backends.ts` hardcoded
  `<ingestDir>/papyri.db`; the one function reading the variable
  (`paths.ts:ingestDb()`) had no production callers, so the variable
  documented in CLAUDE.md, `viewer/PLAN.md`, `DEPLOY.md` and
  `backends.ts`'s own header did nothing. `paths.ts` now owns both paths
  (`ingestDir()` added, `ingestDb()` defaults inside it) and `backends.ts`
  calls them, so pointing `PAPYRI_INGEST_DIR` elsewhere moves the DB with
  it and `PAPYRI_INGEST_DB` still wins.
- **`papyri/py.types` → `papyri/py.typed`.** The PEP 561 marker was
  misnamed, so a strictly-typed package shipped no inline types to
  downstream mypy users.
- **`examples/pandas.toml` deleted.** Its `[pandas.expected_errors]` table
  should have been `[global.expected_errors]` and was silently dropped by
  the loader (~35 entries), and it carried an empty-string `exclude`
  entry. Nothing in CI generated it. The underlying "unknown top-level
  section is ignored" bug is filed under CLI uniformisation above.

### Sphinx-fidelity pass (2026-07 example sweep)
- Ref resolution: leading-`!` suppression → plain `InlineCode`, no warning;
  trailing `()` stripped before resolve (display keeps parens); scope walk
  most-specific-first *including the current scope*; colon-form qa
  ("numpy:any") normalized before scope derivation — all in `tree.py`
  (`replace_InlineRole` / `resolve_`).
- Narrative↔API cross-linking: `Gen._scan_narrative_sources()` (cached first
  pass) gives API visitors the narrative `doc_targets`/`external_targets`
  maps (`:ref:` from docstrings resolves) and narrative visitors get the API
  `known_refs` (`Gen._known_refs`); narrative doc keys resolve refs against
  the package root (`DirectiveVisiter._resolve`).
- `:doc:` role resolved through the visitor (`_resolve_doc_path`) → doc-key
  LocalRefs ("api:axes_api", not "/api/axes_api"); free-function
  `py_doc_handler` deleted.
- plot directive: external-script arguments embedded (doc_path/doc_root),
  doctest-format bodies execute with prompts stripped, exec namespace
  pre-seeded with np/plt (matplotlib `plot_pre_code` default); `doc_root`
  threaded into API visitors for `/`-rooted image paths.
- numpydoc leniency: unknown section headings fall through to upstream
  warn+skip (was: ValueError → sentinel/object drop); backticked See Also
  entries (`numpy.polynomial`) accepted (`numpydoc_compat.py`).
- Import solver: objects `full_qual()` cannot name (method descriptors,
  numpy.ufunc.reduce) fall back to longest-imported-module-prefix qualname.
- `W-unresolved-default-role` (default `info`): bare-backtick lookup misses
  split off from `W-unresolved-ref` (Sphinx autolink degrades silently).
- Doctest-execution `catch_warnings()` now actually encloses the run, so
  example code cannot leak warning-filter mutations into gen.
- Pack lint enforcement: `_check_lint` in the pack path — Substitution nodes
  always fatal (IR invariant); missing Figure assets + `DocstringSentinel`
  warn by default, error under `--strict`; sentinel check added to
  `lint_bundle` (closes the former "pack strict-mode / lint gaps" item).
- examples/numpy.toml exclusions pruned to just MaskedArray.resize (#361).
- Examples collected *after* API docs and their visitor now gets
  `known_refs` + narrative target maps, so example pages cross-link.
- `[global.roles]` config table (`Config.roles` → `DirectiveVisiter`
  `_role_handlers`): project-local roles map to handlers
  (`role_verbatim`/`role_text`/`role_drop` in `directives.py`);
  matplotlib's `:mpltype:` mapped in its example config. Docs:
  `configuration.rst` `[global.roles]`.

### Everything-explicit pass (2026-07 review + adversarial audit)
- Unknown roles are a hard failure: `W-unknown-role` (default `error`),
  emitted *before* any resolution attempt (`replace_InlineRole` gate on
  `_PYTHON_OBJECT_ROLES` / `ref` / `doc`), so an unmapped role can never
  accidentally cross-link. Standard Sphinx std-domain formatting roles
  (`envvar`, `dfn`, `abbr`, `guilabel`, `menuselection`, `option`, …) are
  built in as verbatim under both `py` (bare) and `std` domains; IPython,
  distributed, and scikit-image example configs carry explicit
  `[global.roles]` mappings for their project-local roles.
- `DocstringSentinel` always refuses to pack (`_check_lint`), alongside
  substitution nodes; `InlineRole` residue and `Unimplemented` placeholders
  are counted by `lint_bundle` and refuse under `pack --strict`.
- `resolve_` substring fallback (`ref in q`) deleted; both suffix searches
  require a component boundary and dedupe through the RefInfo; ambiguity
  is unresolved (audit N1).
- numpydoc compat: `_guess_header` restricted to trailing-`:`/missing-`s`/
  alias-table normalizations; every rewrite (heading, See Also backticks)
  recorded on the instance and emitted by gen as
  `W-section-heading-normalized` / `W-see-also-syntax` (audit N7).
  `__setitem__` records only sections that landed (audit N2: unknown
  sections used to KeyError and silently drop the object).
- `exec_failure="fallback"` no longer trips the end-of-collection assert
  (audit N9).

### Gen-time diagnostics
- Core framework: `Severity`, `DIAGNOSTICS` registry, `DiagnosticConfig`
  resolver (default → global → first-match per-target glob), `Diagnostics`
  collector — all in `error_collector.py`. Config `[global.diagnostics]` +
  `per-target` sub-table (`from_raw`, unknown codes/severities fail the run);
  `Config.diagnostics` / `error_on_warning`. `papyri gen` logs a per-severity
  summary, records into `papyri.json` `diagnostics`, and exits non-zero on any
  `error` (`--no-error-on-warning` escape hatch). Docs: `configuration.rst`.
- Codes: `W-unresolved-ref`, `W-unsupported-substitution`,
  `W-malformed-directive`, `W-missing-github-slug` (tree.py); `W-doctest-syntax`,
  `W-doctest-exec`, `W-numpydoc-parse`, `W-module-docstring` (gen.py).
- `directives.py` malformed-directive wiring: `list-table`/`csv-table`/`image`/
  `figure`/`include`/`plot` route recoverable failures through a `warn`
  callback the visitor binds to `DirectiveVisiter._directive_warn`
  (`W-malformed-directive`); handlers built outside gen still log plainly.
- `:ghpull:`/`:ghissue:` de-globalized: resolved in `replace_InlineRole`
  against a per-visitor `github_slug` (from `[meta].github_slug`),
  `W-missing-github-slug` when unset. `_GITHUB_SLUG`/`set_github_slug()` gone.

### Gen / Python
- Bundle-edit API for script-generated pages: `papyri.bundle_edit`
  (`read_doc`/`write_doc`/`narrative_doc`/`replace_block`/`add_toc_entry`)
  — the supported custom step *between* gen and pack. Projects whose
  Sphinx build generates pages (IPython's config options / magics /
  shortcuts) inject the same content directly as IR nodes, no RST
  round-trip: one CI line running an injector script against the bundle
  dir. `replace_block` works on the flat Section-with-level page shape,
  making injectors idempotent. Reference injector:
  `examples/ipython_inject.py` (traitlets options → admonitions, magics →
  DefList, prompt_toolkit bindings → Table). Docs: `docs/injecting.rst`.
- `csv-table` `:file:`/`:delim:`/`:encoding:` support (`:url:` warns and
  drops — no network at gen time); resolution mirrors `include`
  (`doc_path`/`doc_root`); the delimiter also applies to the `:header:`
  option row — `csv_table_handler`, `directives.py`.
- Clinic signatures as fallback `ObjectSignature` for `type`s
  (`strip_clinic_signature` → `extract_docstring`, `gen.py`).
- Missing block directives audit (2026-04-30/05-01) closed: `rubric`, `only`,
  `literalinclude`, `csv-table` handled; sphinx-design / `automodule` /
  `currentmodule` / `testcode`&c. / `highlight` / py-domain `function`/`class`
  in `_SPHINX_ONLY_DIRECTIVES` (silent drop). No known remaining directives.
- Unified "version unknown" marker: gen emits `RefInfo(version="?",
  kind="module")`; `visitor.ts` normalization gone; `graph.ts` uses `'?'`.
- Configurable doctest `optionflags` (`config_loader.py` / `gen.py`).
- Module-docstring parse failures → `DocstringSentinel` (tag 4072) + warning,
  recorded in `failure_collection`; encoder/IR/renderer updated.
- `LocalRef` dangling-ref policy: gen does not rewrite; `pack` surfaces it.
- Deleted dead code: `normalise_ref`'s test, `mod_root`. (2026-09: the
  function itself is still in `gen.py` and still has no callers — see
  "Small dead/incoherent bits" under Open work — Gen.)
- Renamed `crosslink.py` → `ingested_doc.py`.
- `graphstore.py` write-side removed (`put`/`put_meta`/`remove`/
  `_maybe_insert_node`, schema creation) — read-only; TS ingest owns writes.
- `ts.parse()` LRU-cached (`_parse_cached(bytes)`, maxsize 512).
- `Gen.collect_api_docs` decomposed (`_collect_and_filter_items`,
  `_process_one_api_item`).
- Directive-handler registry: `self._handlers` dict is the sole dispatch path;
  legacy getattr path removed.
- Stale `papyri ingest` reference in `describe.py` fixed → `papyri upload` /
  `POST /api/reingest`.
- Serialization paths documented (`node_serializer.py` ↔ `serde.py`).

### Pack / lint
- `papyri lint` subcommand + `lint_bundle()` (SubstitutionRef/Def, missing
  Figure assets); tests in `test_pack.py`.
- `pack --strict`: orphan-doc warnings → hard `BundleError`.
- Dangling local refs: `_check_local_refs` (warn; `--strict` errors); viewer
  renders `.xref.unresolved.broken-local`; `/…/validate` lists them.
- Toc↔narrative: `_check_toc_refs` (hard fail on dangling toc entry);
  `find_orphan_docs`/`_warn_orphan_docs` (warn); numpy smoke tests.
- Silent-drop→hard-pack-failure gate tried and reverted (gen/pack are separate
  steps; pack must not gate on a stale gen-time error record).

### Ingest (TypeScript)
- Dropped directory-based ingest: `ingestBundle(node)` (decoded packed
  `.papyri`) is the sole contract; `ingest()`/`_put`/`_ingest*Dir`/
  `explodeBundleToDir`/`IngestOptions.check` and the standalone
  `papyri-ingest` CLI removed. `papyri-ingest` is now a library.
- Raw upload timestamps: `FsRawStore.put()` writes `<ver>.meta.json` sidecar;
  `RawStore.getMeta()`.
- Type-safe key parsing: `keyStr`/`parseKeyStr` in `keys.ts`.
- Unified forward-ref collection: single `forwardRefKeys(subtree)` path.
- `assertBundle` deep shape validation (`bundle.ts`).

### Viewer
- Async storage+graph layer: `BlobStore`/`GraphDb`/`RawStore`, built per-request
  by `backends.ts`; pages call `getBackends()`; xref batched per page.
- In-process upload `PUT /api/bundle` (gunzip → CBOR decode → `ingestBundle`);
  raw archive `_raw/<pkg>/<ver>.papyri.gz`; `POST /api/reingest` (NDJSON,
  `?pkg=`/`?ver=`).
- Precomputed `node_index` table (migration `0006`) for image-index /
  node-browser; falls back to `walkBundle` for pre-migration bundles.
- CSS dead-code audit (only `.sidebar-stub` was dead); per-kind admonition
  styling (`admonition-${kind}` + tokens/icons).
- Dual Shiki themes + dark-aware KaTeX (`.katex-html { color: inherit }`).
- Crossrefs default to latest linking version per source package
  (`bucketBackrefs`/`filterToLatestVersionPerPkg`; PEP 440 pre-release
  exclusion; wildcards always kept).
- Unresolved-link warnings: inline `<span class="xref unresolved">` +
  `/…/validate` report (`collectXrefsDetailed`).
- Incoming broken-link report `/…/backref-validate` + count badge
  (`getBrokenBackrefs`/`countBrokenBackrefs`).
- Content-hash dedup: `papyri upload` skips when the viewer holds the same
  `bundles.content_hash` (`--force` bypasses).

### Auth / security
- User/session store in a separate SQLite DB (`auth-db.ts`, `PAPYRI_AUTH_DB`):
  Argon2id passwords (constant-time verify + decoy on unknown user), opaque
  server-side sessions with `created_at`/`expires_at` (7-day TTL, revocable),
  fail-closed when unseeded (`PAPYRI_USERNAME`/`PAPYRI_PASSWORD`); dev-only demo
  admin gated by `PAPYRI_DEV_SEED`. Three-tier middleware (public / admin-only /
  signed-in / guest-browsable).
- Per-user upload authz: `users.is_admin`, `projects`/`project_members`,
  personal `upload_tokens` (SHA-256 stored) minted at `/settings`; `PUT
  /api/bundle` authenticates bearer → principal, authorizes per `module`
  (global `PAPYRI_UPLOAD_TOKEN` = escape hatch).
- Path traversal closed via `safeJoin` (`fs-safe.ts`) in `FsBlobStore`/
  `FsRawStore`; `_safe_child` in `pack.py` for `papyri unpack`.
- `javascript:`/`data:` URL blocking: `isSafeUrl` (`url-safety.ts`) enforced at
  renderer (`render-node.ts`), ingest (`assertSafeUrls`), pack
  (`_assert_safe_urls`).
- Upload robustness: streaming PUT idle timeout (`_UPLOAD_IDLE_TIMEOUT_S`);
  zip-bomb ceiling (`_MAX_BUNDLE_BYTES`, 256 MiB); `inflateZlib` corrupt-body
  try/catch in `parseObjectsInv`.
- External (intersphinx) linking: `POST /api/inventory` +
  `ingest/src/inventory.ts` parser + `0001_external_inventory.sql` +
  `resolveExternalRefs` (`xref.ts`); admin UI `ExternalInventoryPanel.tsx`.
