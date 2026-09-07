// Static-build flag and the handful of policies that differ between the
// long-running SSR server and a `pnpm build:static` snapshot.
//
// A static snapshot is rendered by the *same* Astro templates as the server —
// `getStaticPaths` enumerates the URLs (see `static-paths.ts`) and Astro walks
// them. Nothing here changes how a page renders; every use is "omit an
// affordance that needs a live server". Keep that property: if a branch on
// `isStaticBuild()` would produce different *content*, it belongs somewhere
// else.
//
// The flag is a build-time constant substituted by Vite (see the `buildDefine`
// block in `astro.config.mjs`), so dead branches are eliminated from the
// client bundles rather than shipped and skipped.
//
// It is deliberately *not* named after the `PAPYRI_STATIC` env var that
// selects the mode: Astro exposes process env vars on `import.meta.env`
// itself, and a same-named var wins over the injected constant — which would
// silently hand every caller the string "1" instead of a boolean.

/** True while rendering a static snapshot (`PAPYRI_STATIC=1 astro build`). */
export function isStaticBuild(): boolean {
  return import.meta.env.PAPYRI_STATIC_BUILD === true;
}

/**
 * URL version to link a package's newest bundle under.
 *
 * The server serves `/project/<pkg>/latest/` and resolves the alias per
 * request. A snapshot cannot: `latest` would have to be emitted as a second
 * full copy of the newest bundle's pages, roughly doubling the output for the
 * bundle that has the most pages. So a snapshot links concrete versions and
 * `latest` URLs are handled by the 404 page instead.
 */
export function urlVersionForLatest(latest: string): string {
  return isStaticBuild() ? latest : "latest";
}
