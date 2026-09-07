/// <reference types="astro/client" />

interface ImportMetaEnv {
  /** Git commit SHA baked in at build time (see astro.config.mjs). */
  readonly PAPYRI_BUILD_COMMIT: string;
  /** Astro adapter baked in at build time (currently always "node"). */
  readonly PAPYRI_BUILD_ADAPTER: string;
  /** True while rendering a static snapshot (`PAPYRI_STATIC=1`). See lib/static.ts. */
  readonly PAPYRI_STATIC_BUILD: boolean;
  /** Copyright line in the site footer. Footer hidden when all PAPYRI_FOOTER_* are unset. */
  readonly PAPYRI_FOOTER_COPYRIGHT: string;
  /** "Privacy policy" link URL in the site footer. */
  readonly PAPYRI_FOOTER_PRIVACY_URL: string;
  /** "Terms of service" link URL in the site footer. */
  readonly PAPYRI_FOOTER_TERMS_URL: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
