// Pure helpers for rendering a SignatureNode. Extracted from Signature.astro
// so the parameter classification logic is testable without Astro.
import type { SignatureNodeT, SigParamT } from "./ir-reader.ts";

/**
 * Return the Python declaration keyword prefix for a callable.
 * Classes get "class"; async callables get "async def"; built-ins get null
 * (no `def` in Python syntax); everything else gets "def".
 */
export function sigQualifier(
  kind: string,
  itemType: string | null
): "def" | "async def" | "class" | null {
  if (itemType === "class") return "class";
  if (kind === "coroutine function" || kind === "async_generator function") return "async def";
  if (itemType === "built-in" || kind === "built-in function") return null;
  return "def";
}

/**
 * Return a badge label for callable kinds that warrant an extra decoration,
 * or null for plain functions and classes.
 */
export function sigKindBadge(kind: string, itemType: string | null): string | null {
  if (itemType === "property") return "property";
  if (itemType === "classmethod") return "classmethod";
  if (itemType === "staticmethod") return "staticmethod";
  if (itemType === "built-in" || kind === "built-in function") return "built-in";
  return null;
}

export interface SignatureEntry {
  marker: "/" | "*" | null;
  param: SigParamT;
}

/** Return the annotation as a plain string, or null for Empty/missing. */
export function annText(
  ann: SigParamT["annotation"] | SignatureNodeT["return_annotation"]
): string | null {
  if (ann == null) return null;
  if (typeof ann === "string") return ann;
  return null;
}

/**
 * Return the default value as a plain string, or null for Empty/missing.
 * The IR stores defaults as `repr(value)`, so string defaults keep their quotes.
 */
export function defaultText(def: SigParamT["default"]): string | null {
  if (def == null) return null;
  if (typeof def === "string") return def;
  return null;
}

/**
 * Build the ordered list of (marker, param) entries for rendering a signature.
 * Inserts `/` before the first non-positional-only parameter and `*` before
 * the first keyword-only parameter when no VAR_POSITIONAL is already present.
 * Also returns `trailingSlash` for the case where every parameter is
 * positional-only.
 */
export function buildSignatureEntries(signature: SignatureNodeT): {
  entries: SignatureEntry[];
  trailingSlash: boolean;
} {
  type Kind = SigParamT["kind"];
  const entries: SignatureEntry[] = [];
  let prevKind: Kind | null = null;
  let sawVarPositional = false;
  for (const p of signature.parameters) {
    let marker: SignatureEntry["marker"] = null;
    if (prevKind === "POSITIONAL_ONLY" && p.kind !== "POSITIONAL_ONLY") {
      marker = "/";
    } else if (p.kind === "KEYWORD_ONLY" && prevKind !== "KEYWORD_ONLY" && !sawVarPositional) {
      marker = "*";
    }
    entries.push({ marker, param: p });
    if (p.kind === "VAR_POSITIONAL") sawVarPositional = true;
    prevKind = p.kind;
  }
  const trailingSlash = prevKind === "POSITIONAL_ONLY";
  return { entries, trailingSlash };
}
