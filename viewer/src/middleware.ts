import { defineMiddleware } from "astro:middleware";
import { getAuthDb, SESSION_COOKIE } from "./lib/auth-db.ts";

// Routes that must remain reachable without a session (login form, auth
// endpoints, bundle upload). Any new pre-auth route needs an entry here.
// `/api/bundle` is the upload endpoint hit by `papyri upload`; it carries
// its own bearer-token check and must stay reachable without a session cookie.
const PUBLIC_PREFIXES = ["/login", "/api/auth/", "/api/bundle"] as const;

// Routes restricted to authenticated users. Everything not listed here (and
// not in PUBLIC_PREFIXES) is accessible to guests so they can browse docs
// without logging in.
//
// Admin-only routes are computationally expensive (full corpus walks), carry
// destructive write operations, or manage accounts / project membership;
// guests and non-admin users have no need for them. These require a session
// whose user has `is_admin` set.
const ADMIN_ONLY_PREFIXES = [
  "/admin",
  "/nodes",
  "/ir-stats",
  "/api/nodes.json",
  "/api/ir-stats.json",
  "/api/clear",
  "/api/clear-raw",
  "/api/delete-bundle",
  "/api/reingest",
  "/api/inventory",
  "/api/stats",
  "/api/users",
  "/api/projects",
] as const;

// Routes any signed-in user may reach but guests may not — self-service
// account management (change password, mint/revoke personal upload tokens).
// These require a session but NOT admin.
const AUTH_REQUIRED_PREFIXES = ["/settings", "/api/account"] as const;

/** True when `pathname` equals `prefix`, `prefix + "/"`, or any deeper path. */
function matchesPrefix(prefix: string, pathname: string): boolean {
  return pathname === prefix || pathname === prefix + "/" || pathname.startsWith(prefix + "/");
}

function matchesAny(prefixes: readonly string[], pathname: string): boolean {
  return prefixes.some((p) => matchesPrefix(p, pathname));
}

export const onRequest = defineMiddleware(async (context, next) => {
  const { pathname } = context.url;

  // Public routes bypass all auth checks.
  if (matchesAny(PUBLIC_PREFIXES, pathname)) {
    return next();
  }

  // Admin-only and signed-in-user routes both require an active, unexpired
  // session. We validate the token against the auth store (not just its
  // presence) so a stale, forged, or revoked cookie is rejected.
  const adminOnly = matchesAny(ADMIN_ONLY_PREFIXES, pathname);
  if (adminOnly || matchesAny(AUTH_REQUIRED_PREFIXES, pathname)) {
    const token = context.cookies.get(SESSION_COOKIE)?.value;
    const user = token ? (await getAuthDb()).resolveSession(token) : null;
    const isApi = pathname.startsWith("/api/");
    if (!user) {
      // API callers receive a JSON 403 instead of an HTML redirect.
      if (isApi) {
        return new Response(JSON.stringify({ error: "Authentication required" }), {
          status: 403,
          headers: { "Content-Type": "application/json" },
        });
      }
      return context.redirect("/login");
    }
    // Admin-only routes additionally require the admin role. A signed-in but
    // non-admin user is forbidden (API) or bounced to the bundle index (page).
    if (adminOnly && !user.is_admin) {
      if (isApi) {
        return new Response(JSON.stringify({ error: "Admin privileges required" }), {
          status: 403,
          headers: { "Content-Type": "application/json" },
        });
      }
      return context.redirect("/");
    }
  }

  // Everything else (docs, bundle index, text search, …) is open to guests.
  return next();
});
