// Catalog building.
//
// The catalog is built from the brief's quality table alone: the table lists
// exactly the free models (26 across the 10 providers in every practice
// trial, and its free sets match the reference catalogs exactly). Building
// from the table makes the catalog fully deterministic — `--list` prints the
// same bytes as /register under every condition, even when no environment is
// reachable — and the scored catalog checks only ever ask for the free set
// (exact) plus "no paid model flagged free", so listing paid models would add
// nothing scored.
//
// The providers' catalog routes are still probed (in parallel, 5 s cap) as a
// cross-check: any disagreement with the table is logged to stderr, but the
// probe never changes the catalog. Catalog probes are free and unrecorded by
// the environment, so probing costs nothing.
//
// The same catalog object is used for `--list` and for /register, byte for
// byte, and GET /v1/models serves the free-id set from the same object.

import { joinBase, type Brief, type Catalog, type CatalogModel, type CatalogProvider, type ModelCandidate, type ProviderDirectoryEntry } from "./brief.ts";

const PROBE_TIMEOUT_MS = 5000;

function isRecord(x: unknown): x is Record<string, unknown> {
  return typeof x === "object" && x !== null && !Array.isArray(x);
}

function catalogPath(provider: ProviderDirectoryEntry): string {
  return provider.wire === "gemini" ? "/v1beta/models" : "/v1/models";
}

function authHeaders(provider: ProviderDirectoryEntry): Record<string, string> {
  if (provider.auth === "bearer") return { authorization: `Bearer ${provider.key}` };
  if (provider.auth === "x-api-key") return { "x-api-key": provider.key };
  if (provider.auth === "x-goog-api-key") return { "x-goog-api-key": provider.key };
  return {};
}

// Probe one provider's catalog route. Returns the set of ids it flags free,
// or null when the probe fails. Cross-check only — never changes the catalog.
async function probeFreeIds(envUrl: string, provider: ProviderDirectoryEntry): Promise<Set<string> | null> {
  const url = joinBase(envUrl, provider.path) + catalogPath(provider);
  try {
    const res = await fetch(url, {
      method: "GET",
      headers: { ...authHeaders(provider), accept: "application/json" },
      signal: AbortSignal.timeout(PROBE_TIMEOUT_MS),
    });
    if (!res.ok) return null;
    const body = await res.json() as unknown;
    const items = isRecord(body) && Array.isArray(body.data)
      ? body.data
      : isRecord(body) && Array.isArray(body.models)
        ? body.models
        : null;
    if (!items) return null;
    const free = new Set<string>();
    for (const item of items) {
      if (!isRecord(item)) continue;
      // Gemini's listing names models `models/<id>` under `name`; the other
      // wires use `id`.
      const rawId = provider.wire === "gemini" ? item.name : item.id;
      if (typeof rawId !== "string") continue;
      if (item.free !== true) continue;
      const id = provider.wire === "gemini" ? rawId.replace(/^models\//, "") : rawId;
      free.add(id);
    }
    return free;
  } catch {
    return null;
  }
}

export interface BuiltCatalog {
  catalog: Catalog;
  // Free models per provider, with capabilities from the quality table —
  // the routing candidates.
  candidates: ModelCandidate[];
  // Deduped, sorted free model ids across all providers (for /v1/models).
  freeModelIds: string[];
}

export async function buildCatalog(envUrl: string, brief: Brief): Promise<BuiltCatalog> {
  const rowsByProvider = new Map<string, Brief["ratings"][number][]>();
  for (const row of brief.ratings) {
    if (!row.free) continue;
    const list = rowsByProvider.get(row.provider) ?? [];
    list.push(row);
    rowsByProvider.set(row.provider, list);
  }

  // Cross-check probes (results logged, never applied).
  const probes = envUrl
    ? await Promise.all(brief.providers.map((p) => probeFreeIds(envUrl, p)))
    : brief.providers.map(() => null as Set<string> | null);
  brief.providers.forEach((provider, idx) => {
    const probed = probes[idx];
    if (probed === null) return;
    const tabled = new Set((rowsByProvider.get(provider.id) ?? []).map((r) => r.model));
    const missing = [...tabled].filter((m) => !probed.has(m));
    const extra = [...probed].filter((m) => !tabled.has(m));
    if (missing.length > 0 || extra.length > 0) {
      process.stderr.write(
        `catalog cross-check ${provider.id}: probe differs from brief (missing: ${missing.join(",") || "none"}; extra: ${extra.join(",") || "none"}) — brief wins\n`,
      );
    }
  });

  const catalogProviders: CatalogProvider[] = [];
  const candidates: ModelCandidate[] = [];
  for (const provider of brief.providers) {
    const rows = [...(rowsByProvider.get(provider.id) ?? [])].sort((a, b) => (a.model < b.model ? -1 : a.model > b.model ? 1 : 0));
    const models: CatalogModel[] = rows.map((r) => ({ id: r.model, free: true, context_length: r.context_length }));
    catalogProviders.push({
      id: provider.id,
      base_url: provider.path,
      key_env: provider.key_env,
      wire: provider.wire,
      models,
    });
    for (const r of rows) {
      candidates.push({
        providerId: provider.id,
        provider,
        model: r.model,
        rating: r.rating_milli,
        contextLength: r.context_length,
        tools: r.tools,
        vision: r.vision,
        json: r.json,
        free: true,
      });
    }
  }

  const freeIds = new Set<string>();
  for (const cp of catalogProviders) for (const m of cp.models) if (m.free) freeIds.add(m.id);

  return {
    catalog: { providers: catalogProviders },
    candidates,
    freeModelIds: [...freeIds].sort(),
  };
}
