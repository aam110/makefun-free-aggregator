// Brief parsing + shared types for the free-model-aggregator entry.
//
// The trial brief (MAKEFUN_PROMPT_FILE) is a text file embedding three JSON
// documents: the provider directory, the quality table, and the workload
// manifest. This module extracts them by scanning for balanced JSON objects
// and identifying each document by its shape. All parsing is deterministic.

export type Wire = "openai" | "anthropic" | "gemini";

export interface ProviderDirectoryEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: "bearer" | "x-api-key" | "x-goog-api-key" | "none";
  key_env: string;
  key: string;
  // The brief's directory writes `path` (/p/<id>, joined to $MAKEFUN_ENV_URL);
  // a directory spelling an absolute base instead may use base_path, base_url
  // or base. Normalized at parse time: `path` is the first non-empty of those.
  path: string;
}

export interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

export interface BestRow {
  id: string;
  rating_milli: number;
}

export interface ManifestRequest {
  id: string;
  api: "openai" | "anthropic";
  stream: boolean;
  weight: number;
  requires: {
    tools?: boolean;
    vision?: boolean;
    json?: boolean;
    min_context?: number;
  };
}

export interface Brief {
  providers: ProviderDirectoryEntry[];
  ratings: RatingRow[];
  best: BestRow[];
  manifest: ManifestRequest[];
}

function isRecord(x: unknown): x is Record<string, unknown> {
  return typeof x === "object" && x !== null && !Array.isArray(x);
}

// Extract every top-level JSON object from the text by matching braces,
// skipping string literals (the brief's bodies contain braces inside strings).
export function extractDocuments(text: string): unknown[] {
  const docs: unknown[] = [];
  let i = 0;
  while (i < text.length) {
    const start = text.indexOf("{", i);
    if (start < 0) break;
    let depth = 0;
    let inString = false;
    let escaped = false;
    let j = start;
    for (; j < text.length; j++) {
      const ch = text[j];
      if (inString) {
        if (escaped) escaped = false;
        else if (ch === "\\") escaped = true;
        else if (ch === '"') inString = false;
        continue;
      }
      if (ch === '"') inString = true;
      else if (ch === "{") depth++;
      else if (ch === "}") {
        depth--;
        if (depth === 0) break;
      }
    }
    if (depth !== 0) break;
    try {
      docs.push(JSON.parse(text.slice(start, j + 1)));
    } catch {
      // Not a JSON object (e.g. a brace inside prose) — keep scanning.
    }
    i = j + 1;
  }
  return docs;
}

function stringOf(x: unknown): string {
  return typeof x === "string" ? x : "";
}

function normalizeDirectoryEntry(raw: Record<string, unknown>): ProviderDirectoryEntry | null {
  const id = stringOf(raw.id);
  if (id === "") return null;
  const wireRaw = stringOf(raw.wire);
  const wire: Wire = wireRaw === "anthropic" ? "anthropic" : wireRaw === "gemini" ? "gemini" : "openai";
  const authRaw = stringOf(raw.auth);
  const auth: ProviderDirectoryEntry["auth"] =
    authRaw === "x-api-key" || authRaw === "x-goog-api-key" || authRaw === "none" ? authRaw : "bearer";
  const path = stringOf(raw.path) !== ""
    ? stringOf(raw.path)
    : stringOf(raw.base_path) !== ""
      ? stringOf(raw.base_path)
      : stringOf(raw.base_url) !== ""
        ? stringOf(raw.base_url)
        : stringOf(raw.base);
  return {
    id,
    name: stringOf(raw.name) !== "" ? stringOf(raw.name) : id,
    wire,
    auth,
    key_env: stringOf(raw.key_env),
    key: stringOf(raw.key),
    path,
  };
}

// Join a provider's base to the env URL; an absolute base is used as it stands.
export function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (b.startsWith("http://") || b.startsWith("https://")) return b.replace(/\/$/, "");
  const root = envUrl.replace(/\/$/, "");
  return root + (b.startsWith("/") ? b : "/" + b);
}

export function parseBrief(text: string): Brief {
  const docs = extractDocuments(text);
  const brief: Brief = { providers: [], ratings: [], best: [], manifest: [] };
  for (const doc of docs) {
    if (!isRecord(doc)) continue;
    if (Array.isArray(doc.providers) && doc.providers.length > 0) {
      const entries: ProviderDirectoryEntry[] = [];
      for (const raw of doc.providers) {
        if (!isRecord(raw)) continue;
        const entry = normalizeDirectoryEntry(raw);
        if (entry) entries.push(entry);
      }
      if (entries.length > 0) brief.providers = entries;
    }
    if (Array.isArray(doc.ratings) && Array.isArray(doc.best)) {
      brief.ratings = doc.ratings as unknown as RatingRow[];
      brief.best = doc.best as unknown as BestRow[];
    }
    if (Array.isArray(doc.manifest) && doc.manifest.length > 0 && isRecord(doc.manifest[0])) {
      const r0 = doc.manifest[0] as Record<string, unknown>;
      if (typeof r0.id === "string" && typeof r0.api === "string" && isRecord(r0.requires)) {
        brief.manifest = doc.manifest as unknown as ManifestRequest[];
      }
    }
  }
  return brief;
}

// ---- shared model/catalog types -------------------------------------------

export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

export interface Catalog {
  providers: CatalogProvider[];
}

export interface ModelCandidate {
  providerId: string;
  provider: ProviderDirectoryEntry;
  model: string;
  rating: number;
  contextLength: number;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}
