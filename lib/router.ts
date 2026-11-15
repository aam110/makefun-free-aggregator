// Routing: startup planning over the workload manifest, runtime failover.
//
// The plan is built once, from the brief's quality table and the workload
// manifest, before any request arrives. Requests are assigned primaries in
// order of candidate scarcity (fewest legal models first) and then weight,
// each taking the best-rated legal model whose provider still has planning
// capacity. The planning caps are deliberately conservative: the sealed
// per-provider budget minimum minus two, reserving room for the scheduled
// failures the environment always draws (1–2 per provider, never on call 1).
//
// At runtime each request tries its planned primary first, then walks the
// full rating-ordered candidate list. Refusals are classified from the
// provider's error MESSAGE — the status/code alone cannot tell them apart:
//   "budget is spent"  -> provider is dead for the trial (never call again)
//   "is out"           -> outage, provider is dead for the trial
//   "refused this call"-> scheduled failure, transient: skip it for THIS
//                         request only, it stays usable afterwards
// Retry-after headers are a red herring (failures are call-count based), so
// nothing ever sleeps. A 401/400 spends no budget but is a bug on our side;
// the provider/model is skipped for the rest of the trial rather than retried.

import type { Brief, ManifestRequest, ModelCandidate } from "./brief.ts";

// Sealed per-provider budget minima (the trial draws each provider's budget
// uniformly in [min, max]). Once a provider has taken its minimum number of
// calls, its budget MAY be spent; probing further costs a call when it is
// (and the score's quality column is floored, so the few best-rated calls a
// generous draw would still allow are worth less than the calls saved).
// The cutoff is a soft preference, not a hard wall: when no provider with
// remaining budget is available for a request, the cutoff is lifted and the
// best exhausted provider is probed rather than failing the request.
export const CALLS_MIN: Record<string, number> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};

// Conservative planning capacity per provider: sealed calls_min − 2.
// Unknown providers get a small default (3) so the plan still works.
// Exception: a provider holding a model at the overall best rating gets an
// aggressive cap (the whole workload). It is the only way to serve requests
// at the best rating, and over-planning it is cheap: the first budget-spent
// refusal marks it dead, so at most one call is ever wasted on it, and the
// rest fail over to the next tier. Without the aggressive cap, a generous
// budget draw would leave best-rated capacity unused.
export const PLANNING_CAPS: Record<string, number> = {
  groq: 7,
  cerebras: 3,
  gemini: 4,
  zai: 2,
  openrouter: 8,
  sambanova: 2,
  pollinations: 8,
  cloudflare: 5,
  ovh: 3,
  kilol: 10,
};
const DEFAULT_PLANNING_CAP = 3;

export interface Requirements {
  tools: boolean;
  vision: boolean;
  json: boolean;
  minContext: number;
}

export function requirementsOf(manifest: ManifestRequest | undefined): Requirements {
  const r = manifest?.requires ?? {};
  return {
    tools: r.tools === true,
    vision: r.vision === true,
    json: r.json === true,
    minContext: typeof r.min_context === "number" ? r.min_context : 0,
  };
}

// A candidate is legal for a request if it is free, meets every requirement,
// and (only for an OpenAI-client JSON request) runs on a wire that carries
// JSON mode. The Anthropic wire echoes json_mode false, so an OpenAI JSON
// request may not be served by an Anthropic-wire provider. An Anthropic JSON
// request carries no JSON-mode field on its own wire, so any wire serves it.
export function isLegal(
  c: ModelCandidate,
  req: Requirements,
  clientApi: "openai" | "anthropic",
): boolean {
  if (!c.free) return false;
  if (req.tools && !c.tools) return false;
  if (req.vision && !c.vision) return false;
  if (req.json && !c.json) return false;
  if (req.minContext > 0 && c.contextLength < req.minContext) return false;
  if (clientApi === "openai" && req.json && c.provider.wire === "anthropic") return false;
  return true;
}

// Deterministic candidate order: rating desc, then provider id, then model.
export function compareCandidates(a: ModelCandidate, b: ModelCandidate): number {
  if (a.rating !== b.rating) return b.rating - a.rating;
  if (a.providerId !== b.providerId) return a.providerId < b.providerId ? -1 : 1;
  return a.model < b.model ? -1 : a.model > b.model ? 1 : 0;
}

export interface PlannedRequest {
  request: ManifestRequest;
  index: number;
  requirements: Requirements;
  // Full rating-ordered legal candidate list (primary first at runtime).
  candidates: ModelCandidate[];
  primary: ModelCandidate | null;
}

export function planWorkload(brief: Brief, candidates: ModelCandidate[]): PlannedRequest[] {
  const planned: PlannedRequest[] = brief.manifest.map((req, index) => {
    const reqs = requirementsOf(req);
    const legal = candidates.filter((c) => isLegal(c, reqs, req.api)).sort(compareCandidates);
    return { request: req, index, requirements: reqs, candidates: legal, primary: null as ModelCandidate | null };
  });

  // Assign primaries: scarcest first, then heaviest, then manifest order.
  const maxRating = candidates.reduce((m, c) => Math.max(m, c.rating), 0);
  const remaining = new Map<string, number>();
  for (const c of candidates) {
    if (!remaining.has(c.providerId)) {
      // A provider holding a best-rated model gets the whole workload: it is
      // the only source of best-rated answers, and dead-marking bounds the
      // waste of over-planning to a single call.
      const aggressive = candidates.some((d) => d.providerId === c.providerId && d.rating === maxRating);
      remaining.set(c.providerId, aggressive ? brief.manifest.length : (PLANNING_CAPS[c.providerId] ?? DEFAULT_PLANNING_CAP));
    }
  }
  const order = [...planned].sort((a, b) => {
    if (a.candidates.length !== b.candidates.length) return a.candidates.length - b.candidates.length;
    if (a.request.weight !== b.request.weight) return b.request.weight - a.request.weight;
    return a.index - b.index;
  });
  for (const p of order) {
    let pick = p.candidates.find((c) => (remaining.get(c.providerId) ?? 0) > 0);
    if (!pick) pick = p.candidates[0] ?? null;
    if (pick) {
      p.primary = pick;
      remaining.set(pick.providerId, (remaining.get(pick.providerId) ?? 0) - 1);
    }
  }
  return planned;
}

// ---- runtime provider state ------------------------------------------------

export type RefusalKind = "budget" | "outage" | "scheduled" | "client" | "unknown";

export function classifyRefusal(message: string, status: number): RefusalKind {
  if (status === 401 || status === 400) return "client";
  if (message.includes("budget is spent")) return "budget";
  if (message.includes(" is out") || message.endsWith(" is out")) return "outage";
  if (message.includes("refused this call")) return "scheduled";
  if (message.includes("concurrency")) return "scheduled";
  return "unknown";
}

export class ProviderState {
  private dead = new Set<string>();
  calls = new Map<string, number>();

  isDead(providerId: string): boolean {
    return this.dead.has(providerId);
  }
  markDead(providerId: string): void {
    this.dead.add(providerId);
  }
  recordCall(providerId: string): void {
    this.calls.set(providerId, (this.calls.get(providerId) ?? 0) + 1);
  }
  // True once the provider has taken its sealed minimum number of calls:
  // its budget may be spent. Unknown providers have no known minimum.
  isExhausted(providerId: string): boolean {
    const min = CALLS_MIN[providerId];
    if (min === undefined) return false;
    return (this.calls.get(providerId) ?? 0) >= min;
  }
  totalCalls(): number {
    let n = 0;
    for (const v of this.calls.values()) n += v;
    return n;
  }
}

// Order candidates for one request: planned primary first (if alive), then
// the rest in rating order, skipping dead providers and providers that
// already refused this request. Providers at their budget minimum are
// skipped too — unless no other candidate remains, in which case the best
// exhausted provider is probed rather than failing the request.
export function orderForRequest(
  planned: PlannedRequest,
  state: ProviderState,
  refusedThisRequest: Set<string>,
): ModelCandidate[] {
  const usable = (c: ModelCandidate) => !state.isDead(c.providerId) && !refusedThisRequest.has(c.providerId);
  let alive = planned.candidates.filter((c) => usable(c) && !state.isExhausted(c.providerId));
  if (alive.length === 0) alive = planned.candidates.filter(usable);
  const primary = planned.primary;
  if (primary && alive.includes(primary)) {
    return [primary, ...alive.filter((c) => c !== primary)];
  }
  return alive;
}
