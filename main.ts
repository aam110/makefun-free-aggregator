// Free-model aggregator entry for make.fun's "The best free AI model
// aggregator" bounty (free-model-aggregator-01).
//
// One server, two APIs (OpenAI Chat Completions + Anthropic Messages) over
// ten simulated free-tier providers. The trial gives this server a brief and
// an env, and runs it as `node main.ts` from the repository root with Node
// 22's standard library alone.
//
// Flow:
//   1. read the brief (MAKEFUN_PROMPT_FILE): provider directory, quality
//      table, workload manifest
//   2. build one catalog from the quality table (probes cross-check it) and
//      plan the whole workload: each request gets a rating-ordered candidate
//      list and a primary, under conservative per-provider planning caps
//   3. listen on 0.0.0.0 at a free port (never 8080), POST /register
//      {protocol_version: 1, port, catalog}
//   4. serve GET /v1/models + POST /v1/chat/completions + POST /v1/messages,
//      routing each request to its planned primary and failing over across
//      the candidate list on refusal (budget/outage refusals kill a provider
//      for the trial; scheduled refusals skip it for that request only)
//   5. poll GET /work every 100 ms; exit 0 when the env says done
//
// `node main.ts --list` prints the same catalog JSON and exits 0 before any
// server starts. Determinism: every choice derives from the brief and the
// providers' answers; no wall clock, no unseeded randomness anywhere.

import * as http from "node:http";
import * as fs from "node:fs";
import { Readable } from "node:stream";
import { parseBrief } from "./lib/brief.ts";
import type { Brief, Catalog, ModelCandidate } from "./lib/brief.ts";
import { buildCatalog, type BuiltCatalog } from "./lib/catalog.ts";
import {
  compareCandidates,
  isLegal,
  orderForRequest,
  planWorkload,
  ProviderState,
  requirementsOf,
  type PlannedRequest,
  type RefusalKind,
  type Requirements,
} from "./lib/router.ts";
import {
  buildProviderRequest,
  computeInputUsage,
  computeOutputUsage,
  emitAnthropicStream,
  emitClientJson,
  emitOpenAIStream,
  FIXED_CREATED,
  FIXED_CREATED_ISO,
  normalizeAnthropic,
  normalizeOpenAI,
  type NormalizedRequest,
  type ProviderAnswer,
} from "./lib/wire.ts";
import { callProviderJson, callProviderStream, callProviderStreamRaw } from "./lib/upstream.ts";

const WORK_POLL_MS = 100;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
const REGISTER_TIMEOUT_MS = 10000;

function log(msg: string): void {
  process.stderr.write(`aggregator: ${msg}\n`);
}

function loadBrief(): Brief {
  const path = process.env.MAKEFUN_PROMPT_FILE ?? "";
  if (!path) return { providers: [], ratings: [], best: [], manifest: [] };
  try {
    return parseBrief(fs.readFileSync(path, "utf8"));
  } catch (e) {
    log(`could not read brief ${path}: ${e instanceof Error ? e.message : String(e)}`);
    return { providers: [], ratings: [], best: [], manifest: [] };
  }
}

// Requirements derived from a request body, used only when the manifest has
// no row for this request (or no manifest was given at all).
function requirementsFromBody(clientApi: "openai" | "anthropic", body: Record<string, unknown>, norm: NormalizedRequest): Requirements {
  const tools = Array.isArray(body.tools) && body.tools.length > 0;
  const vision = norm.messages.some((m) => m.images.length > 0);
  const json = clientApi === "openai" ? norm.jsonMode : false;
  return { tools, vision, json, minContext: 0 };
}

interface ServerContext {
  envUrl: string;
  brief: Brief;
  built: BuiltCatalog;
  planned: PlannedRequest[];
  state: ProviderState;
  manifestIndex: number;
  // Inference requests are served one at a time (the client sends them one
  // at a time; the chain also guards against overlap).
  queue: Promise<void>;
}

function modelListResponse(ctx: ServerContext, anthropicShape: boolean): Record<string, unknown> {
  const ids = ctx.built.freeModelIds;
  if (anthropicShape) {
    return {
      data: ids.map((id) => ({
        type: "model",
        id,
        display_name: id,
        created_at: FIXED_CREATED_ISO,
      })),
      has_more: false,
      first_id: ids[0] ?? null,
      last_id: ids[ids.length - 1] ?? null,
    };
  }
  return {
    object: "list",
    data: ids.map((id) => ({
      id,
      object: "model",
      created: FIXED_CREATED,
      owned_by: "aggregator",
    })),
  };
}

function fillUsage(answer: ProviderAnswer, req: NormalizedRequest): ProviderAnswer {
  // Safety net: the provider's usage follows the same formulas, so a missing
  // value can be recomputed exactly from the request and the answer text.
  const input = answer.usage.input > 0 ? answer.usage.input : computeInputUsage(req);
  const output = answer.usage.output > 0 ? answer.usage.output : computeOutputUsage(answer.text);
  return { ...answer, usage: { input, output } };
}

function errorBody(clientApi: "openai" | "anthropic", message: string): Record<string, unknown> {
  if (clientApi === "openai") {
    return { error: { message, type: "server_error", code: "all_providers_failed" } };
  }
  return { type: "error", error: { type: "api_error", message } };
}

// Serve one inference request: route with failover, answer in the client's
// wire. Returns when the response has been fully written.
async function serveRequest(
  ctx: ServerContext,
  clientApi: "openai" | "anthropic",
  body: Record<string, unknown>,
  anthropicVersion: string | null,
  res: http.ServerResponse,
): Promise<void> {
  const index = ctx.manifestIndex++;
  const norm = clientApi === "openai" ? normalizeOpenAI(body) : normalizeAnthropic(body);
  const planned = ctx.planned[index];
  const reqs = planned ? planned.requirements : requirementsFromBody(clientApi, body, norm);
  const requestId = planned ? planned.request.id : `req-${index + 1}`;

  // Candidate list: the plan's rating-ordered list, or (no manifest row) the
  // legal candidates recomputed from the body's requirements.
  let candidates: ModelCandidate[];
  let primary: ModelCandidate | null = null;
  if (planned) {
    candidates = planned.candidates;
    primary = planned.primary;
  } else {
    candidates = ctx.built.candidates.filter((c) => isLegal(c, reqs, clientApi)).sort(compareCandidates);
    primary = candidates[0] ?? null;
  }
  const fakePlanned: PlannedRequest = planned ?? {
    request: { id: requestId, api: clientApi, stream: norm.stream, weight: 1, requires: {} },
    index,
    requirements: reqs,
    candidates,
    primary,
  };

  const refused = new Set<string>();
  const retried = new Set<string>();
  // Decide what a refusal means for this request. Returns true when the same
  // candidate should be retried immediately, false when the request moves on
  // to the next candidate. A scheduled refusal is tied to one call number, so
  // the next call to the same provider usually succeeds — retrying it keeps
  // the best rating. But never retry when the next candidate is as good or
  // better: failing over to it is worth at least as much, and a fresh
  // provider's next call usually succeeds while a retry's next call number
  // may itself be a scheduled failure (likely for small-budget providers,
  // whose fail calls are often adjacent). orderForRequest itself answers
  // what the loop will try next, so the decision always matches the loop
  // (including the budget cutoff).
  function onRefusal(candidate: ModelCandidate, kind: RefusalKind): boolean {
    if (kind === "budget" || kind === "outage") {
      ctx.state.markDead(candidate.providerId);
      refused.add(candidate.providerId);
      return false;
    }
    if (kind === "client") {
      // A 401/400 is our bug; don't hammer the provider.
      refused.add(candidate.providerId);
      return false;
    }
    const next = orderForRequest(fakePlanned, ctx.state, new Set([...refused, candidate.providerId]));
    if (next.length > 0 && next[0].rating >= candidate.rating) {
      refused.add(candidate.providerId);
      return false;
    }
    if (retried.has(candidate.providerId)) {
      refused.add(candidate.providerId);
      return false;
    }
    retried.add(candidate.providerId);
    return true;
  }
  let attempts = 0;
  let lastError = "no legal model for this request";
  for (;;) {
    const ordered = orderForRequest(fakePlanned, ctx.state, refused);
    if (ordered.length === 0) break;
    const candidate = ordered[0];
    const provider = candidate.provider;
    const providerBody = buildProviderRequest(provider.wire, norm, candidate.model);
    const sameWireStream =
      norm.stream &&
      ((clientApi === "openai" && provider.wire === "openai") ||
        (clientApi === "anthropic" && provider.wire === "anthropic"));
    attempts++;
    ctx.state.recordCall(provider.id);

    if (sameWireStream) {
      const raw = await callProviderStreamRaw(ctx.envUrl, provider, candidate.model, providerBody, anthropicVersion);
      if (!raw.ok) {
        lastError = raw.message || `provider ${provider.id} refused (${raw.status})`;
        onRefusal(candidate, raw.kind);
        continue;
      }
      // Passthrough: the provider's SSE is already valid for this client.
      res.writeHead(200, {
        "content-type": "text/event-stream",
        "cache-control": "no-cache",
        connection: "keep-alive",
      });
      const stream = raw.response?.body;
      if (stream) {
        await new Promise<void>((resolve) => {
          const nodeStream = Readable.fromWeb(stream as Parameters<typeof Readable.fromWeb>[0]);
          nodeStream.on("error", () => res.end());
          nodeStream.pipe(res);
          nodeStream.on("end", resolve);
        });
      } else {
        res.end("data: [DONE]\n\n");
      }
      log(`${requestId}: ${provider.id}/${candidate.model} (${candidate.rating}) stream passthrough, ${attempts} attempt(s)`);
      return;
    }

    const result = norm.stream
      ? await callProviderStream(ctx.envUrl, provider, candidate.model, providerBody, anthropicVersion)
      : await callProviderJson(ctx.envUrl, provider, candidate.model, providerBody, anthropicVersion);
    if (!result.ok) {
      lastError = result.message || `provider ${provider.id} refused (${result.status})`;
      onRefusal(candidate, result.kind);
      continue;
    }
    const answer = fillUsage(result.answer, norm);
    if (norm.stream) {
      const text = clientApi === "openai" ? emitOpenAIStream(answer, candidate.model) : emitAnthropicStream(answer, candidate.model);
      res.writeHead(200, {
        "content-type": "text/event-stream",
        "cache-control": "no-cache",
        connection: "keep-alive",
      });
      res.end(text);
    } else {
      const payload = emitClientJson(clientApi, answer, candidate.model);
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify(payload));
    }
    log(`${requestId}: ${provider.id}/${candidate.model} (${candidate.rating})${norm.stream ? " stream" : ""}, ${attempts} attempt(s)`);
    return;
  }

  // Every candidate refused (or none was legal): answer 502 in the client's
  // wire so the trial continues to the next request.
  log(`${requestId}: FAILED after ${attempts} attempt(s): ${lastError}`);
  res.writeHead(502, { "content-type": "application/json" });
  res.end(JSON.stringify(errorBody(clientApi, lastError)));
}

function readBody(req: http.IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    req.on("data", (chunk: Buffer) => {
      size += chunk.length;
      if (size > MAX_BODY_BYTES) {
        reject(new Error("body too large"));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });
}

async function register(envUrl: string, port: number, catalog: Catalog): Promise<boolean> {
  try {
    const res = await fetch(envUrl + "/register", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ protocol_version: 1, port, catalog }),
      signal: AbortSignal.timeout(REGISTER_TIMEOUT_MS),
    });
    if (!res.ok) {
      log(`register failed: HTTP ${res.status}`);
      return false;
    }
    const body = await res.json() as unknown;
    log(`registered (env says ${JSON.stringify(body)})`);
    return true;
  } catch (e) {
    log(`register failed: ${e instanceof Error ? e.message : String(e)}`);
    return false;
  }
}

async function pollWork(envUrl: string): Promise<void> {
  for (;;) {
    try {
      const res = await fetch(envUrl + "/work", { signal: AbortSignal.timeout(5000) });
      if (res.ok) {
        const body = await res.json() as unknown;
        if (typeof body === "object" && body !== null && (body as Record<string, unknown>).done === true) return;
      }
    } catch {
      // The env may be briefly unreachable; keep polling.
    }
    await new Promise((r) => setTimeout(r, WORK_POLL_MS));
  }
}

async function main(): Promise<number> {
  const args = process.argv.slice(2);
  const listOnly = args.includes("--list");
  const envUrl = (process.env.MAKEFUN_ENV_URL ?? "").replace(/\/$/, "");
  const brief = loadBrief();
  const built = await buildCatalog(envUrl, brief);

  if (listOnly) {
    process.stdout.write(JSON.stringify(built.catalog) + "\n");
    return 0;
  }

  const planned = planWorkload(brief, built.candidates);
  const ctx: ServerContext = {
    envUrl,
    brief,
    built,
    planned,
    state: new ProviderState(),
    manifestIndex: 0,
    queue: Promise.resolve(),
  };
  log(`brief: ${brief.providers.length} providers, ${built.candidates.length} free models, ${brief.manifest.length} requests planned`);
  for (const p of planned) {
    log(`plan ${p.request.id} (${p.request.api}${p.request.stream ? ", stream" : ""}, w${p.request.weight}): primary ${p.primary ? `${p.primary.providerId}/${p.primary.model} (${p.primary.rating})` : "none"}, ${p.candidates.length} legal`);
  }

  const server = http.createServer((req, res) => {
    void (async () => {
      try {
        const url = new URL(req.url ?? "/", "http://127.0.0.1");
        if (req.method === "GET" && url.pathname === "/v1/models") {
          const anthropicShape = req.headers["x-api-key"] !== undefined;
          res.writeHead(200, { "content-type": "application/json" });
          res.end(JSON.stringify(modelListResponse(ctx, anthropicShape)));
          return;
        }
        if (req.method === "POST" && (url.pathname === "/v1/chat/completions" || url.pathname === "/v1/messages")) {
          const clientApi = url.pathname === "/v1/messages" ? "anthropic" : "openai";
          const anthropicVersion = typeof req.headers["anthropic-version"] === "string" ? req.headers["anthropic-version"] : null;
          let text: string;
          try {
            text = await readBody(req);
          } catch {
            res.writeHead(413, { "content-type": "application/json" });
            res.end(JSON.stringify(errorBody(clientApi, "body too large")));
            return;
          }
          let body: unknown;
          try {
            body = JSON.parse(text);
          } catch {
            res.writeHead(400, { "content-type": "application/json" });
            res.end(JSON.stringify(errorBody(clientApi, "invalid JSON body")));
            return;
          }
          if (typeof body !== "object" || body === null || Array.isArray(body)) {
            res.writeHead(400, { "content-type": "application/json" });
            res.end(JSON.stringify(errorBody(clientApi, "body must be a JSON object")));
            return;
          }
          const task = ctx.queue.then(() => serveRequest(ctx, clientApi, body as Record<string, unknown>, anthropicVersion, res));
          ctx.queue = task.catch(() => {});
          await task;
          return;
        }
        res.writeHead(404, { "content-type": "application/json" });
        res.end(JSON.stringify({ error: { message: "not found", type: "not_found" } }));
      } catch (e) {
        log(`handler error: ${e instanceof Error ? e.message : String(e)}`);
        if (!res.headersSent) {
          res.writeHead(500, { "content-type": "application/json" });
          res.end(JSON.stringify({ error: { message: "internal error", type: "server_error" } }));
        } else {
          res.end();
        }
      }
    })();
  });

  const port = await new Promise<number>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "0.0.0.0", () => {
      const addr = server.address();
      resolve(typeof addr === "object" && addr !== null ? addr.port : 0);
    });
  });
  log(`listening on 0.0.0.0:${port}`);

  if (!envUrl) {
    log("MAKEFUN_ENV_URL is not set; serving without a trial env (no register, no work poll)");
    await new Promise(() => {});
    return 0;
  }
  if (!(await register(envUrl, port, built.catalog))) return 1;
  await pollWork(envUrl);
  log(`work done; ${ctx.state.totalCalls()} provider calls, exiting 0`);
  server.close();
  return 0;
}

process.on("SIGTERM", () => process.exit(0));
process.on("SIGINT", () => process.exit(0));

main().then(
  (code) => process.exit(code),
  (err) => {
    log(`fatal: ${err instanceof Error ? err.message : String(err)}`);
    process.exit(1);
  },
);
