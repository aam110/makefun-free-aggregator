// Provider calls: HTTP to one simulated provider, error classification.
//
// A call either succeeds (JSON answer or a full SSE stream, accumulated into
// a ProviderAnswer) or fails with a classified refusal. Classification uses
// the error MESSAGE — the status/code alone cannot tell a spent budget from
// a transient refusal:
//   "budget is spent"  -> permanent, provider dead for the trial
//   "is out"           -> outage, permanent
//   "refused this call"-> scheduled, transient
// Retry-after headers are ignored on purpose: the environment's failures are
// call-count based, so waiting can never help. One call per attempt, 10 s
// timeout, no retries inside this module — failover lives in main.ts.

import { joinBase, type ProviderDirectoryEntry } from "./brief.ts";
import { classifyRefusal, type RefusalKind } from "./router.ts";
import { parseProviderJson, parseSseBuffer, StreamAccumulator, type ProviderAnswer } from "./wire.ts";

export const CALL_TIMEOUT_MS = 10000;

export type CallResult =
  | { ok: true; answer: ProviderAnswer }
  | { ok: false; status: number; message: string; kind: RefusalKind };

export interface RawStreamResult {
  ok: boolean;
  status: number;
  message: string;
  kind: RefusalKind;
  response: Response | null;
}

function isRecord(x: unknown): x is Record<string, unknown> {
  return typeof x === "object" && x !== null && !Array.isArray(x);
}

function authHeaders(provider: ProviderDirectoryEntry): Record<string, string> {
  if (provider.auth === "bearer") return { authorization: `Bearer ${provider.key}` };
  if (provider.auth === "x-api-key") return { "x-api-key": provider.key };
  if (provider.auth === "x-goog-api-key") return { "x-goog-api-key": provider.key };
  return {};
}

function providerUrl(envUrl: string, provider: ProviderDirectoryEntry, model: string, stream: boolean): string {
  const base = joinBase(envUrl, provider.path);
  if (provider.wire === "openai") return base + "/v1/chat/completions";
  if (provider.wire === "anthropic") return base + "/v1/messages";
  const verb = stream ? ":streamGenerateContent" : ":generateContent";
  return base + "/v1beta/models/" + encodeURIComponent(model) + verb;
}

function extractErrorMessage(bodyText: string): string {
  try {
    const parsed = JSON.parse(bodyText) as unknown;
    if (isRecord(parsed)) {
      const err = parsed.error;
      if (isRecord(err) && typeof err.message === "string") return err.message;
      if (typeof parsed.message === "string") return parsed.message;
    }
  } catch {
    // fall through to raw text
  }
  return bodyText.slice(0, 500);
}

async function postProvider(
  envUrl: string,
  provider: ProviderDirectoryEntry,
  model: string,
  body: Record<string, unknown>,
  stream: boolean,
  anthropicVersion: string | null,
): Promise<Response> {
  const headers: Record<string, string> = {
    ...authHeaders(provider),
    "content-type": "application/json",
    accept: stream ? "text/event-stream" : "application/json",
  };
  if (provider.wire === "anthropic") {
    headers["anthropic-version"] = anthropicVersion ?? "2023-06-01";
  }
  return fetch(providerUrl(envUrl, provider, model, stream), {
    method: "POST",
    headers,
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
  });
}

// Non-streaming call: full JSON answer or a classified refusal.
export async function callProviderJson(
  envUrl: string,
  provider: ProviderDirectoryEntry,
  model: string,
  body: Record<string, unknown>,
  anthropicVersion: string | null,
): Promise<CallResult> {
  let res: Response;
  try {
    res = await postProvider(envUrl, provider, model, body, false, anthropicVersion);
  } catch (e) {
    return { ok: false, status: 0, message: e instanceof Error ? e.message : String(e), kind: "scheduled" };
  }
  const text = await res.text();
  if (!res.ok) {
    const message = extractErrorMessage(text);
    return { ok: false, status: res.status, message, kind: classifyRefusal(message, res.status) };
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return { ok: false, status: 502, message: "provider returned invalid JSON", kind: "scheduled" };
  }
  if (!isRecord(parsed)) {
    return { ok: false, status: 502, message: "provider returned a non-object answer", kind: "scheduled" };
  }
  return { ok: true, answer: parseProviderJson(provider.wire, parsed) };
}

// Streaming call with translation: accumulate the provider's SSE into a
// complete answer, or a classified refusal (nothing is streamed to the
// client before the provider's status is known).
export async function callProviderStream(
  envUrl: string,
  provider: ProviderDirectoryEntry,
  model: string,
  body: Record<string, unknown>,
  anthropicVersion: string | null,
): Promise<CallResult> {
  let res: Response;
  try {
    res = await postProvider(envUrl, provider, model, body, true, anthropicVersion);
  } catch (e) {
    return { ok: false, status: 0, message: e instanceof Error ? e.message : String(e), kind: "scheduled" };
  }
  if (!res.ok) {
    const text = await res.text();
    const message = extractErrorMessage(text);
    return { ok: false, status: res.status, message, kind: classifyRefusal(message, res.status) };
  }
  const acc = new StreamAccumulator(provider.wire);
  const reader = res.body?.getReader();
  if (!reader) {
    return { ok: false, status: 502, message: "provider returned no stream body", kind: "scheduled" };
  }
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const { events, rest } = parseSseBuffer(buffer);
      buffer = rest;
      for (const ev of events) acc.feed(ev);
    }
    buffer += decoder.decode();
    const { events } = parseSseBuffer(buffer + "\n\n");
    for (const ev of events) acc.feed(ev);
  } catch (e) {
    return { ok: false, status: 502, message: e instanceof Error ? e.message : String(e), kind: "scheduled" };
  }
  return { ok: true, answer: acc.answer() };
}

// Streaming call for same-wire passthrough: returns the live Response so the
// caller can pipe its bytes to the client unchanged. On refusal the body is
// consumed here and the caller fails over (nothing piped yet).
export async function callProviderStreamRaw(
  envUrl: string,
  provider: ProviderDirectoryEntry,
  model: string,
  body: Record<string, unknown>,
  anthropicVersion: string | null,
): Promise<RawStreamResult> {
  let res: Response;
  try {
    res = await postProvider(envUrl, provider, model, body, true, anthropicVersion);
  } catch (e) {
    return { ok: false, status: 0, message: e instanceof Error ? e.message : String(e), kind: "scheduled", response: null };
  }
  if (!res.ok) {
    const text = await res.text();
    const message = extractErrorMessage(text);
    return { ok: false, status: res.status, message, kind: classifyRefusal(message, res.status), response: null };
  }
  return { ok: true, status: res.status, message: "", kind: "scheduled", response: res };
}
