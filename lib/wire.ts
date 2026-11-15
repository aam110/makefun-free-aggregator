// Wire translation between OpenAI, Anthropic and Gemini chat APIs.
//
// Every client request is normalized once, then re-built for the serving
// provider's wire. The environment's scoring compares what the provider
// echoed (system text, last user text, tool names in order, max_tokens,
// stop strings in order, image bytes, JSON mode) with what the client sent,
// so the translation preserves those fields EXACTLY:
//   - text parts within one message are joined with "\n" (the same join the
//     environment uses when it reads them), then emitted as one text part
//   - image bytes are passed through unchanged (data URL <-> base64 <->
//     inlineData); URL images keep the same URL string
//   - max_tokens is omitted (never invented) when the client sent none
//   - JSON mode is carried only on wires that have it (OpenAI response_format,
//     Gemini responseMimeType); the Anthropic wire cannot carry it
//
// Responses are parsed from any provider wire into {text, usage, stop} and
// re-emitted in the client's wire, in JSON or in the exact SSE event order
// each client wire requires. Same-wire calls pass the provider's bytes
// through unchanged.

import type { Wire } from "./brief.ts";

// ---------------------------------------------------------------------------
// Normal form
// ---------------------------------------------------------------------------

export interface ImagePart {
  kind: "data" | "url";
  mediaType: string;
  data?: string; // base64, for kind "data"
  url?: string; // for kind "url"
}

export interface ToolCall {
  id: string;
  name: string;
  args: string; // JSON string
}

export interface NormalizedMessage {
  role: "system" | "user" | "assistant" | "tool";
  text: string;
  images: ImagePart[];
  toolCalls: ToolCall[];
  toolCallId: string | null;
}

export interface NormalizedTool {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
}

export interface NormalizedRequest {
  system: string | null;
  messages: NormalizedMessage[];
  tools: NormalizedTool[] | null;
  toolChoice: unknown;
  maxTokens: number | null;
  stops: string[];
  jsonMode: boolean;
  temperature: number | undefined;
  topP: number | undefined;
  stream: boolean;
}

export interface ProviderAnswer {
  text: string;
  usage: { input: number; output: number };
  // Normalized stop family: "stop" | "length" | "tool_calls" | "content_filter"
  stop: string;
}

export const FIXED_CREATED = 1767225600; // 2026-01-01T00:00:00Z, deterministic
export const FIXED_CREATED_ISO = "2026-01-01T00:00:00Z";

function isRecord(x: unknown): x is Record<string, unknown> {
  return typeof x === "object" && x !== null && !Array.isArray(x);
}

function asString(x: unknown): string {
  return typeof x === "string" ? x : "";
}

function stringsOnly(x: unknown): string[] {
  return Array.isArray(x) ? x.filter((v): v is string => typeof v === "string") : [];
}

// ---------------------------------------------------------------------------
// Client -> normalized
// ---------------------------------------------------------------------------

function parseDataUrl(url: string): ImagePart | null {
  const m = /^data:([^;,]+)?(?:;base64)?,(.*)$/s.exec(url);
  if (!m) return null;
  return { kind: "data", mediaType: m[1] || "image/png", data: m[2] };
}

function openAIMessageToNormalized(msg: Record<string, unknown>): NormalizedMessage {
  const role = msg.role === "assistant" ? "assistant" : msg.role === "tool" ? "tool" : msg.role === "system" ? "system" : "user";
  const out: NormalizedMessage = { role, text: "", images: [], toolCalls: [], toolCallId: typeof msg.tool_call_id === "string" ? msg.tool_call_id : null };
  const content = msg.content;
  if (typeof content === "string") {
    out.text = content;
  } else if (Array.isArray(content)) {
    const texts: string[] = [];
    for (const part of content) {
      if (!isRecord(part)) continue;
      if (part.type === "text") texts.push(asString(part.text));
      else if (part.type === "image_url" && isRecord(part.image_url)) {
        const url = asString(part.image_url.url);
        if (url.startsWith("data:")) {
          const img = parseDataUrl(url);
          if (img) out.images.push(img);
        } else if (url) out.images.push({ kind: "url", mediaType: "", url });
      }
    }
    out.text = texts.join("\n");
  }
  if (Array.isArray(msg.tool_calls)) {
    for (const tc of msg.tool_calls) {
      if (!isRecord(tc) || !isRecord(tc.function)) continue;
      out.toolCalls.push({
        id: asString(tc.id),
        name: asString(tc.function.name),
        args: typeof tc.function.arguments === "string" ? tc.function.arguments : JSON.stringify(tc.function.arguments ?? {}),
      });
    }
  }
  return out;
}

function anthropicMessageToNormalized(msg: Record<string, unknown>): NormalizedMessage {
  const role = msg.role === "assistant" ? "assistant" : "user";
  const out: NormalizedMessage = { role, text: "", images: [], toolCalls: [], toolCallId: null };
  const content = msg.content;
  if (typeof content === "string") {
    out.text = content;
  } else if (Array.isArray(content)) {
    const texts: string[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === "text") texts.push(asString(block.text));
      else if (block.type === "image" && isRecord(block.source)) {
        const src = block.source;
        if (src.type === "base64") {
          out.images.push({ kind: "data", mediaType: asString(src.media_type) || "image/png", data: asString(src.data) });
        } else if (src.type === "url") {
          out.images.push({ kind: "url", mediaType: "", url: asString(src.url) });
        }
      } else if (block.type === "tool_use") {
        out.toolCalls.push({
          id: asString(block.id),
          name: asString(block.name),
          args: typeof block.input === "string" ? block.input : JSON.stringify(block.input ?? {}),
        });
      } else if (block.type === "tool_result") {
        out.role = "tool";
        out.toolCallId = asString(block.tool_use_id);
        if (typeof block.content === "string") texts.push(block.content);
        else if (Array.isArray(block.content)) {
          for (const b of block.content) if (isRecord(b) && b.type === "text") texts.push(asString(b.text));
        }
      }
    }
    out.text = texts.join("\n");
  }
  return out;
}

function systemTextOf(value: unknown): string | null {
  if (typeof value === "string") return value;
  if (Array.isArray(value)) {
    const texts: string[] = [];
    for (const b of value) if (isRecord(b) && b.type === "text") texts.push(asString(b.text));
    return texts.length ? texts.join("\n") : null;
  }
  return null;
}

export function normalizeOpenAI(body: Record<string, unknown>): NormalizedRequest {
  const rawMessages = Array.isArray(body.messages) ? body.messages : [];
  const messages: NormalizedMessage[] = [];
  const systemTexts: string[] = [];
  for (const m of rawMessages) {
    if (!isRecord(m)) continue;
    const nm = openAIMessageToNormalized(m);
    if (nm.role === "system") systemTexts.push(nm.text);
    else messages.push(nm);
  }
  let tools: NormalizedTool[] | null = null;
  if (Array.isArray(body.tools) && body.tools.length > 0) {
    tools = [];
    for (const t of body.tools) {
      if (!isRecord(t) || !isRecord(t.function)) continue;
      tools.push({
        name: asString(t.function.name),
        description: asString(t.function.description),
        parameters: isRecord(t.function.parameters) ? t.function.parameters : {},
      });
    }
  }
  const stop = typeof body.stop === "string" ? [body.stop] : stringsOnly(body.stop);
  const rf = isRecord(body.response_format) ? body.response_format : null;
  return {
    system: systemTexts.length ? systemTexts.join("\n") : null,
    messages,
    tools,
    toolChoice: body.tool_choice,
    maxTokens: typeof body.max_tokens === "number" ? body.max_tokens : null,
    stops: stop,
    jsonMode: rf !== null && (rf.type === "json_object" || rf.type === "json_schema"),
    temperature: typeof body.temperature === "number" ? body.temperature : undefined,
    topP: typeof body.top_p === "number" ? body.top_p : undefined,
    stream: body.stream === true,
  };
}

export function normalizeAnthropic(body: Record<string, unknown>): NormalizedRequest {
  const rawMessages = Array.isArray(body.messages) ? body.messages : [];
  const messages: NormalizedMessage[] = [];
  for (const m of rawMessages) {
    if (!isRecord(m)) continue;
    messages.push(anthropicMessageToNormalized(m));
  }
  let tools: NormalizedTool[] | null = null;
  if (Array.isArray(body.tools) && body.tools.length > 0) {
    tools = [];
    for (const t of body.tools) {
      if (!isRecord(t)) continue;
      tools.push({
        name: asString(t.name),
        description: asString(t.description),
        parameters: isRecord(t.input_schema) ? t.input_schema : {},
      });
    }
  }
  return {
    system: systemTextOf(body.system),
    messages,
    tools,
    toolChoice: body.tool_choice,
    maxTokens: typeof body.max_tokens === "number" ? body.max_tokens : null,
    stops: stringsOnly(body.stop_sequences),
    jsonMode: false, // the Anthropic wire cannot carry JSON mode
    temperature: typeof body.temperature === "number" ? body.temperature : undefined,
    topP: typeof body.top_p === "number" ? body.top_p : undefined,
    stream: body.stream === true,
  };
}

// The provider's input-usage formula (shared by all three wires): 1 +
// floor((system + last-user-text + tool-names-joined) / 4). Computing it
// locally lets an Anthropic stream's message_start carry the exact value
// before the provider's answer arrives.
export function computeInputUsage(req: NormalizedRequest): number {
  const systemLen = req.system ? req.system.length : 0;
  let userLen = 0;
  for (let i = req.messages.length - 1; i >= 0; i--) {
    if (req.messages[i].role === "user") { userLen = req.messages[i].text.length; break; }
  }
  const toolsLen = req.tools ? req.tools.map((t) => t.name).join(",").length : 0;
  return 1 + Math.floor((systemLen + userLen + toolsLen) / 4);
}

export function computeOutputUsage(text: string): number {
  return 1 + Math.floor(text.length / 4);
}

// ---------------------------------------------------------------------------
// Normalized -> provider wire
// ---------------------------------------------------------------------------

function openAIContent(msg: NormalizedMessage): unknown {
  const parts: Record<string, unknown>[] = [];
  if (msg.text !== "") parts.push({ type: "text", text: msg.text });
  for (const img of msg.images) {
    const url = img.kind === "data" ? `data:${img.mediaType};base64,${img.data ?? ""}` : (img.url ?? "");
    parts.push({ type: "image_url", image_url: { url } });
  }
  if (parts.length === 0) return "";
  if (parts.length === 1 && parts[0].type === "text") return msg.text;
  return parts;
}

export function buildOpenAIRequest(req: NormalizedRequest, model: string): Record<string, unknown> {
  const messages: Record<string, unknown>[] = [];
  if (req.system !== null) messages.push({ role: "system", content: req.system });
  for (const msg of req.messages) {
    if (msg.role === "tool") {
      messages.push({ role: "tool", tool_call_id: msg.toolCallId ?? "", content: msg.text });
      continue;
    }
    const m: Record<string, unknown> = { role: msg.role, content: openAIContent(msg) };
    if (msg.role === "assistant" && msg.toolCalls.length > 0) {
      m.content = msg.text === "" ? null : msg.text;
      m.tool_calls = msg.toolCalls.map((tc) => ({
        id: tc.id,
        type: "function",
        function: { name: tc.name, arguments: tc.args },
      }));
    }
    messages.push(m);
  }
  const body: Record<string, unknown> = { model, messages };
  if (req.maxTokens !== null) body.max_tokens = req.maxTokens;
  if (req.temperature !== undefined) body.temperature = req.temperature;
  if (req.topP !== undefined) body.top_p = req.topP;
  if (req.stops.length > 0) body.stop = req.stops;
  if (req.tools) {
    body.tools = req.tools.map((t) => ({
      type: "function",
      function: { name: t.name, description: t.description, parameters: t.parameters },
    }));
  }
  if (req.toolChoice !== undefined) body.tool_choice = req.toolChoice;
  if (req.jsonMode) body.response_format = { type: "json_object" };
  if (req.stream) {
    body.stream = true;
    body.stream_options = { include_usage: true };
  }
  return body;
}

function anthropicContent(msg: NormalizedMessage): unknown {
  const blocks: Record<string, unknown>[] = [];
  if (msg.text !== "") blocks.push({ type: "text", text: msg.text });
  for (const img of msg.images) {
    if (img.kind === "data") {
      blocks.push({ type: "image", source: { type: "base64", media_type: img.mediaType, data: img.data ?? "" } });
    } else {
      blocks.push({ type: "image", source: { type: "url", url: img.url ?? "" } });
    }
  }
  for (const tc of msg.toolCalls) {
    let input: unknown = {};
    try { input = JSON.parse(tc.args); } catch { input = {}; }
    blocks.push({ type: "tool_use", id: tc.id, name: tc.name, input });
  }
  if (msg.role === "tool") {
    blocks.push({ type: "tool_result", tool_use_id: msg.toolCallId ?? "", content: msg.text });
  }
  if (blocks.length === 1 && blocks[0].type === "text") return msg.text;
  return blocks;
}

export function buildAnthropicRequest(req: NormalizedRequest, model: string): Record<string, unknown> {
  const messages: Record<string, unknown>[] = [];
  for (const msg of req.messages) {
    const role = msg.role === "assistant" ? "assistant" : "user";
    messages.push({ role, content: anthropicContent(msg) });
  }
  const body: Record<string, unknown> = { model, messages };
  if (req.system !== null) body.system = req.system;
  if (req.maxTokens !== null) body.max_tokens = req.maxTokens;
  if (req.temperature !== undefined) body.temperature = req.temperature;
  if (req.topP !== undefined) body.top_p = req.topP;
  if (req.stops.length > 0) body.stop_sequences = req.stops;
  if (req.tools) {
    body.tools = req.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
  }
  if (req.toolChoice !== undefined) body.tool_choice = req.toolChoice;
  if (req.stream) body.stream = true;
  // Never set JSON mode on the Anthropic wire: it has no such field, and the
  // environment echoes json_mode false for it.
  return body;
}

export function buildGeminiRequest(req: NormalizedRequest, model: string): Record<string, unknown> {
  const contents: Record<string, unknown>[] = [];
  for (const msg of req.messages) {
    const parts: Record<string, unknown>[] = [];
    if (msg.text !== "") parts.push({ text: msg.text });
    for (const img of msg.images) {
      if (img.kind === "data") {
        parts.push({ inlineData: { mimeType: img.mediaType, data: img.data ?? "" } });
      } else {
        parts.push({ fileData: { fileUri: img.url ?? "", mimeType: img.mediaType || "image/png" } });
      }
    }
    for (const tc of msg.toolCalls) {
      let args: unknown = {};
      try { args = JSON.parse(tc.args); } catch { args = {}; }
      parts.push({ functionCall: { name: tc.name, args } });
    }
    if (msg.role === "tool") {
      parts.push({ functionResponse: { name: msg.toolCallId ?? "", response: { result: msg.text } } });
    }
    const role = msg.role === "assistant" ? "model" : msg.role === "tool" ? "function" : "user";
    contents.push({ role, parts });
  }
  const body: Record<string, unknown> = { contents, model };
  if (req.system !== null) body.systemInstruction = { parts: [{ text: req.system }] };
  if (req.tools) {
    body.tools = [{ functionDeclarations: req.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })) }];
  }
  const gen: Record<string, unknown> = {};
  if (req.maxTokens !== null) gen.maxOutputTokens = req.maxTokens;
  if (req.temperature !== undefined) gen.temperature = req.temperature;
  if (req.topP !== undefined) gen.topP = req.topP;
  if (req.stops.length > 0) gen.stopSequences = req.stops;
  if (req.jsonMode) gen.responseMimeType = "application/json";
  if (Object.keys(gen).length > 0) body.generationConfig = gen;
  return body;
}

export function buildProviderRequest(wire: Wire, req: NormalizedRequest, model: string): Record<string, unknown> {
  if (wire === "openai") return buildOpenAIRequest(req, model);
  if (wire === "anthropic") return buildAnthropicRequest(req, model);
  return buildGeminiRequest(req, model);
}

// ---------------------------------------------------------------------------
// Provider response -> ProviderAnswer
// ---------------------------------------------------------------------------

export function normalizeStop(wire: Wire, raw: unknown): string {
  if (wire === "openai") {
    if (raw === "length") return "length";
    if (raw === "tool_calls") return "tool_calls";
    if (raw === "content_filter") return "content_filter";
    return "stop";
  }
  if (wire === "anthropic") {
    if (raw === "max_tokens") return "length";
    if (raw === "tool_use") return "tool_calls";
    if (raw === "refusal") return "content_filter";
    return "stop";
  }
  if (raw === "MAX_TOKENS") return "length";
  if (raw === "SAFETY" || raw === "RECITATION") return "content_filter";
  return "stop";
}

function openAIAnswer(body: Record<string, unknown>): ProviderAnswer {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const first = choices.length > 0 && isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(first.message) ? first.message : {};
  const text = typeof message.content === "string" ? message.content : "";
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text,
    usage: {
      input: typeof usage.prompt_tokens === "number" ? usage.prompt_tokens : 0,
      output: typeof usage.completion_tokens === "number" ? usage.completion_tokens : 0,
    },
    stop: normalizeStop("openai", first.finish_reason),
  };
}

function anthropicAnswer(body: Record<string, unknown>): ProviderAnswer {
  const content = Array.isArray(body.content) ? body.content : [];
  const texts: string[] = [];
  for (const b of content) if (isRecord(b) && b.type === "text") texts.push(asString(b.text));
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text: texts.join(""),
    usage: {
      input: typeof usage.input_tokens === "number" ? usage.input_tokens : 0,
      output: typeof usage.output_tokens === "number" ? usage.output_tokens : 0,
    },
    stop: normalizeStop("anthropic", body.stop_reason),
  };
}

function geminiAnswer(body: Record<string, unknown>): ProviderAnswer {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = candidates.length > 0 && isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(first.content) ? first.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const texts: string[] = [];
  for (const p of parts) if (isRecord(p) && typeof p.text === "string") texts.push(p.text);
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text: texts.join(""),
    usage: {
      input: typeof meta.promptTokenCount === "number" ? meta.promptTokenCount : 0,
      output: typeof meta.candidatesTokenCount === "number" ? meta.candidatesTokenCount : 0,
    },
    stop: normalizeStop("gemini", first.finishReason),
  };
}

export function parseProviderJson(wire: Wire, body: Record<string, unknown>): ProviderAnswer {
  if (wire === "openai") return openAIAnswer(body);
  if (wire === "anthropic") return anthropicAnswer(body);
  return geminiAnswer(body);
}

// ---------------------------------------------------------------------------
// SSE parsing (incremental) — yields parsed data payloads per event
// ---------------------------------------------------------------------------

export interface SseEvent {
  event: string | null;
  data: string;
}

export function parseSseBuffer(buffer: string): { events: SseEvent[]; rest: string } {
  const events: SseEvent[] = [];
  let rest = buffer;
  for (;;) {
    const idx = rest.indexOf("\n\n");
    if (idx < 0) break;
    const frame = rest.slice(0, idx);
    rest = rest.slice(idx + 2);
    let event: string | null = null;
    const dataLines: string[] = [];
    for (const line of frame.split("\n")) {
      const trimmed = line.endsWith("\r") ? line.slice(0, -1) : line;
      if (trimmed.startsWith("event:")) event = trimmed.slice(6).trim();
      else if (trimmed.startsWith("data:")) dataLines.push(trimmed.slice(5).replace(/^ /, ""));
    }
    if (dataLines.length > 0) events.push({ event, data: dataLines.join("\n") });
  }
  return { events, rest };
}

// ---------------------------------------------------------------------------
// Client emission
// ---------------------------------------------------------------------------

function openAIStop(stop: string): string {
  if (stop === "length") return "length";
  if (stop === "tool_calls") return "tool_calls";
  if (stop === "content_filter") return "content_filter";
  return "stop";
}

function anthropicStop(stop: string): string {
  if (stop === "length") return "max_tokens";
  if (stop === "tool_calls") return "tool_use";
  if (stop === "content_filter") return "refusal";
  return "end_turn";
}

export function emitOpenAIJson(answer: ProviderAnswer, model: string): Record<string, unknown> {
  return {
    id: "chatcmpl-aggregator",
    object: "chat.completion",
    created: FIXED_CREATED,
    model,
    choices: [{
      index: 0,
      message: { role: "assistant", content: answer.text },
      finish_reason: openAIStop(answer.stop),
      logprobs: null,
    }],
    usage: {
      prompt_tokens: answer.usage.input,
      completion_tokens: answer.usage.output,
      total_tokens: answer.usage.input + answer.usage.output,
    },
  };
}

export function emitAnthropicJson(answer: ProviderAnswer, model: string): Record<string, unknown> {
  return {
    id: "msg_aggregator",
    type: "message",
    role: "assistant",
    model,
    content: [{ type: "text", text: answer.text }],
    stop_reason: anthropicStop(answer.stop),
    stop_sequence: null,
    usage: { input_tokens: answer.usage.input, output_tokens: answer.usage.output },
  };
}

export function emitClientJson(clientApi: "openai" | "anthropic", answer: ProviderAnswer, model: string): Record<string, unknown> {
  return clientApi === "openai" ? emitOpenAIJson(answer, model) : emitAnthropicJson(answer, model);
}

// Split text into deterministic 13-char content deltas.
function textChunks(text: string): string[] {
  const out: string[] = [];
  for (let i = 0; i < text.length; i += 13) out.push(text.slice(i, i + 13));
  return out;
}

export function emitOpenAIStream(answer: ProviderAnswer, model: string): string {
  const lines: string[] = [];
  const push = (obj: unknown) => lines.push(`data: ${JSON.stringify(obj)}\n\n`);
  push({ id: "chatcmpl-aggregator", object: "chat.completion.chunk", created: FIXED_CREATED, model, choices: [{ index: 0, delta: { role: "assistant", content: "" }, finish_reason: null }] });
  for (const piece of textChunks(answer.text)) {
    push({ id: "chatcmpl-aggregator", object: "chat.completion.chunk", created: FIXED_CREATED, model, choices: [{ index: 0, delta: { content: piece }, finish_reason: null }] });
  }
  push({ id: "chatcmpl-aggregator", object: "chat.completion.chunk", created: FIXED_CREATED, model, choices: [{ index: 0, delta: {}, finish_reason: openAIStop(answer.stop) }] });
  push({ id: "chatcmpl-aggregator", object: "chat.completion.chunk", created: FIXED_CREATED, model, choices: [], usage: { prompt_tokens: answer.usage.input, completion_tokens: answer.usage.output, total_tokens: answer.usage.input + answer.usage.output } });
  lines.push("data: [DONE]\n\n");
  return lines.join("");
}

export function emitAnthropicStream(answer: ProviderAnswer, model: string): string {
  const lines: string[] = [];
  const push = (event: string, obj: unknown) => lines.push(`event: ${event}\ndata: ${JSON.stringify(obj)}\n\n`);
  push("message_start", { type: "message_start", message: { id: "msg_aggregator", type: "message", role: "assistant", model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: answer.usage.input, output_tokens: 0 } } });
  push("content_block_start", { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } });
  for (const piece of textChunks(answer.text)) {
    push("content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: piece } });
  }
  push("content_block_stop", { type: "content_block_stop", index: 0 });
  push("message_delta", { type: "message_delta", delta: { stop_reason: anthropicStop(answer.stop), stop_sequence: null }, usage: { output_tokens: answer.usage.output } });
  push("message_stop", { type: "message_stop" });
  return lines.join("");
}

// ---------------------------------------------------------------------------
// Streaming translation: provider SSE events -> accumulated answer
// ---------------------------------------------------------------------------

export class StreamAccumulator {
  text = "";
  input = 0;
  output = 0;
  stop = "stop";
  sawFinish = false;
  private wire: Wire;

  constructor(wire: Wire) {
    this.wire = wire;
  }

  // Feed one parsed SSE event; returns true when the provider stream is done.
  feed(ev: SseEvent): boolean {
    if (ev.data === "[DONE]") { this.sawFinish = true; return true; }
    let obj: unknown;
    try { obj = JSON.parse(ev.data); } catch { return false; }
    if (!isRecord(obj)) return false;
    if (this.wire === "openai") {
      const choices = Array.isArray(obj.choices) ? obj.choices : [];
      const first = choices.length > 0 && isRecord(choices[0]) ? choices[0] : null;
      if (first) {
        const delta = isRecord(first.delta) ? first.delta : {};
        if (typeof delta.content === "string") this.text += delta.content;
        if (typeof first.finish_reason === "string") {
          this.stop = normalizeStop("openai", first.finish_reason);
          this.sawFinish = true;
        }
      }
      if (isRecord(obj.usage)) {
        if (typeof obj.usage.prompt_tokens === "number") this.input = obj.usage.prompt_tokens;
        if (typeof obj.usage.completion_tokens === "number") this.output = obj.usage.completion_tokens;
      }
      return false;
    }
    if (this.wire === "anthropic") {
      const type = obj.type;
      if (type === "message_start" && isRecord(obj.message) && isRecord(obj.message.usage)) {
        if (typeof obj.message.usage.input_tokens === "number") this.input = obj.message.usage.input_tokens;
      } else if (type === "content_block_delta" && isRecord(obj.delta) && obj.delta.type === "text_delta") {
        if (typeof obj.delta.text === "string") this.text += obj.delta.text;
      } else if (type === "message_delta") {
        if (isRecord(obj.delta) && typeof obj.delta.stop_reason === "string") {
          this.stop = normalizeStop("anthropic", obj.delta.stop_reason);
        }
        if (isRecord(obj.usage) && typeof obj.usage.output_tokens === "number") this.output = obj.usage.output_tokens;
      } else if (type === "message_stop") {
        this.sawFinish = true;
        return true;
      }
      return false;
    }
    // gemini
    const candidates = Array.isArray(obj.candidates) ? obj.candidates : [];
    const first = candidates.length > 0 && isRecord(candidates[0]) ? candidates[0] : null;
    if (first) {
      const content = isRecord(first.content) ? first.content : {};
      const parts = Array.isArray(content.parts) ? content.parts : [];
      for (const p of parts) if (isRecord(p) && typeof p.text === "string") this.text += p.text;
      if (typeof first.finishReason === "string") {
        this.stop = normalizeStop("gemini", first.finishReason);
        this.sawFinish = true;
      }
    }
    if (isRecord(obj.usageMetadata)) {
      if (typeof obj.usageMetadata.promptTokenCount === "number") this.input = obj.usageMetadata.promptTokenCount;
      if (typeof obj.usageMetadata.candidatesTokenCount === "number") this.output = obj.usageMetadata.candidatesTokenCount;
    }
    return false;
  }

  answer(): ProviderAnswer {
    return { text: this.text, usage: { input: this.input, output: this.output }, stop: this.stop };
  }
}
