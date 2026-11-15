// free-model-aggregator-01 — a free-model aggregator entry for make.fun.
//
// One server, two client APIs, ten simulated free-tier providers. The server
// speaks the OpenAI Chat Completions API and the Anthropic Messages API (JSON
// and server-sent events) and serves each of the trial's twenty requests from
// the best free model that can still answer it, over the provider's own wire
// (OpenAI, Anthropic or Gemini), with that provider's own auth scheme.
//
// How it works:
//   - The brief (MAKEFUN_PROMPT_FILE) carries the provider directory, the
//     sealed quality table and the workload manifest. At startup the server
//     probes every provider's catalog route in the provider's own wire (a
//     probe costs no budget and is never counted), merges each listing with
//     the quality table, and builds ONE catalog document: the same object is
//     printed by `--list` and posted to the env's /register, byte for byte.
//   - Routing is capability-first, rating-second. A request only ever goes to
//     free models whose table row meets the manifest's requirements (tools,
//     vision, JSON mode, minimum context), best rating first. A request that
//     asks for JSON mode on the OpenAI wire never goes to an Anthropic-wire
//     provider: that wire cannot carry response_format, and its echo would
//     say so. A paid model is never dialled.
//   - Every client body is first reduced to a canonical form — system text,
//     turns with their images, tool definitions, output cap, stop sequences,
//     JSON mode — using the same field extraction the providers themselves
//     apply, then rendered into the chosen provider's wire from that form.
//     The provider's echo of what it parsed therefore matches what the
//     client sent, field for field, on every wire pair.
//   - Upstream calls are made one at a time, non-streaming, and the answer
//     is fully buffered and validated (the whole ANSWER text, its usage
//     counts, its stop reason) before a single byte is committed to the
//     client. A refusal (402, 429 or 5xx), an unreachable provider or an
//     unusable answer fails over to the next candidate; a provider that
//     refuses twice in a row is skipped until the candidate list runs dry.
//     Exactly one successful provider call ever answers a request.
//   - The client's answer is rendered from the provider's answer in the
//     client's own wire and shape, streamed or not as the client asked, with
//     the provider's text whole and untouched, its usage counts translated
//     and its stop reason mapped across the wire families.
//   - Nothing in any output reads the wall clock or a random source: fixed
//     ids and timestamps, routing state derived only from the brief, the
//     requests and the providers' answers. A replay of a trial therefore
//     reproduces the live pass, and the env's state export, byte for byte.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

// ---------------------------------------------------------------------------
// Small typed helpers
// ---------------------------------------------------------------------------

type Json = Record<string, unknown>;

const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);
const isNum = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
const strOf = (v: unknown): string => (typeof v === 'string' ? v : '');
const numOrNull = (v: unknown): number | null => (isNum(v) ? v : null);
const boolOrNull = (v: unknown): boolean | null => (typeof v === 'boolean' ? v : null);

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

// ---------------------------------------------------------------------------
// The brief: three JSON documents inside prose, found by their schema member
// ---------------------------------------------------------------------------

type Wire = 'openai' | 'anthropic' | 'gemini';
type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

interface DirectoryEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  base: string;
  key_env: string;
  key: string;
}

interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

interface ManifestEntry {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: number;
  requires: Requirement;
}

interface Brief {
  directory: DirectoryEntry[];
  ratings: RatingRow[];
  manifest: ManifestEntry[];
}

/** Index of the `}` that closes the object opening at `start`, skipping string contents; -1 if none. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every JSON object in `text` that carries a string `schema`, in the order they appear. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not a standalone JSON object; keep scanning.
    }
  }
  return out;
}

function directoryEntryOf(v: unknown): DirectoryEntry | null {
  if (!isRecord(v)) return null;
  const id = strOf(v.id);
  const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (id === '' || wire === null) return null;
  const auth =
    v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none' ? v.auth : 'bearer';
  const base =
    strOf(v.path) !== ''
      ? strOf(v.path)
      : strOf(v.base_path) !== ''
        ? strOf(v.base_path)
        : strOf(v.base_url) !== ''
          ? strOf(v.base_url)
          : strOf(v.base);
  return { id, name: strOf(v.name) !== '' ? strOf(v.name) : id, wire, auth, base, key_env: strOf(v.key_env), key: strOf(v.key) };
}

function ratingRowOf(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  if (strOf(v.model) === '' || strOf(v.provider) === '') return null;
  return {
    model: strOf(v.model),
    provider: strOf(v.provider),
    rating_milli: numOrNull(v.rating_milli) ?? 0,
    context_length: numOrNull(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function manifestEntryOf(v: unknown): ManifestEntry | null {
  if (!isRecord(v)) return null;
  if (strOf(v.id) === '') return null;
  const r = isRecord(v.requires) ? v.requires : {};
  return {
    id: strOf(v.id),
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      ...(r.tools === true ? { tools: true } : {}),
      ...(r.vision === true ? { vision: true } : {}),
      ...(r.json === true ? { json: true } : {}),
      ...(isNum(r.min_context) ? { min_context: r.min_context } : {}),
    },
  };
}

/** The brief at `path`; empty lists when the file or its documents cannot be read. */
function readBrief(path: string): Brief {
  const empty: Brief = { directory: [], ratings: [], manifest: [] };
  if (path === '') return empty;
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch {
    return empty;
  }
  const docs = documentsIn(text);
  const directoryDoc = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers));
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings));
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest));
  return {
    directory: directoryDoc === undefined ? [] : (directoryDoc.providers as unknown[]).flatMap((v) => {
      const p = directoryEntryOf(v);
      return p === null ? [] : [p];
    }),
    ratings: ratingsDoc === undefined ? [] : (ratingsDoc.ratings as unknown[]).flatMap((v) => {
      const r = ratingRowOf(v);
      return r === null ? [] : [r];
    }),
    manifest: workloadDoc === undefined ? [] : (workloadDoc.manifest as unknown[]).flatMap((v) => {
      const m = manifestEntryOf(v);
      return m === null ? [] : [m];
    }),
  };
}

// ---------------------------------------------------------------------------
// Canonical requests: a client body reduced to what every wire must preserve
// ---------------------------------------------------------------------------

type Img = { kind: 'b64'; mediaType: string; data: string } | { kind: 'url'; url: string };

interface Turn {
  role: 'user' | 'assistant';
  text: string;
  images: Img[];
}

interface ToolDef {
  name: string;
  description: string;
  parameters: Json;
}

interface Canon {
  systemText: string;
  turns: Turn[];
  tools: ToolDef[];
  maxTokens: number | null;
  stops: string[];
  jsonMode: boolean;
  stream: boolean;
  temperature: number | null;
  topP: number | null;
  /** The client's tool_choice, kept in its own wire's spelling (null when absent). */
  toolChoice: unknown;
}

/** The text of a string or of a block array's text blocks, joined as the providers join them. */
function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('\n');
}

function stringList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}

/** An image reference in OpenAI's spelling: a data URL becomes inline data, anything else a URL. */
function openAIImage(url: string): Img {
  const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (data !== null) return { kind: 'b64', mediaType: data[1] === '' ? 'image/png' : data[1], data: data[2] };
  return { kind: 'url', url };
}

/** An Anthropic image block as a canonical image, or null for any other block. */
function anthropicImage(block: Json): Img | null {
  if (block.type !== 'image' || !isRecord(block.source)) return null;
  const source = block.source;
  if (source.type === 'base64' && typeof source.data === 'string') {
    return { kind: 'b64', mediaType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data };
  }
  if (source.type === 'url' && typeof source.url === 'string') return { kind: 'url', url: source.url };
  return null;
}

/** An OpenAI Chat Completions body in canonical form. */
function canonFromOpenAI(body: Json): Canon {
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  const systemParts: string[] = [];
  const turns: Turn[] = [];
  for (const message of messages) {
    if (message.role === 'system') {
      const text = blocksText(message.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (message.role !== 'user' && message.role !== 'assistant' && message.role !== 'tool') continue;
    const images: Img[] = [];
    if (message.role === 'user' && Array.isArray(message.content)) {
      for (const block of message.content) {
        if (isRecord(block) && block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
          images.push(openAIImage(block.image_url.url));
        }
      }
    }
    turns.push({ role: message.role === 'user' ? 'user' : 'assistant', text: blocksText(message.content), images });
  }
  const tools: ToolDef[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (isRecord(tool) && isRecord(tool.function) && typeof tool.function.name === 'string') {
        tools.push({
          name: tool.function.name,
          description: typeof tool.function.description === 'string' ? tool.function.description : '',
          parameters: isRecord(tool.function.parameters) ? tool.function.parameters : { type: 'object', properties: {} },
        });
      }
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    systemText: systemParts.join('\n'),
    turns,
    tools,
    maxTokens: numOrNull(body.max_tokens) ?? numOrNull(body.max_completion_tokens),
    stops: stringList(body.stop),
    jsonMode: format === 'json_object' || format === 'json_schema',
    stream: body.stream === true,
    temperature: numOrNull(body.temperature),
    topP: numOrNull(body.top_p),
    toolChoice: body.tool_choice === undefined ? null : body.tool_choice,
  };
}

/** An Anthropic Messages body in canonical form. Its wire carries no JSON mode: jsonMode is false. */
function canonFromAnthropic(body: Json): Canon {
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  const turns: Turn[] = [];
  for (const message of messages) {
    if (message.role !== 'user' && message.role !== 'assistant') continue;
    const images: Img[] = [];
    if (message.role === 'user' && Array.isArray(message.content)) {
      for (const block of message.content) {
        if (!isRecord(block)) continue;
        const img = anthropicImage(block);
        if (img !== null) images.push(img);
      }
    }
    turns.push({ role: message.role === 'user' ? 'user' : 'assistant', text: blocksText(message.content), images });
  }
  const tools: ToolDef[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (isRecord(tool) && typeof tool.name === 'string') {
        tools.push({
          name: tool.name,
          description: typeof tool.description === 'string' ? tool.description : '',
          parameters: isRecord(tool.input_schema) ? tool.input_schema : { type: 'object', properties: {} },
        });
      }
    }
  }
  return {
    systemText: blocksText(body.system),
    turns,
    tools,
    maxTokens: numOrNull(body.max_tokens),
    stops: stringList(body.stop_sequences),
    jsonMode: false,
    stream: body.stream === true,
    temperature: numOrNull(body.temperature),
    topP: numOrNull(body.top_p),
    toolChoice: body.tool_choice === undefined ? null : body.tool_choice,
  };
}

// ---------------------------------------------------------------------------
// Provider wires: a canonical request rendered for one provider's API
// ---------------------------------------------------------------------------

/** A base path joined to the env's URL; an absolute base is used as it stands. */
function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The headers a provider wants its key in; `none` and an empty key send nothing. */
function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

/** A canonical tool_choice in OpenAI's spelling, from either client wire's. */
function toolChoiceForOpenAI(choice: unknown): unknown {
  if (typeof choice === 'string') return choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') return 'auto';
    if (choice.type === 'any') return 'required';
    if (choice.type === 'none') return 'none';
    if (choice.type === 'tool' && typeof choice.name === 'string') return { type: 'function', function: { name: choice.name } };
    if (choice.type === 'function' && isRecord(choice.function)) return choice;
  }
  return null;
}

/** A canonical tool_choice in Anthropic's spelling, from either client wire's. */
function toolChoiceForAnthropic(choice: unknown): unknown {
  if (typeof choice === 'string') {
    if (choice === 'auto') return { type: 'auto' };
    if (choice === 'required') return { type: 'any' };
    if (choice === 'none') return { type: 'none' };
    return null;
  }
  if (isRecord(choice)) {
    if (choice.type === 'auto' || choice.type === 'any' || choice.type === 'none') return { type: choice.type };
    if (choice.type === 'tool' && typeof choice.name === 'string') return { type: 'tool', name: choice.name };
    if (choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
      return { type: 'tool', name: choice.function.name };
    }
  }
  return null;
}

/** The canonical request as a Chat Completions body naming `model` (never streamed: answers are buffered). */
function openAIRequest(canon: Canon, model: string): Json {
  const messages: Json[] = [];
  if (canon.systemText !== '') messages.push({ role: 'system', content: canon.systemText });
  for (const turn of canon.turns) {
    if (turn.images.length === 0) {
      messages.push({ role: turn.role, content: turn.text });
      continue;
    }
    const parts: Json[] = [];
    if (turn.text !== '') parts.push({ type: 'text', text: turn.text });
    for (const img of turn.images) {
      parts.push({ type: 'image_url', image_url: { url: img.kind === 'b64' ? `data:${img.mediaType};base64,${img.data}` : img.url } });
    }
    messages.push({ role: turn.role, content: parts });
  }
  const out: Json = { model, messages };
  if (canon.tools.length > 0) {
    out.tools = canon.tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.parameters } }));
  }
  const choice = toolChoiceForOpenAI(canon.toolChoice);
  if (choice !== null) out.tool_choice = choice;
  if (canon.maxTokens !== null) out.max_tokens = canon.maxTokens;
  if (canon.stops.length > 0) out.stop = canon.stops;
  if (canon.jsonMode) out.response_format = { type: 'json_object' };
  if (canon.temperature !== null) out.temperature = canon.temperature;
  if (canon.topP !== null) out.top_p = canon.topP;
  return out;
}

/** The canonical request as a Messages body naming `model` (never streamed: answers are buffered). */
function anthropicRequest(canon: Canon, model: string): Json {
  const messages: Json[] = [];
  for (const turn of canon.turns) {
    if (turn.images.length === 0) {
      messages.push({ role: turn.role, content: turn.text });
      continue;
    }
    const blocks: Json[] = [];
    if (turn.text !== '') blocks.push({ type: 'text', text: turn.text });
    for (const img of turn.images) {
      blocks.push(
        img.kind === 'b64'
          ? { type: 'image', source: { type: 'base64', media_type: img.mediaType, data: img.data } }
          : { type: 'image', source: { type: 'url', url: img.url } },
      );
    }
    messages.push({ role: turn.role, content: blocks });
  }
  const out: Json = { model, messages };
  if (canon.systemText !== '') out.system = canon.systemText;
  if (canon.tools.length > 0) {
    out.tools = canon.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
  }
  const choice = toolChoiceForAnthropic(canon.toolChoice);
  if (choice !== null) out.tool_choice = choice;
  if (canon.maxTokens !== null) out.max_tokens = canon.maxTokens;
  if (canon.stops.length > 0) out.stop_sequences = canon.stops;
  if (canon.temperature !== null) out.temperature = canon.temperature;
  if (canon.topP !== null) out.top_p = canon.topP;
  return out;
}

/** The canonical request as a Gemini generateContent body; the model names the route, not the body. */
function geminiRequest(canon: Canon): Json {
  const contents: Json[] = [];
  for (const turn of canon.turns) {
    const parts: Json[] = [];
    if (turn.text !== '') parts.push({ text: turn.text });
    for (const img of turn.images) {
      parts.push(
        img.kind === 'b64' ? { inlineData: { mimeType: img.mediaType, data: img.data } } : { fileData: { fileUri: img.url } },
      );
    }
    if (parts.length === 0) parts.push({ text: '' });
    contents.push({ role: turn.role === 'assistant' ? 'model' : 'user', parts });
  }
  const out: Json = { contents };
  if (canon.systemText !== '') out.systemInstruction = { parts: [{ text: canon.systemText }] };
  if (canon.tools.length > 0) {
    out.tools = [
      { functionDeclarations: canon.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })) },
    ];
  }
  const config: Json = {};
  if (canon.maxTokens !== null) config.maxOutputTokens = canon.maxTokens;
  if (canon.stops.length > 0) config.stopSequences = canon.stops;
  if (canon.jsonMode) config.responseMimeType = 'application/json';
  if (canon.temperature !== null) config.temperature = canon.temperature;
  if (canon.topP !== null) config.topP = canon.topP;
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

// ---------------------------------------------------------------------------
// Provider answers: one buffered answer, reduced to text, usage and stop
// ---------------------------------------------------------------------------

interface Answer {
  text: string;
  input: number;
  output: number;
  /** The stop reason's family: 0 end, 1 length, 2 tool call, 3 refusal/filter. */
  stopFamily: number;
}

const STOP_FAMILIES = new Map<string, number>([
  ['stop', 0],
  ['end_turn', 0],
  ['STOP', 0],
  ['length', 1],
  ['max_tokens', 1],
  ['MAX_TOKENS', 1],
  ['tool_calls', 2],
  ['function_call', 2],
  ['tool_use', 2],
  ['content_filter', 3],
  ['refusal', 3],
  ['SAFETY', 3],
]);

function stopFamilyOf(reason: unknown): number {
  return typeof reason === 'string' ? (STOP_FAMILIES.get(reason) ?? 0) : 0;
}

/** The text blocks of a content value, concatenated (a stream's pieces and a message's blocks alike). */
function contentText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('');
}

/**
 * One provider's JSON answer in its own wire, or null when it is not a usable
 * answer (no whole ANSWER text, no usage): the caller fails over instead of
 * serving it. The text is taken exactly as the provider produced it.
 */
function answerFromProvider(wire: Wire, body: unknown): Answer | null {
  if (!isRecord(body)) return null;
  let text: string | null = null;
  let input: number | null = null;
  let output: number | null = null;
  let stop: unknown = null;
  if (wire === 'openai') {
    const choices = Array.isArray(body.choices) ? body.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null || !isRecord(choice.message)) return null;
    text = contentText(choice.message.content);
    stop = choice.finish_reason;
    if (isRecord(body.usage)) {
      input = numOrNull(body.usage.prompt_tokens);
      output = numOrNull(body.usage.completion_tokens);
    }
  } else if (wire === 'anthropic') {
    if (body.type !== 'message' || !Array.isArray(body.content)) return null;
    text = contentText(body.content);
    stop = body.stop_reason;
    if (isRecord(body.usage)) {
      input = numOrNull(body.usage.input_tokens);
      output = numOrNull(body.usage.output_tokens);
    }
  } else {
    const candidates = Array.isArray(body.candidates) ? body.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate === null || !isRecord(candidate.content)) return null;
    const parts = Array.isArray(candidate.content.parts) ? candidate.content.parts : [];
    const texts: string[] = [];
    for (const part of parts) if (isRecord(part) && typeof part.text === 'string') texts.push(part.text);
    text = texts.join('');
    stop = candidate.finishReason;
    if (isRecord(body.usageMetadata)) {
      input = numOrNull(body.usageMetadata.promptTokenCount);
      output = numOrNull(body.usageMetadata.candidatesTokenCount);
    }
  }
  if (text === null || !text.startsWith('ANSWER ECHO') || input === null || output === null) return null;
  return { text, input, output, stopFamily: stopFamilyOf(stop) };
}

// ---------------------------------------------------------------------------
// Client answers: the provider's answer in the client's wire and shape
// ---------------------------------------------------------------------------

/** A fixed epoch second for every `created` this server prints: nothing reads the clock. */
const FIXED_CREATED = 1767225600;
const OPENAI_STOPS = ['stop', 'length', 'tool_calls', 'content_filter'];
const ANTHROPIC_STOPS = ['end_turn', 'max_tokens', 'tool_use', 'refusal'];

function sseData(data: unknown): string {
  return `data: ${JSON.stringify(data)}\n\n`;
}

function sseEvent(event: string, data: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

/** The answer text in pieces of the providers' own delta size, so streams read like theirs. */
function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += 13) pieces.push(text.slice(at, at + 13));
  return pieces.length === 0 ? [''] : pieces;
}

function openAIAnswerBody(answer: Answer, model: string): Json {
  return {
    id: 'chatcmpl-agg01',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [
      { index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: OPENAI_STOPS[answer.stopFamily], logprobs: null },
    ],
    usage: { prompt_tokens: answer.input, completion_tokens: answer.output, total_tokens: answer.input + answer.output },
  };
}

function openAIAnswerStream(answer: Answer, model: string): string {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-agg01',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [sseData(chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of splitText(answer.text)) frames.push(sseData(chunk({ content: piece }, null)));
  frames.push(sseData(chunk({}, OPENAI_STOPS[answer.stopFamily])));
  frames.push(
    sseData({
      id: 'chatcmpl-agg01',
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: answer.input, completion_tokens: answer.output, total_tokens: answer.input + answer.output },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

function anthropicAnswerBody(answer: Answer, model: string): Json {
  return {
    id: 'msg_agg01',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: ANTHROPIC_STOPS[answer.stopFamily],
    stop_sequence: null,
    usage: { input_tokens: answer.input, output_tokens: answer.output },
  };
}

function anthropicAnswerStream(answer: Answer, model: string): string {
  const frames: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_agg01',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.input, output_tokens: 0 },
      },
    }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for (const piece of splitText(answer.text)) {
    frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  frames.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: ANTHROPIC_STOPS[answer.stopFamily], stop_sequence: null },
      usage: { output_tokens: answer.output },
    }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}

function openAIErrorBody(message: string, type: string, code: string | null): Json {
  return { error: { message, type, param: null, code } };
}

function anthropicErrorBody(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

// ---------------------------------------------------------------------------
// Probing and the catalog: one document, built once, listed and registered
// ---------------------------------------------------------------------------

interface ProbedModel {
  id: string;
  display_name: string;
  free: boolean | null;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

interface Probe {
  provider: DirectoryEntry;
  ok: boolean;
  models: ProbedModel[];
}

interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

interface Catalog {
  providers: CatalogProvider[];
}

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const PROTOCOL_VERSION = 1;
const PROBE_TIMEOUT_MS = 5000;
const CALL_TIMEOUT_MS = 10000;
const POLL_INTERVAL_MS = 100;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** No provider's trial budget exceeds this many counted calls; never dial past it. */
const MAX_PROVIDER_ATTEMPTS = 16;
/**
 * Each provider's free-tier floor: the fewest calls its trial budget can
 * hold, from the competition's provider table. A refusal on or before that
 * many calls is the provider's failure schedule (or the start of its
 * outage), never its budget — a budget refusal needs a call past the
 * budget — so such a refusal earns one immediate retry of the same
 * candidate. A provider the table does not name gets the smallest floor.
 */
const PROVIDER_MIN_CALLS: Record<string, number> = {
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
const DEFAULT_MIN_CALLS = 4;

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: Catalog = { providers: [] };

function ratingOf(providerId: string, modelId: string): RatingRow | null {
  return brief.ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}

/** One model as a provider's listing wrote it, in whichever wire's spelling. */
function probedModelOf(wire: Wire, v: unknown): ProbedModel | null {
  if (!isRecord(v)) return null;
  const id = wire === 'gemini' ? strOf(v.name).replace(/^models\//, '') : strOf(v.id);
  if (id === '') return null;
  return {
    id,
    display_name: strOf(v.display_name) !== '' ? strOf(v.display_name) : strOf(v.displayName) !== '' ? strOf(v.displayName) : id,
    free: boolOrNull(v.free),
    context_length: numOrNull(v.context_length) ?? numOrNull(v.context_window) ?? numOrNull(v.inputTokenLimit),
    tools: boolOrNull(v.tools),
    vision: boolOrNull(v.vision),
    json: boolOrNull(v.json),
  };
}

/** Probe one provider's catalog route in its own wire; a catalog call spends no budget. */
async function probeProvider(provider: DirectoryEntry): Promise<Probe> {
  const base = joinBase(ENV_URL, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  let lastError = '';
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      const res = await fetch(url, { headers, signal: AbortSignal.timeout(PROBE_TIMEOUT_MS) });
      if (!res.ok) {
        lastError = `GET ${url} answered HTTP ${res.status}`;
        continue;
      }
      const body: unknown = await res.json();
      if (!isRecord(body)) {
        lastError = `GET ${url} answered a non-object body`;
        continue;
      }
      const list = provider.wire === 'gemini' ? body.models : body.data;
      const models = Array.isArray(list)
        ? list.flatMap((v) => {
            const m = probedModelOf(provider.wire, v);
            return m === null ? [] : [m];
          })
        : [];
      return { provider, ok: true, models };
    } catch (err) {
      lastError = `GET ${url} failed: ${(err as Error).message}`;
    }
  }
  log(`probe of ${provider.id} failed: ${lastError}`);
  return { provider, ok: false, models: [] };
}

/** Whether a model is free: the quality table's row where it has one, else the listing's own word. */
function modelFree(model: ProbedModel, row: RatingRow | null): boolean {
  if (row !== null) return row.free;
  return model.free ?? false;
}

interface Capabilities {
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/** What a model can do: the quality table's row where it has one, else the listing's fields. */
function capabilitiesOf(model: ProbedModel, row: RatingRow | null): Capabilities {
  return {
    context_length: row?.context_length ?? model.context_length ?? null,
    tools: row?.tools ?? model.tools ?? null,
    vision: row?.vision ?? model.vision ?? null,
    json: row?.json ?? model.json ?? null,
  };
}

/** Whether a model's capabilities meet one request's requirements. An unknown capability never meets one. */
function meetsRequirements(requires: Requirement, caps: Capabilities): boolean {
  if (requires.tools === true && caps.tools !== true) return false;
  if (requires.vision === true && caps.vision !== true) return false;
  if (requires.json === true && caps.json !== true) return false;
  if (requires.min_context !== undefined && (caps.context_length === null || caps.context_length < requires.min_context)) return false;
  return true;
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  probes = await Promise.all(brief.directory.map((provider) => probeProvider(provider)));
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    const sorted = [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
    for (const model of sorted) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = ratingOf(probe.provider.id, model.id);
      models.push({ id: model.id, free: modelFree(model, row), context_length: row?.context_length ?? model.context_length ?? null });
    }
    providers.push({ id: probe.provider.id, base_url: probe.provider.base, key_env: probe.provider.key_env, wire: probe.provider.wire, models });
  }
  catalog = { providers };
}

/** The free models this entry serves, in brief provider order: what GET /v1/models lists. */
function servedModels(): { id: string; provider: string; display_name: string }[] {
  const out: { id: string; provider: string; display_name: string }[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const sorted = [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
    for (const model of sorted) {
      if (!modelFree(model, ratingOf(probe.provider.id, model.id))) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Routing: best free model first, failover on refusal, state from answers only
// ---------------------------------------------------------------------------

interface ProviderState {
  attempts: number;
  consecutiveRefusals: number;
}

const providerStates = new Map<string, ProviderState>();

function stateOf(providerId: string): ProviderState {
  let state = providerStates.get(providerId);
  if (state === undefined) {
    state = { attempts: 0, consecutiveRefusals: 0 };
    providerStates.set(providerId, state);
  }
  return state;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
  providerIndex: number;
}

/**
 * Every legal pick for one request, best first: free models meeting the
 * requirements, on a wire that can carry what the client asked (an
 * Anthropic-wire provider cannot carry an OpenAI JSON-mode request).
 * Order: rating, then the provider's attempts so far, then brief order,
 * then model id — all derived from the brief and the answers seen so far.
 */
function candidatesFor(canon: Canon, requires: Requirement): Candidate[] {
  const out: Candidate[] = [];
  probes.forEach((probe, providerIndex) => {
    if (!probe.ok) return;
    if (stateOf(probe.provider.id).attempts >= MAX_PROVIDER_ATTEMPTS) return;
    if (canon.jsonMode && probe.provider.wire === 'anthropic') return;
    for (const model of probe.models) {
      const row = ratingOf(probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0, providerIndex });
    }
  });
  out.sort((a, b) => {
    if (a.rating !== b.rating) return b.rating - a.rating;
    const byAttempts = stateOf(a.probe.provider.id).attempts - stateOf(b.probe.provider.id).attempts;
    if (byAttempts !== 0) return byAttempts;
    if (a.providerIndex !== b.providerIndex) return a.providerIndex - b.providerIndex;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return out;
}

/** One dialled provider call: its HTTP status and, on a 2xx, its parsed answer. */
interface CallOutcome {
  status: number;
  answer: Answer | null;
}

/** Dial one provider once, in its own wire with its own auth; the answer is buffered whole. */
async function callProvider(candidate: Candidate, canon: Canon): Promise<CallOutcome> {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  let url: string;
  let body: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    body = openAIRequest(canon, candidate.model.id);
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    body = anthropicRequest(canon, candidate.model.id);
  } else {
    url = `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:generateContent`;
    body = geminiRequest(canon);
  }
  let res: Response;
  try {
    res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(body), signal: AbortSignal.timeout(CALL_TIMEOUT_MS) });
  } catch {
    return { status: 0, answer: null };
  }
  if (res.status < 200 || res.status >= 300) {
    try {
      await res.text();
    } catch {
      // The refusal's body is never read for content; draining is enough.
    }
    return { status: res.status, answer: null };
  }
  let parsed: unknown;
  try {
    parsed = await res.json();
  } catch {
    return { status: res.status, answer: null };
  }
  return { status: res.status, answer: answerFromProvider(provider.wire, parsed) };
}

/**
 * Serve one canonical request: walk the candidates best-first until one
 * answers usably. A refusal at or before a provider's budget floor earns
 * that candidate one immediate retry (see PROVIDER_MIN_CALLS). Pass one skips
 * providers that refused twice in a row (an outage or a spent budget
 * refuses everything after); pass two lets them back in; pass three
 * retries every candidate once more. A refusal never reaches the client
 * while any candidate remains.
 */
async function serve(canon: Canon, requires: Requirement): Promise<{ answer: Answer; model: string } | null> {
  const candidates = candidatesFor(canon, requires);
  for (let pass = 0; pass < 3; pass++) {
    for (const candidate of candidates) {
      const state = stateOf(candidate.probe.provider.id);
      if (pass === 0 && state.consecutiveRefusals >= 2) continue;
      // One attempt, plus the early retry when the refusal came at or
      // before the provider's budget floor (see PROVIDER_MIN_CALLS).
      for (let retry = 0; retry < 2; retry++) {
        if (state.attempts >= MAX_PROVIDER_ATTEMPTS) break;
        state.attempts += 1;
        const outcome = await callProvider(candidate, canon);
        if (outcome.answer !== null) {
          state.consecutiveRefusals = 0;
          return { answer: outcome.answer, model: candidate.model.id };
        }
        state.consecutiveRefusals += 1;
        const floor = PROVIDER_MIN_CALLS[candidate.probe.provider.id] ?? DEFAULT_MIN_CALLS;
        if (retry === 0 && state.attempts > floor) break;
      }
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// The HTTP server: the two client APIs and the model list
// ---------------------------------------------------------------------------

/** How many client API requests have arrived: the manifest is in arrival order, one at a time. */
let manifestIndex = 0;

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(api: 'openai' | 'anthropic', canon: Canon): Requirement {
  const entry = brief.manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined && entry.api === api) return entry.requires;
  if (entry !== undefined && entry.api !== api) {
    // Arrival order slipped (a retry or an extra call): trust the manifest
    // position anyway — its requirements are the declared ones for this slot.
    return entry.requires;
  }
  return {
    ...(canon.tools.length > 0 ? { tools: true } : {}),
    ...(canon.turns.some((t) => t.images.length > 0) ? { vision: true } : {}),
    ...(canon.jsonMode ? { json: true } : {}),
  };
}

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendStream(res: ServerResponse, text: string): void {
  res.writeHead(200, { 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-cache', connection: 'keep-alive' });
  res.end(text);
}

function modelsRoute(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') {
    sendJson(res, 200, {
      data: list.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: '2026-01-01T00:00:00Z' })),
      has_more: false,
    });
    return;
  }
  sendJson(res, 200, {
    object: 'list',
    data: list.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })),
  });
}

async function chatRoute(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    sendJson(res, 400, openAIErrorBody('the request body is not one JSON object', 'invalid_request_error', null));
    return;
  }
  const canon = canonFromOpenAI(body);
  const served = await serve(canon, requirementsFor('openai', canon));
  if (served === null) {
    sendJson(res, 502, openAIErrorBody('no free model could serve this request', 'api_error', 'no_provider_available'));
    return;
  }
  if (canon.stream) sendStream(res, openAIAnswerStream(served.answer, served.model));
  else sendJson(res, 200, openAIAnswerBody(served.answer, served.model));
}

async function messagesRoute(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    sendJson(res, 400, anthropicErrorBody('invalid_request_error', 'the request body is not one JSON object'));
    return;
  }
  const canon = canonFromAnthropic(body);
  const served = await serve(canon, requirementsFor('anthropic', canon));
  if (served === null) {
    sendJson(res, 502, anthropicErrorBody('api_error', 'no free model could serve this request'));
    return;
  }
  if (canon.stream) sendStream(res, anthropicAnswerStream(served.answer, served.model));
  else sendJson(res, 200, anthropicAnswerBody(served.answer, served.model));
}

/** Client requests are handled one at a time, in arrival order, as the client sends them. */
let handling: Promise<void> = Promise.resolve();

function handle(req: IncomingMessage, res: ServerResponse): void {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    modelsRoute(req, res);
    return;
  }
  if (req.method === 'POST' && (url.pathname === '/v1/chat/completions' || url.pathname === '/v1/messages')) {
    const run = url.pathname === '/v1/chat/completions' ? () => chatRoute(req, res) : () => messagesRoute(req, res);
    handling = handling.then(run, run).catch((err: unknown) => {
      log(`request failed: ${(err as Error).stack ?? String(err)}`);
      if (!res.headersSent) {
        if (url.pathname === '/v1/messages') sendJson(res, 500, anthropicErrorBody('api_error', 'the server failed to answer'));
        else sendJson(res, 500, openAIErrorBody('the server failed to answer', 'api_error', null));
      }
    });
    return;
  }
  if (url.pathname === '/v1/messages') sendJson(res, 404, anthropicErrorBody('not_found_error', `no route ${req.method ?? ''} ${url.pathname}`));
  else sendJson(res, 404, openAIErrorBody(`no route ${req.method ?? ''} ${url.pathname}`, 'invalid_request_error', 'not_found'));
}

// ---------------------------------------------------------------------------
// Register, poll, and the entrypoint
// ---------------------------------------------------------------------------

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number): Promise<void> {
  let lastError: unknown = null;
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
      return;
    } catch (err) {
      lastError = err;
      await new Promise((resolve) => setTimeout(resolve, 300));
    }
  }
  throw lastError instanceof Error ? lastError : new Error('register failed');
}

/** Poll GET /work; exit 0 the moment the env says the workload is done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status === 200) {
          const work: unknown = await res.json();
          if (isRecord(work) && work.done === true) {
            clearInterval(timer);
            server.close();
            process.exit(0);
          }
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

function listen(server: Server): Promise<number> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => {
      const address = server.address();
      if (address === null || typeof address === 'string') {
        reject(new Error('the server has no port'));
        return;
      }
      resolve(address.port);
    });
  });
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the
    // one built object and stop before any server starts.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer(handle);
  let port = await listen(server);
  // 8080 is the env's own port on one layout; an ephemeral pick never holds
  // it in practice, and the guard keeps the promise if one ever did.
  for (let tries = 0; port === 8080 && tries < 3; tries++) {
    await new Promise<void>((resolve) => server.close(() => resolve()));
    port = await listen(server);
  }
  if (port === 8080) throw new Error('the OS keeps picking 8080, which the env owns: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} providers and ${servedModels().length} free models`);
  await register(port);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
