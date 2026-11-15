// Local test suite for the free-model-aggregator-01 entry.
//
// Hosts an in-process mock of the trial environment — /register, /work and
// four simulated providers (OpenAI, Anthropic and Gemini wires; bearer,
// x-goog-api-key and no auth; budgets, a failure schedule, an outage and the
// one 402 provider) whose answers are the same ANSWER ECHO shape the real
// providers give — then drives the real entry process over HTTP with a
// twelve-request manifest built to force each behaviour:
//
//   S1  routing: the best-rated eligible free model serves
//   S2  capability filtering: tools / vision / JSON mode / min_context,
//       including the JSON-mode wire rule (never an Anthropic-wire provider)
//   S3  failover: 402 (allowance gone) and 429 (budget spent) move on
//   S4  early retry: a schedule refusal at a provider's second call is
//       retried at once and recovers the top model
//   S5  outage: a provider that refuses everything is skipped after two
//       consecutive refusals, and requests still get served
//   S6  wires: every client wire is served from every provider wire, with
//       the provider's echo matching the client's body field for field
//   S7  streaming: OpenAI SSE ends with a usage chunk and [DONE]; Anthropic
//       SSE keeps the message event order
//   S8  catalog: --list, the register body and GET /v1/models agree exactly
//   S9  determinism: the whole scenario, run twice, is byte-identical
//   S10 a paid model is never dialled, whatever its rating
//   S11 usage and stop reasons are the provider's, translated per wire
//   S12 the give-up path: when nothing legal can serve, a clean 502
//
// Run: node test/run_tests.mjs   (exit 0 = all checks passed)

import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { writeFileSync, mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const MAIN = join(ROOT, 'main.ts');

// ---------------------------------------------------------------------------
// Assertions
// ---------------------------------------------------------------------------

const results = [];
function check(name, cond, detail = '') {
  results.push({ name, pass: !!cond, detail });
  console.log(`${cond ? 'PASS' : 'FAIL'}  ${name}${cond ? '' : `  -- ${detail}`}`);
}
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);

// ---------------------------------------------------------------------------
// The providers' own field extraction, ported from the kit's env: an echo is
// what a provider parsed, and the client compares it with what it sent.
// ---------------------------------------------------------------------------

const sha8Text = (text) => createHash('sha256').update(text).digest('hex').slice(0, 8);
const sha8Bytes = (bytes) => createHash('sha256').update(bytes).digest('hex').slice(0, 8);
const isRecord = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);
const numberOrNull = (v) => (typeof v === 'number' && Number.isFinite(v) ? v : null);

function blocksText(content) {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts = [];
  for (const block of content) if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  return parts.join('\n');
}
function imageShaOfUrl(url) {
  if (typeof url !== 'string' || url === '') return null;
  const data = /^data:[^,]*;base64,(.*)$/s.exec(url);
  if (data !== null) return sha8Bytes(Buffer.from(data[1], 'base64'));
  return sha8Text(url);
}
function stringList(v) {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x) => typeof x === 'string');
}
function echoOf(o) {
  return {
    system_present: o.systemText !== '',
    system_sha8: o.systemText === '' ? null : sha8Text(o.systemText),
    user_sha8: sha8Text(o.userText),
    tools: o.tools,
    max_tokens: o.maxTokens,
    stops: o.stops,
    image_sha8: o.imageSha8,
    json_mode: o.jsonMode,
  };
}
function openAIParsed(body) {
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  const systemParts = [];
  let userText = '';
  let imageSha8 = null;
  for (const message of messages) {
    if (message.role === 'system') {
      const text = blocksText(message.content);
      if (text !== '') systemParts.push(text);
    } else if (message.role === 'user') {
      userText = blocksText(message.content);
      if (Array.isArray(message.content)) {
        for (const block of message.content) {
          if (isRecord(block) && block.type === 'image_url' && isRecord(block.image_url)) imageSha8 = imageShaOfUrl(block.image_url.url) ?? imageSha8;
        }
      }
    }
  }
  const tools = [];
  if (Array.isArray(body.tools)) for (const tool of body.tools) if (isRecord(tool) && isRecord(tool.function) && typeof tool.function.name === 'string') tools.push(tool.function.name);
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    model: typeof body.model === 'string' ? body.model : null,
    echo: echoOf({
      systemText: systemParts.join('\n'),
      userText,
      tools,
      maxTokens: numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens),
      stops: stringList(body.stop),
      imageSha8,
      jsonMode: format === 'json_object' || format === 'json_schema',
    }),
    systemText: systemParts.join('\n'),
    userText,
    stop: 'stop',
  };
}
function anthropicParsed(body) {
  const systemText = blocksText(body.system);
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  let userText = '';
  let imageSha8 = null;
  for (const message of messages) {
    if (message.role !== 'user') continue;
    userText = blocksText(message.content);
    if (Array.isArray(message.content)) {
      for (const block of message.content) {
        if (!isRecord(block) || block.type !== 'image' || !isRecord(block.source)) continue;
        if (block.source.type === 'base64' && typeof block.source.data === 'string') imageSha8 = sha8Bytes(Buffer.from(block.source.data, 'base64'));
        else if (block.source.type === 'url') imageSha8 = imageShaOfUrl(block.source.url) ?? imageSha8;
      }
    }
  }
  const tools = [];
  if (Array.isArray(body.tools)) for (const tool of body.tools) if (isRecord(tool) && typeof tool.name === 'string') tools.push(tool.name);
  return {
    model: typeof body.model === 'string' ? body.model : null,
    echo: echoOf({ systemText, userText, tools, maxTokens: numberOrNull(body.max_tokens), stops: stringList(body.stop_sequences), imageSha8, jsonMode: false }),
    systemText,
    userText,
    stop: 'end_turn',
  };
}
function geminiParsed(body, model) {
  const partsText = (parts) => (Array.isArray(parts) ? parts.filter((p) => isRecord(p) && typeof p.text === 'string').map((p) => p.text).join('\n') : '');
  const systemText = isRecord(body.systemInstruction) ? partsText(body.systemInstruction.parts) : '';
  const contents = Array.isArray(body.contents) ? body.contents.filter(isRecord) : [];
  let userText = '';
  let imageSha8 = null;
  for (const content of contents) {
    if (content.role === 'model') continue;
    userText = partsText(content.parts);
    if (Array.isArray(content.parts)) {
      for (const part of content.parts) {
        if (!isRecord(part)) continue;
        if (isRecord(part.inlineData) && typeof part.inlineData.data === 'string') imageSha8 = sha8Bytes(Buffer.from(part.inlineData.data, 'base64'));
        else if (isRecord(part.fileData) && typeof part.fileData.fileUri === 'string') imageSha8 = imageShaOfUrl(part.fileData.fileUri) ?? imageSha8;
      }
    }
  }
  const tools = [];
  if (Array.isArray(body.tools)) for (const tool of body.tools) if (isRecord(tool) && Array.isArray(tool.functionDeclarations)) for (const fn of tool.functionDeclarations) if (isRecord(fn) && typeof fn.name === 'string') tools.push(fn.name);
  const config = isRecord(body.generationConfig) ? body.generationConfig : {};
  return {
    model,
    echo: echoOf({
      systemText,
      userText,
      tools,
      maxTokens: numberOrNull(config.maxOutputTokens),
      stops: stringList(config.stopSequences),
      imageSha8,
      jsonMode: config.responseMimeType === 'application/json',
    }),
    systemText,
    userText,
    stop: 'STOP',
  };
}
function clientEcho(api, body) {
  return api === 'openai' ? openAIParsed(body).echo : anthropicParsed(body).echo;
}
function usageOf(parsed, text) {
  const prompt = parsed.systemText.length + parsed.userText.length + parsed.echo.tools.join(',').length;
  return { input: 1 + Math.floor(prompt / 4), output: 1 + Math.floor(text.length / 4) };
}

// ---------------------------------------------------------------------------
// The mock environment: four providers, their schedules, the control routes
// ---------------------------------------------------------------------------

const PNG_B64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==';

const PROVIDERS = [
  {
    id: 'alpha', wire: 'openai', auth: 'bearer', key: 'sim-alpha-0001', key_env: 'ALPHA_API_KEY',
    budget: 4, failCalls: new Set(), outageFrom: null, failureStatus: 429, failureCode: 'rate_limit_exceeded',
    models: [
      { id: 'a-top', free: true, context_length: 200000, tools: true, vision: false, json: true, rating_milli: 1475 },
      { id: 'a-vis', free: true, context_length: 131072, tools: true, vision: true, json: false, rating_milli: 1450 },
      { id: 'a-paid', free: false, context_length: 200000, tools: true, vision: true, json: true, rating_milli: 1520 },
    ],
  },
  {
    id: 'beta', wire: 'anthropic', auth: 'bearer', key: 'sim-beta-0002', key_env: 'BETA_API_KEY',
    budget: 20, failCalls: new Set([2]), outageFrom: null, failureStatus: 429, failureCode: 'rate_limit_error',
    models: [
      { id: 'b-top', free: true, context_length: 204800, tools: true, vision: false, json: true, rating_milli: 1470 },
      { id: 'b-vis', free: true, context_length: 204800, tools: true, vision: true, json: false, rating_milli: 1430 },
    ],
  },
  {
    id: 'gamma', wire: 'gemini', auth: 'x-goog-api-key', key: 'sim-gamma-0003', key_env: 'GAMMA_API_KEY',
    budget: 20, failCalls: new Set(), outageFrom: 3, failureStatus: 429, failureCode: 'RESOURCE_EXHAUSTED',
    models: [{ id: 'g-top', free: true, context_length: 1048576, tools: true, vision: true, json: true, rating_milli: 1460 }],
  },
  {
    id: 'delta', wire: 'openai', auth: 'none', key: '', key_env: '',
    budget: 0, failCalls: new Set(), outageFrom: null, failureStatus: 402, failureCode: 'payment_required',
    models: [{ id: 'd-top', free: true, context_length: 300000, tools: true, vision: false, json: true, rating_milli: 1476 }],
  },
];

const MANIFEST = [
  { id: 'm01', api: 'openai', stream: false, weight: 1, requires: { json: true } },
  { id: 'm02', api: 'anthropic', stream: false, weight: 1, requires: { tools: true } },
  { id: 'm03', api: 'openai', stream: false, weight: 2, requires: { tools: true, vision: true } },
  { id: 'm04', api: 'anthropic', stream: true, weight: 2, requires: { vision: true } },
  { id: 'm05', api: 'openai', stream: true, weight: 2, requires: { vision: true } },
  { id: 'm06', api: 'openai', stream: false, weight: 1, requires: {} },
  { id: 'm07', api: 'anthropic', stream: true, weight: 2, requires: { tools: true } },
  { id: 'm08', api: 'anthropic', stream: false, weight: 1, requires: {} },
  { id: 'm09', api: 'openai', stream: false, weight: 1, requires: { min_context: 200000 } },
  { id: 'm10', api: 'anthropic', stream: false, weight: 1, requires: { json: true } },
  { id: 'm11', api: 'openai', stream: true, weight: 1, requires: {} },
  { id: 'm12', api: 'openai', stream: false, weight: 1, requires: { vision: true, min_context: 500000 } },
];

const SYSTEM_TEXT = 'You are a careful operations assistant. Follow the tool contract exactly.';
const TOOLS_OPENAI = [
  { type: 'function', function: { name: 'lookup_customer', description: 'Look up one customer.', parameters: { type: 'object', properties: { account_id: { type: 'string' } }, required: ['account_id'] } } },
  { type: 'function', function: { name: 'create_ticket', description: 'Open one ticket.', parameters: { type: 'object', properties: { subject: { type: 'string' } }, required: ['subject'] } } },
];
const TOOLS_ANTHROPIC = TOOLS_OPENAI.map((t) => ({ name: t.function.name, description: t.function.description, input_schema: t.function.parameters }));

function clientBody(entry) {
  const nonce = `nonce for ${entry.id}`;
  if (entry.api === 'openai') {
    const body = { model: 'auto', max_tokens: 231, temperature: 0.4 };
    const user = { role: 'user', content: `Do the task for ${entry.id}. ${nonce}` };
    if (entry.requires.vision) {
      user.content = [
        { type: 'text', text: `Read the attached image for ${entry.id}. ${nonce}` },
        { type: 'image_url', image_url: { url: `data:image/png;base64,${PNG_B64}` } },
      ];
    }
    body.messages = [{ role: 'system', content: SYSTEM_TEXT }, user];
    if (entry.requires.tools) {
      body.tools = TOOLS_OPENAI;
      body.tool_choice = 'auto';
    }
    if (entry.requires.json) body.response_format = { type: 'json_object' };
    if (entry.id === 'm05' || entry.id === 'm11') body.stop = ['\n\n###'];
    if (entry.stream) body.stream = true;
    return body;
  }
  const body = { model: 'auto', max_tokens: 187, temperature: 0.4, system: SYSTEM_TEXT };
  const user = { role: 'user', content: `Do the task for ${entry.id}. ${nonce}` };
  if (entry.requires.vision || entry.id === 'm02') {
    user.content = [
      { type: 'text', text: `Read the attached image for ${entry.id}. ${nonce}` },
      { type: 'image', source: { type: 'base64', media_type: 'image/png', data: PNG_B64 } },
    ];
  }
  body.messages = [user];
  if (entry.requires.tools) body.tools = TOOLS_ANTHROPIC;
  if (entry.id === 'm10') body.stop_sequences = ['\n\nHuman:'];
  if (entry.stream) body.stream = true;
  return body;
}

function briefText() {
  const directory = {
    schema: 'free-model-aggregator-01/providers/v1',
    providers: PROVIDERS.map((p) => ({ id: p.id, name: p.id, wire: p.wire, auth: p.auth, key_env: p.key_env, key: p.key, path: `/p/${p.id}` })),
  };
  const ratings = {
    schema: 'free-model-aggregator-01/ratings/v1',
    ratings: PROVIDERS.flatMap((p) => p.models.filter((m) => m.free).map((m) => ({ model: m.id, provider: p.id, rating_milli: m.rating_milli, context_length: m.context_length, tools: m.tools, vision: m.vision, json: m.json, free: true }))),
    best: [],
  };
  const workload = { schema: 'free-model-aggregator-01/workload/v1', manifest: MANIFEST };
  return `mock brief\n${JSON.stringify(directory)}\n${JSON.stringify(ratings)}\n${JSON.stringify(workload)}\n`;
}

function errorReply(wire, status, code, message, retryAfter) {
  const retry = retryAfter && status === 429 ? 1 : null;
  let body;
  if (wire === 'openai') body = { error: { message, type: 'rate_limit_error', param: null, code, ...(retry !== null ? { retry_after: retry } : {}) } };
  else if (wire === 'anthropic') body = { type: 'error', error: { type: code, message, ...(retry !== null ? { retry_after: retry } : {}) } };
  else body = { error: { code: status, message, status: code } };
  return { status, body, headers: retry !== null ? { 'retry-after': String(retry) } : {} };
}

function catalogReply(p) {
  if (p.wire === 'openai') {
    return { object: 'list', data: p.models.map((m) => ({ id: m.id, object: 'model', created: 1767225600, owned_by: p.id, free: m.free, context_length: m.context_length, tools: m.tools, vision: m.vision, json: m.json })) };
  }
  if (p.wire === 'anthropic') {
    return { data: p.models.map((m) => ({ type: 'model', id: m.id, display_name: m.id, created_at: '2026-01-01T00:00:00Z', free: m.free, context_length: m.context_length, tools: m.tools, vision: m.vision, json: m.json })), has_more: false };
  }
  return { models: p.models.map((m) => ({ name: `models/${m.id}`, displayName: m.id, inputTokenLimit: m.context_length, free: m.free, context_length: m.context_length, tools: m.tools, vision: m.vision, json: m.json })) };
}

function answerReply(p, model, text, usage) {
  if (p.wire === 'openai') {
    return { id: 'chatcmpl-sim', object: 'chat.completion', created: 1767225600, model, choices: [{ index: 0, message: { role: 'assistant', content: text }, finish_reason: 'stop', logprobs: null }], usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output } };
  }
  if (p.wire === 'anthropic') {
    return { id: 'msg_sim', type: 'message', role: 'assistant', model, content: [{ type: 'text', text }], stop_reason: 'end_turn', stop_sequence: null, usage: { input_tokens: usage.input, output_tokens: usage.output } };
  }
  return { candidates: [{ content: { parts: [{ text }], role: 'model' }, finishReason: 'STOP', index: 0 }], usageMetadata: { promptTokenCount: usage.input, candidatesTokenCount: usage.output, totalTokenCount: usage.input + usage.output }, modelVersion: model };
}

class MockEnv {
  constructor() {
    this.calls = []; // counted provider calls: {provider, model, status, usage}
    this.registerBody = null;
    this.finished = false;
    this.counts = new Map(PROVIDERS.map((p) => [p.id, 0]));
    this.server = createServer((req, res) => {
      void this.handle(req, res);
    });
  }

  listen() {
    return new Promise((resolve) => {
      this.server.listen(0, '127.0.0.1', () => resolve(this.server.address().port));
    });
  }

  async readBody(req) {
    const chunks = [];
    for await (const chunk of req) chunks.push(Buffer.from(chunk));
    return Buffer.concat(chunks).toString('utf8');
  }

  send(res, status, body, headers = {}) {
    const text = typeof body === 'string' ? body : JSON.stringify(body);
    res.writeHead(status, { 'content-type': 'application/json', ...headers });
    res.end(text);
  }

  async handle(req, res) {
    const url = new URL(req.url ?? '/', 'http://mock');
    const path = url.pathname;
    if (path === '/register' && req.method === 'POST') {
      this.registerBody = JSON.parse(await this.readBody(req));
      this.send(res, 200, { ok: true, requests: MANIFEST.length });
      return;
    }
    if (path === '/work' && req.method === 'GET') {
      this.send(res, 200, this.finished ? { done: true } : { done: false, next: 'm01', pending: MANIFEST.length });
      return;
    }
    if (path === '/__finish' && req.method === 'POST') {
      this.finished = true;
      this.send(res, 200, { ok: true });
      return;
    }
    if (path.startsWith('/p/')) {
      const id = path.split('/')[2];
      const provider = PROVIDERS.find((p) => p.id === id);
      if (provider === undefined) {
        this.send(res, 404, { error: 'not_found' });
        return;
      }
      this.providerCall(provider, path.slice(`/p/${id}`.length), req, res);
      return;
    }
    this.send(res, 404, { error: 'not_found' });
  }

  authOk(p, req, url) {
    if (p.auth === 'none') return true;
    if (p.auth === 'bearer') return req.headers.authorization === `Bearer ${p.key}`;
    if (p.auth === 'x-goog-api-key') return req.headers['x-goog-api-key'] === p.key || url.searchParams.get('key') === p.key;
    if (p.auth === 'x-api-key') return req.headers['x-api-key'] === p.key;
    return false;
  }

  async providerCall(p, sub, req, res) {
    const url = new URL(req.url ?? '/', 'http://mock');
    const isCatalog = (p.wire !== 'gemini' && sub === '/v1/models' && req.method === 'GET') || (p.wire === 'gemini' && sub === '/v1beta/models' && req.method === 'GET');
    if (!this.authOk(p, req, url)) {
      const r = errorReply(p.wire, 401, 'invalid_api_key', 'the key is missing or wrong', false);
      this.send(res, r.status, r.body, r.headers);
      return;
    }
    if (isCatalog) {
      this.send(res, 200, catalogReply(p));
      return;
    }
    let body = null;
    try {
      body = JSON.parse(await this.readBody(req));
    } catch {
      body = null;
    }
    if (!isRecord(body)) {
      const r = errorReply(p.wire, 400, 'invalid_request', 'the request body is not one JSON object', false);
      this.send(res, r.status, r.body, r.headers);
      return;
    }
    let parsed;
    let model;
    if (p.wire === 'openai') {
      parsed = openAIParsed(body);
      model = parsed.model;
    } else if (p.wire === 'anthropic') {
      parsed = anthropicParsed(body);
      model = parsed.model;
    } else {
      const match = /^\/v1beta\/models\/([^/:]+):generateContent$/.exec(sub);
      model = match === null ? null : decodeURIComponent(match[1]);
      parsed = geminiParsed(body, model ?? '');
    }
    const spec = model === null ? null : (p.models.find((m) => m.id === model) ?? null);
    if (model === null || spec === null || spec.free !== true) {
      this.calls.push({ provider: p.id, model, status: 400, usage: null });
      const r = errorReply(p.wire, 400, 'invalid_model', 'unknown or paid model', false);
      this.send(res, r.status, r.body, r.headers);
      return;
    }
    const n = (this.counts.get(p.id) ?? 0) + 1;
    this.counts.set(p.id, n);
    const refused = n > p.budget ? 'budget spent' : p.outageFrom !== null && n >= p.outageFrom ? 'out' : p.failCalls.has(n) ? 'refused this call' : null;
    if (refused !== null) {
      this.calls.push({ provider: p.id, model, status: p.failureStatus, usage: null });
      const r = errorReply(p.wire, p.failureStatus, p.failureCode, refused, p.id === 'alpha');
      this.send(res, r.status, r.body, r.headers);
      return;
    }
    const text = `ANSWER ECHO${JSON.stringify(parsed.echo)}`;
    const usage = usageOf(parsed, text);
    this.calls.push({ provider: p.id, model, status: 200, usage });
    this.send(res, 200, answerReply(p, model, text, usage));
  }
}

// ---------------------------------------------------------------------------
// The driver: run the real entry against a fresh mock, drive the manifest
// ---------------------------------------------------------------------------

function sseFrames(text) {
  return text
    .replace(/\r\n/g, '\n')
    .split('\n\n')
    .filter((f) => f.trim() !== '')
    .map((frame) => {
      let event = null;
      const data = [];
      for (const line of frame.split('\n')) {
        if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
        else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
      }
      return { event, data: data.join('\n') };
    });
}

function responseText(api, stream, bodyText) {
  if (!stream) {
    const body = JSON.parse(bodyText);
    if (api === 'openai') return body.choices?.[0]?.message?.content ?? '';
    return (body.content ?? []).filter((b) => b.type === 'text').map((b) => b.text).join('');
  }
  let text = '';
  for (const frame of sseFrames(bodyText)) {
    if (frame.data.trim() === '[DONE]') continue;
    const data = JSON.parse(frame.data);
    if (api === 'openai') {
      const delta = data.choices?.[0]?.delta;
      if (typeof delta?.content === 'string') text += delta.content;
    } else if (frame.event === 'content_block_delta' && data.delta?.type === 'text_delta') {
      text += data.delta.text;
    }
  }
  return text;
}

function responseUsage(api, stream, bodyText) {
  if (!stream) {
    const body = JSON.parse(bodyText);
    if (api === 'openai') return { input: body.usage?.prompt_tokens ?? null, output: body.usage?.completion_tokens ?? null };
    return { input: body.usage?.input_tokens ?? null, output: body.usage?.output_tokens ?? null };
  }
  let usage = { input: null, output: null };
  for (const frame of sseFrames(bodyText)) {
    if (frame.data.trim() === '[DONE]') continue;
    const data = JSON.parse(frame.data);
    if (api === 'openai' && isRecord(data.usage)) usage = { input: data.usage.prompt_tokens ?? null, output: data.usage.completion_tokens ?? null };
    if (api === 'anthropic' && frame.event === 'message_start' && isRecord(data.message?.usage)) usage.input = data.message.usage.input_tokens ?? null;
    if (api === 'anthropic' && frame.event === 'message_delta' && isRecord(data.usage)) usage.output = data.usage.output_tokens ?? null;
  }
  return usage;
}

function responseStop(api, stream, bodyText) {
  if (!stream) {
    const body = JSON.parse(bodyText);
    return api === 'openai' ? (body.choices?.[0]?.finish_reason ?? null) : (body.stop_reason ?? null);
  }
  let stop = null;
  for (const frame of sseFrames(bodyText)) {
    if (frame.data.trim() === '[DONE]') continue;
    const data = JSON.parse(frame.data);
    if (api === 'openai' && typeof data.choices?.[0]?.finish_reason === 'string') stop = data.choices[0].finish_reason;
    if (api === 'anthropic' && frame.event === 'message_delta') stop = data.delta?.stop_reason ?? null;
  }
  return stop;
}

async function waitFor(fn, timeoutMs, what) {
  const start = Date.now();
  for (;;) {
    const value = await fn();
    if (value) return value;
    if (Date.now() - start > timeoutMs) throw new Error(`timed out waiting for ${what}`);
    await new Promise((r) => setTimeout(r, 25));
  }
}

async function runScenario() {
  const mock = new MockEnv();
  const port = await mock.listen();
  const envUrl = `http://127.0.0.1:${port}`;
  const dir = mkdtempSync(join(tmpdir(), 'agg01-test-'));
  const briefPath = join(dir, 'prompt.txt');
  writeFileSync(briefPath, briefText());
  const child = spawn('node', [MAIN], {
    env: { ...process.env, MAKEFUN_ENV_URL: envUrl, MAKEFUN_PROMPT_FILE: briefPath, MAKEFUN_TRIAL_SEED: 'test-seed' },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  let stderr = '';
  child.stderr.on('data', (d) => {
    stderr += d;
  });
  const exitPromise = new Promise((resolve) => child.on('exit', (code) => resolve(code)));
  const transcript = { responses: [], register: null };
  try {
    await waitFor(() => mock.registerBody !== null, 10000, 'register');
    transcript.register = mock.registerBody;
    // The entry's port is the one it registered with the env.
    const entryBase = `http://127.0.0.1:${mock.registerBody.port}`;
    for (const entry of MANIFEST) {
      const body = clientBody(entry);
      const callsBefore = mock.calls.length;
      const headers = { 'content-type': 'application/json' };
      if (entry.api === 'openai') headers.authorization = 'Bearer makefun-client';
      else {
        headers['x-api-key'] = 'makefun-client';
        headers['anthropic-version'] = '2023-06-01';
      }
      if (entry.stream) headers.accept = 'text/event-stream';
      const res = await fetch(`${entryBase}${entry.api === 'openai' ? '/v1/chat/completions' : '/v1/messages'}`, {
        method: 'POST',
        headers,
        body: JSON.stringify(body),
      });
      const text = await res.text();
      transcript.responses.push({ id: entry.id, status: res.status, body: text, calls: mock.calls.slice(callsBefore), sent: body });
    }
    transcript.modelsOpenAI = await fetchModels(entryBase, { authorization: 'Bearer makefun-client' });
    transcript.modelsAnthropic = await fetchModels(entryBase, { 'x-api-key': 'makefun-client', 'anthropic-version': '2023-06-01' });
    await fetch(`${envUrl}/__finish`, { method: 'POST' });
    const code = await Promise.race([exitPromise, new Promise((r) => setTimeout(() => r('timeout'), 10000))]);
    transcript.exitCode = code;
    transcript.stderr = stderr;
    return { transcript, mock };
  } finally {
    child.kill('SIGKILL');
    mock.server.close();
  }
}

async function fetchModels(base, headers) {
  try {
    const res = await fetch(`${base}/v1/models`, { headers });
    return { status: res.status, body: await res.text() };
  } catch (err) {
    return { status: 0, body: String(err) };
  }
}

async function runList(envUrl, briefPath) {
  const child = spawn('node', [MAIN, '--list'], {
    env: { ...process.env, MAKEFUN_ENV_URL: envUrl, MAKEFUN_PROMPT_FILE: briefPath },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  let out = '';
  child.stdout.on('data', (d) => {
    out += d;
  });
  const code = await new Promise((resolve) => child.on('exit', resolve));
  return { code, out };
}

// ---------------------------------------------------------------------------
// The scenario's expectations
// ---------------------------------------------------------------------------

const EXPECTED_SERVER = {
  m01: { provider: 'alpha', model: 'a-top' },
  m02: { provider: 'alpha', model: 'a-top' },
  m03: { provider: 'gamma', model: 'g-top' },
  m04: { provider: 'gamma', model: 'g-top' },
  m05: { provider: 'alpha', model: 'a-vis' },
  m06: { provider: 'alpha', model: 'a-top' },
  m07: { provider: 'beta', model: 'b-top' },
  m08: { provider: 'beta', model: 'b-top' },
  m09: { provider: 'beta', model: 'b-top' },
  m10: { provider: 'beta', model: 'b-top' },
  m11: { provider: 'beta', model: 'b-top' },
  m12: null, // nothing legal can serve it: gamma is out, the rest lack the context
};

async function main() {
  const { transcript, mock } = await runScenario();
  const byId = Object.fromEntries(transcript.responses.map((r) => [r.id, r]));

  check('entry registered and exited 0 when the workload finished', transcript.exitCode === 0, `exit=${transcript.exitCode} stderr=${transcript.stderr}`);

  // S1/S2/S3/S5/S10: who served what, and what it cost.
  for (const entry of MANIFEST) {
    const r = byId[entry.id];
    const want = EXPECTED_SERVER[entry.id];
    const successes = r.calls.filter((c) => c.status >= 200 && c.status < 300);
    if (want === null) {
      check(`S12 ${entry.id}: unservable request gets a clean 502`, r.status === 502 && successes.length === 0, `status=${r.status} calls=${JSON.stringify(r.calls)}`);
      continue;
    }
    check(
      `S1 ${entry.id}: served by ${want.provider}/${want.model}`,
      r.status === 200 && successes.length === 1 && successes[0].provider === want.provider && successes[0].model === want.model,
      `status=${r.status} calls=${JSON.stringify(r.calls)}`,
    );
    // S6: the provider's echo of the served answer matches the client's body.
    const text = responseText(entry.api, entry.stream, r.body);
    const echo = text.startsWith('ANSWER ECHO') ? JSON.parse(text.slice('ANSWER ECHO'.length)) : null;
    check(`S6 ${entry.id}: echo matches the client body field for field`, echo !== null && eq(echo, clientEcho(entry.api, r.sent)), `text=${text.slice(0, 160)}`);
    // S11: usage and stop are the serving provider's own, in the client's wire.
    const usage = responseUsage(entry.api, entry.stream, r.body);
    const served = successes[0];
    check(
      `S11 ${entry.id}: usage is the provider's`,
      served !== undefined && served.usage !== null && usage.input === served.usage.input && usage.output === served.usage.output,
      `client=${JSON.stringify(usage)} provider=${JSON.stringify(served?.usage)}`,
    );
    const stop = responseStop(entry.api, entry.stream, r.body);
    check(`S11 ${entry.id}: stop reason mapped to the client wire`, stop === (entry.api === 'openai' ? 'stop' : 'end_turn'), `stop=${stop}`);
  }

  // S10: the paid model was never dialled.
  check('S10 paid model a-paid never dialled', mock.calls.every((c) => c.model !== 'a-paid'), JSON.stringify(mock.calls.filter((c) => c.model === 'a-paid')));

  // S2: the JSON-mode request never touched the anthropic-wire provider.
  check('S2 m01 (JSON mode) never dialled beta (anthropic wire)', byId.m01.calls.every((c) => c.provider !== 'beta'), JSON.stringify(byId.m01.calls));
  // S3: the 402 provider was tried first for m01 (top rating) and failed over.
  check('S3 m01 failed over from delta\u2019s 402', byId.m01.calls.length === 3 && byId.m01.calls[0].provider === 'delta' && byId.m01.calls[0].status === 402 && byId.m01.calls[1].status === 402 && byId.m01.calls[2].provider === 'alpha', JSON.stringify(byId.m01.calls));
  // S3: budget exhaustion on alpha failed m07 over to beta.
  check('S3 m07 failed over from alpha\u2019s spent budget', byId.m07.calls.some((c) => c.provider === 'alpha' && c.status === 429), JSON.stringify(byId.m07.calls));
  // S4: beta's schedule refusal on its second call was retried at once.
  const m08providers = byId.m08.calls.map((c) => `${c.provider}:${c.status}`);
  const betaCalls = mock.calls.filter((c) => c.provider === 'beta');
  check('S4 m08 recovered b-top by retrying beta after its schedule refusal', m08providers.join(',').includes('beta:429,beta:200') || (m08providers.filter((x) => x.startsWith('beta:')).join(',') === 'beta:429,beta:200'), JSON.stringify(byId.m08.calls));
  check('S4 beta saw exactly the calls the scenario scripts', betaCalls.length === 6, `beta calls=${JSON.stringify(betaCalls)}`);
  // S5: after m05's outage refusal and m12's, gamma is skipped early: m09-m11 never dialled it.
  check('S5 gamma not dialled by m09\u2013m11 (outage, skipped)', ['m09', 'm10', 'm11'].every((id) => byId[id].calls.every((c) => c.provider !== 'gamma')), JSON.stringify(['m09', 'm10', 'm11'].map((id) => byId[id].calls)));
  // S7: stream shapes.
  const m05frames = sseFrames(byId.m05.body);
  const m05last = m05frames[m05frames.length - 1];
  const m05usage = m05frames.some((f) => {
    try {
      const d = JSON.parse(f.data);
      return isRecord(d.usage) && Array.isArray(d.choices) && d.choices.length === 0;
    } catch {
      return false;
    }
  });
  check('S7 openai stream ends with [DONE] after a usage chunk', m05last !== undefined && m05last.data.trim() === '[DONE]' && m05usage, byId.m05.body.slice(-300));
  const m04events = sseFrames(byId.m04.body).map((f) => f.event);
  const at = (name) => m04events.indexOf(name);
  check(
    'S7 anthropic stream event order',
    m04events[0] === 'message_start' &&
      m04events[m04events.length - 1] === 'message_stop' &&
      at('content_block_start') > at('message_start') &&
      at('content_block_delta') > at('content_block_start') &&
      at('content_block_stop') > at('content_block_delta') &&
      at('message_delta') > at('content_block_stop'),
    JSON.stringify(m04events),
  );

  // S8: catalog parity across --list, register and the models routes.
  const catalog = transcript.register?.catalog;
  const expectedFree = { alpha: ['a-top', 'a-vis'], beta: ['b-top', 'b-vis'], gamma: ['g-top'], delta: ['d-top'] };
  let catalogOk = catalog !== undefined && Array.isArray(catalog.providers) && catalog.providers.length === 4;
  if (catalogOk) {
    for (const p of catalog.providers) {
      const free = (p.models ?? []).filter((m) => m.free === true).map((m) => m.id).sort();
      if (!eq(free, [...(expectedFree[p.id] ?? [])].sort())) catalogOk = false;
    }
    const alpha = catalog.providers.find((p) => p.id === 'alpha');
    if (alpha === undefined || alpha.base_url !== '/p/alpha' || alpha.key_env !== 'ALPHA_API_KEY' || alpha.wire !== 'openai') catalogOk = false;
    if (!(alpha?.models ?? []).some((m) => m.id === 'a-paid' && m.free === false)) catalogOk = false;
    const gamma = catalog.providers.find((p) => p.id === 'gamma');
    if (gamma === undefined || gamma.wire !== 'gemini' || gamma.key_env !== 'GAMMA_API_KEY') catalogOk = false;
  }
  check('S8 register catalog matches the providers\u2019 true catalogs', catalogOk, JSON.stringify(catalog));
  const allFree = ['a-top', 'a-vis', 'b-top', 'b-vis', 'd-top', 'g-top'];
  const openAIList = JSON.parse(transcript.modelsOpenAI.body);
  const anthropicList = JSON.parse(transcript.modelsAnthropic.body);
  check(
    'S8 GET /v1/models lists exactly the free ids on both wires',
    eq([...new Set(openAIList.data.map((m) => m.id))].sort(), allFree) && eq([...new Set(anthropicList.data.map((m) => m.id))].sort(), allFree),
    `openai=${JSON.stringify(openAIList.data?.map((m) => m.id))} anthropic=${JSON.stringify(anthropicList.data?.map((m) => m.id))}`,
  );

  // --list against the same live mock must equal the register body.
  const mock2 = new MockEnv();
  const port2 = await mock2.listen();
  const dir = mkdtempSync(join(tmpdir(), 'agg01-list-'));
  const briefPath = join(dir, 'prompt.txt');
  writeFileSync(briefPath, briefText());
  const listed = await runList(`http://127.0.0.1:${port2}`, briefPath);
  mock2.server.close();
  check('S8 --list prints the register catalog byte for byte', listed.code === 0 && listed.out === `${JSON.stringify(catalog)}\n`, `code=${listed.code} out=${listed.out.slice(0, 200)}`);

  // S9: determinism — a second full run, byte for byte.
  const second = await runScenario();
  const sameResponses = eq(
    transcript.responses.map((r) => [r.id, r.status, r.body]),
    second.transcript.responses.map((r) => [r.id, r.status, r.body]),
  );
  check('S9 second run reproduces every response byte for byte', sameResponses && eq(transcript.register?.catalog, second.transcript.register?.catalog), 'responses or catalog differ between runs');

  const failed = results.filter((r) => !r.pass);
  console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
  if (failed.length > 0) process.exitCode = 1;
}

try {
  await main();
} catch (err) {
  console.error(`suite error: ${err.stack ?? err}`);
  process.exitCode = 1;
}
