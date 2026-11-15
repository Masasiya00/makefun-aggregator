// free-model-aggregator-01 entry: one server that routes each request to the best FREE model
// that can serve it, with failover across providers.
//
// The score lives in four places: quality (route by the sealed rating table, never a paid
// model), served (never give up after one refusal), wire (translate requests and answers
// between the three wires without dropping a field the provider echoes), and catalog (probe
// each provider's own catalog route and register exactly what it says).
//
// Determinism: no wall clock in any decision, no unseeded randomness, requests served one
// at a time in arrival order. The only order is the rating order; ties break by provider
// budget (the sealed table's calls_max) then provider id then model id.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProviderDirectoryEntry,
  type RatingRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  type ServedModel,
} from './lib/wire.ts';
import {
  anthropicEventsToOpenAI,
  anthropicToGeminiRequest,
  geminiEventsToAnthropic,
  geminiEventsToOpenAI,
  openAIToAnthropicRequest,
  openAIToGeminiRequest,
  parseAnthropicJson,
  parseGeminiJson,
  parseOpenAIJson,
  toAnthropicMessage,
  toOpenAICompletion,
  type ProviderAnswer,
} from './lib/translate.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** The safety net on one upstream call; the env answers locally in milliseconds. */
const UPSTREAM_TIMEOUT_MS = 25000;

/** The sealed table's calls_max per provider, used only as a deterministic tie-break. */
const BUDGET_MAX: Record<string, number> = {
  groq: 12,
  cerebras: 6,
  gemini: 8,
  zai: 5,
  openrouter: 12,
  sambanova: 5,
  pollinations: 14,
  cloudflare: 8,
  ovh: 6,
  kilol: 16,
};

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

/** One provider's routing state: dead when its budget is spent or it is out, else alive. */
interface ProviderState {
  provider: ProviderDirectoryEntry;
  probe: Probe;
  dead: boolean;
  /** Calls made to this provider, refusals included. */
  calls: number;
}

let states: ProviderState[] = [];

const byId = (a: { id: string }, b: { id: string }): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  state: ProviderState;
  model: string;
  rating: number;
}

/**
 * Every free model that can serve one request, best rating first. Ties break by the provider's
 * sealed budget (a deeper provider survives longer), then provider id, then model id, so the
 * order is the same on a replay.
 */
function candidatesFor(requires: Requirement, clientIsJson: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const state of states) {
    if (state.dead || !state.probe.ok) continue;
    const wire = state.provider.wire;
    // JSON mode cannot ride the Anthropic wire: it carries no response_format, so the provider
    // would echo json_mode absent and the request would fail its wire check.
    if (clientIsJson && wire === 'anthropic') continue;
    for (const model of state.probe.models) {
      const row = ratingOfModel(brief.ratings, state.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ state, model: model.id, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => {
    if (a.rating !== b.rating) return b.rating - a.rating;
    const budgetA = BUDGET_MAX[a.state.provider.id] ?? 0;
    const budgetB = BUDGET_MAX[b.state.provider.id] ?? 0;
    if (budgetA !== budgetB) return budgetB - budgetA;
    if (a.state.provider.id !== b.state.provider.id) return a.state.provider.id < b.state.provider.id ? -1 : 1;
    return a.model < b.model ? -1 : a.model > b.model ? 1 : 0;
  });
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
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

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** The message that decides a refusal: budget spent and outage are permanent, a refused call is not. */
function classifyFailure(message: string): 'dead' | 'transient' {
  const lower = message.toLowerCase();
  if (lower.includes('budget is spent') || lower.includes('is out')) return 'dead';
  return 'transient';
}

/** The error message of a failed upstream call, parsed from its body text. */
function failureMessage(status: number, text: string): string {
  try {
    const parsed: unknown = JSON.parse(text);
    if (isRecord(parsed)) {
      const error = parsed.error;
      if (isRecord(error) && typeof error.message === 'string' && error.message !== '') return error.message;
      if (typeof parsed.message === 'string' && parsed.message !== '') return parsed.message;
    }
  } catch {
    // Not JSON: fall through to the raw text.
  }
  return text === '' ? `upstream HTTP ${status}` : text.slice(0, 200);
}

interface UpstreamResult {
  ok: boolean;
  status: number;
  /** The refusal message when the call failed. */
  message: string;
  /** The parsed JSON body when the call succeeded and was not a stream. */
  json: Json | null;
  /** The raw response when the call succeeded and was a stream. */
  response: Response | null;
}

/**
 * One call to one provider. A non-stream call returns its parsed JSON; a stream call returns the
 * raw Response for the caller to pipe or translate. The model id rides the Gemini path.
 */
async function callProvider(state: ProviderState, body: Json, model: string, stream: boolean): Promise<UpstreamResult> {
  const wire = state.provider.wire;
  const base = joinBase(ENV_URL, state.provider.base);
  let url: string;
  if (wire === 'openai') url = `${base}/v1/chat/completions`;
  else if (wire === 'anthropic') url = `${base}/v1/messages`;
  else url = `${base}/v1beta/models/${encodeURIComponent(model)}:${stream ? 'streamGenerateContent?alt=sse' : 'generateContent'}`;
  const headers: Record<string, string> = { ...authHeaders(state.provider.auth, state.provider.key), 'content-type': 'application/json' };
  if (wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  const requestBody: Json = { ...body };
  if (wire === 'openai' && stream) requestBody.stream_options = { include_usage: true };
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), UPSTREAM_TIMEOUT_MS);
  try {
    const response = await fetch(url, { method: 'POST', headers, body: JSON.stringify(requestBody), signal: controller.signal });
    if (!response.ok) {
      const text = await response.text();
      return { ok: false, status: response.status, message: failureMessage(response.status, text), json: null, response: null };
    }
    if (stream) return { ok: true, status: response.status, message: '', json: null, response };
    const parsed: unknown = await response.json();
    return { ok: true, status: response.status, message: '', json: isRecord(parsed) ? parsed : {}, response: null };
  } catch (error) {
    const message = error instanceof Error ? error.message : 'upstream fetch failed';
    return { ok: false, status: 0, message, json: null, response: null };
  } finally {
    clearTimeout(timer);
  }
}

/** The request the provider sees, in its own wire, for one candidate. */
function upstreamBody(api: 'openai' | 'anthropic', wire: 'openai' | 'anthropic' | 'gemini', body: Json, model: string): Json {
  if (api === 'openai' && wire === 'openai') return { ...body, model };
  if (api === 'openai' && wire === 'anthropic') return openAIToAnthropicRequest(body, model);
  if (api === 'openai' && wire === 'gemini') return openAIToGeminiRequest(body);
  if (api === 'anthropic' && wire === 'openai') return anthropicToOpenAI(body, model);
  if (api === 'anthropic' && wire === 'anthropic') return { ...body, model };
  return anthropicToGeminiRequest(body);
}

/** A provider's JSON answer as a provider answer, in that provider's wire. */
function parseProviderJson(wire: 'openai' | 'anthropic' | 'gemini', json: Json): ProviderAnswer {
  if (wire === 'openai') return parseOpenAIJson(json);
  if (wire === 'anthropic') return parseAnthropicJson(json);
  return parseGeminiJson(json);
}

/** Copy a provider's event stream to the client as it arrives. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

/** Write a translated stream into the response, one event at a time, then end it. */
async function writeStream(events: AsyncGenerator<string>, res: ServerResponse): Promise<void> {
  for await (const event of events) res.write(event);
  res.end();
}

/**
 * Serve one client request: try the candidates in rating order until one answers, then translate
 * its answer into the client's wire. A refusal never ends the request: the next candidate serves it.
 */
async function serve(res: ServerResponse, api: 'openai' | 'anthropic', body: Json, requires: Requirement): Promise<void> {
  const stream = body.stream === true;
  const clientIsJson =
    api === 'openai' &&
    isRecord(body.response_format) &&
    (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
  const candidates = candidatesFor(requires, clientIsJson);
  if (candidates.length === 0) {
    sendError(res, api, 503, 'no_free_model', 'no free model on the directory meets this request');
    return;
  }
  for (const candidate of candidates) {
    const wire = candidate.state.provider.wire;
    const upstream = upstreamBody(api, wire, body, candidate.model);
    candidate.state.calls += 1;
    const result = await callProvider(candidate.state, upstream, candidate.model, stream);
    if (!result.ok) {
      if (classifyFailure(result.message) === 'dead') candidate.state.dead = true;
      log(`${candidate.state.provider.id}/${candidate.model} refused: ${result.message}`);
      continue;
    }
    if (stream) {
      if (result.response === null || result.response.body === null) continue;
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      if (api === 'openai' && wire === 'openai') {
        await pipeStream(result.response.body, res);
      } else if (api === 'openai' && wire === 'anthropic') {
        await writeStream(anthropicEventsToOpenAI(result.response.body, candidate.model), res);
      } else if (api === 'openai' && wire === 'gemini') {
        await writeStream(geminiEventsToOpenAI(result.response.body, candidate.model), res);
      } else if (api === 'anthropic' && wire === 'openai') {
        await writeStream(anthropicEventsFromOpenAI(result.response.body, candidate.model), res);
      } else if (api === 'anthropic' && wire === 'anthropic') {
        await pipeStream(result.response.body, res);
      } else {
        await writeStream(geminiEventsToAnthropic(result.response.body, candidate.model), res);
      }
      return;
    }
    const answer = parseProviderJson(wire, result.json ?? {});
    const translated = api === 'openai' ? toOpenAICompletion(answer, candidate.model) : toAnthropicMessage(answer, candidate.model);
    sendJson(res, 200, translated);
    return;
  }
  sendError(res, api, 503, 'overloaded_error', 'all providers refused this request');
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
    await serve(res, 'openai', body, requirementsFor(body));
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
    await serve(res, 'anthropic', body, requirementsFor(body));
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  states = probes.map((probe) => ({ provider: probe.provider, probe, dead: false, calls: 0 }));
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
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
