// Cross-wire translation for free-model-aggregator-01.
//
// The starter's lib/wire.ts covers Anthropic-client -> OpenAI-provider and back. This file
// covers the rest: OpenAI-client -> Anthropic/Gemini providers, Anthropic-client -> Gemini
// provider, and every provider answer back to either client wire, in JSON and in SSE.
//
// The whole point is echo fidelity: each provider echoes what it parsed (system text, user
// text, tool names, max_tokens, stop sequences, image, JSON mode) and the env's client compares
// that echo with what it sent. A field rewritten or dropped loses the request. So every builder
// below passes the request through untouched apart from the wire's own spelling, and every
// parser returns the provider's text whole with its usage counts as that wire reports them.

import { isRecord, type Json } from './providers.ts';
import { sseEvent } from './wire.ts';

/** A parsed provider answer: its text, its usage, and its stop family (always 0 here). */
export interface ProviderAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
export function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

function stringList(value: unknown): string[] {
  if (typeof value === 'string') return [value];
  if (!Array.isArray(value)) return [];
  return value.filter((x): x is string => typeof x === 'string');
}

/** One OpenAI content part as an Anthropic content block: text or an inline image; null otherwise. */
function openAIPartToBlock(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
    if (data !== null) {
      return { type: 'image', source: { type: 'base64', media_type: data[1] !== undefined && data[1] !== '' ? data[1] : 'image/png', data: data[2] } };
    }
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null otherwise. */
function anthropicBlockToPart(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** An OpenAI message's content as Anthropic content blocks (text and images). */
function openAIContentToBlocks(content: unknown): Json[] {
  if (typeof content === 'string') return content === '' ? [] : [{ type: 'text', text: content }];
  if (!Array.isArray(content)) return [];
  const out: Json[] = [];
  for (const part of content) {
    if (!isRecord(part)) continue;
    const block = openAIPartToBlock(part);
    if (block !== null) out.push(block);
  }
  return out;
}

/** An Anthropic message's content as OpenAI content parts (text and images). */
function anthropicContentToParts(content: unknown): Json[] {
  if (typeof content === 'string') return content === '' ? [] : [{ type: 'text', text: content }];
  if (!Array.isArray(content)) return [];
  const out: Json[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    const part = anthropicBlockToPart(block);
    if (part !== null) out.push(part);
  }
  return out;
}

/** OpenAI tools as Anthropic tools: name, description and input_schema. */
function openAIToolsToAnthropic(tools: unknown): Json[] {
  if (!Array.isArray(tools)) return [];
  return tools.flatMap((raw) => {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
    return [
      {
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      },
    ];
  });
}

/** Anthropic tools as OpenAI tools: the function wrapper the Chat Completions wire wants. */
function anthropicToolsToOpenAI(tools: unknown): Json[] {
  if (!Array.isArray(tools)) return [];
  return tools.flatMap((raw) => {
    if (!isRecord(raw) || typeof raw.name !== 'string') return [];
    return [
      {
        type: 'function',
        function: {
          name: raw.name,
          description: typeof raw.description === 'string' ? raw.description : '',
          parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
        },
      },
    ];
  });
}

/** Either wire's tools as Gemini function declarations. */
function toolsToGemini(tools: unknown, from: 'openai' | 'anthropic'): Json[] {
  if (!Array.isArray(tools)) return [];
  return tools.flatMap((raw) => {
    if (!isRecord(raw)) return [];
    if (from === 'openai') {
      const fn = isRecord(raw.function) ? raw.function : null;
      if (fn === null || typeof fn.name !== 'string') return [];
      return [{ name: fn.name, description: typeof fn.description === 'string' ? fn.description : '', parameters: isRecord(fn.parameters) ? fn.parameters : { type: 'object', properties: {} } }];
    }
    if (typeof raw.name !== 'string') return [];
    return [{ name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } }];
  });
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request naming `model`.
 * response_format is dropped on purpose: the Anthropic wire cannot carry JSON mode, and the
 * router never sends a JSON-mode request down this wire (the provider would echo it absent).
 */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const out: Json = { model, messages: [] };
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  const systemParts: string[] = [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'assistant') {
      const content: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') content.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        let input: Json = {};
        if (typeof call.function.arguments === 'string') {
          try {
            const parsed: unknown = JSON.parse(call.function.arguments);
            if (isRecord(parsed)) input = parsed;
          } catch {
            // Not JSON: an empty input.
          }
        }
        content.push({ type: 'tool_use', id: typeof call.id === 'string' ? call.id : '', name: typeof call.function.name === 'string' ? call.function.name : '', input });
      }
      messages.push({ role: 'assistant', content });
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: typeof raw.content === 'string' ? raw.content : textOf(raw.content) }],
      });
      continue;
    }
    const blocks = openAIContentToBlocks(raw.content);
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  out.messages = messages;
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') out.max_tokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stringList(body.stop);
  if (stops.length > 0) out.stop_sequences = stops;
  const tools = openAIToolsToAnthropic(body.tools);
  if (tools.length > 0) out.tools = tools;
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (choice === 'none') out.tool_choice = { type: 'none' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** One message's content as Gemini parts: text, inline image data, or a file URI. */
function contentToGeminiParts(content: unknown): Json[] {
  if (typeof content === 'string') return content === '' ? [] : [{ text: content }];
  if (!Array.isArray(content)) return [];
  const out: Json[] = [];
  for (const part of content) {
    if (!isRecord(part)) continue;
    if (part.type === 'text' && typeof part.text === 'string') {
      out.push({ text: part.text });
      continue;
    }
    if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
      const url = part.image_url.url;
      const data = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
      if (data !== null) out.push({ inlineData: { mimeType: data[1] !== undefined && data[1] !== '' ? data[1] : 'image/png', data: data[2] } });
      else out.push({ fileData: { fileUri: url } });
      continue;
    }
    if (part.type === 'image' && isRecord(part.source)) {
      const source = part.source;
      if (source.type === 'base64' && typeof source.data === 'string') {
        out.push({ inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } });
      } else if (source.type === 'url' && typeof source.url === 'string') {
        out.push({ fileData: { fileUri: source.url } });
      }
    }
  }
  return out;
}

/** Gemini generationConfig from the request's cap, temperature, stops and JSON mode. */
function geminiConfig(body: Json, jsonMode: boolean): Json {
  const config: Json = {};
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  if (maxTokens !== undefined) config.maxOutputTokens = maxTokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  const stops = stringList(body.stop ?? body.stop_sequences);
  if (stops.length > 0) config.stopSequences = stops;
  if (jsonMode) config.responseMimeType = 'application/json';
  return config;
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent body. The model id goes in the
 * request path, not the body.
 */
export function openAIToGeminiRequest(body: Json): Json {
  const out: Json = {};
  const source = Array.isArray(body.messages) ? body.messages : [];
  const systemParts: string[] = [];
  const contents: Json[] = [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') parts.push({ text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        let args: Json = {};
        if (typeof call.function.arguments === 'string') {
          try {
            const parsed: unknown = JSON.parse(call.function.arguments);
            if (isRecord(parsed)) args = parsed;
          } catch {
            // Not JSON: an empty input.
          }
        }
        parts.push({ functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args } });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (raw.role === 'tool') {
      contents.push({
        role: 'function',
        parts: [{ functionResponse: { name: '', response: { result: typeof raw.content === 'string' ? raw.content : textOf(raw.content) } } }],
      });
      continue;
    }
    const parts = contentToGeminiParts(raw.content);
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  if (systemParts.length > 0) out.systemInstruction = { parts: [{ text: systemParts.join('\n') }] };
  out.contents = contents;
  const tools = toolsToGemini(body.tools, 'openai');
  if (tools.length > 0) out.tools = [{ functionDeclarations: tools }];
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  const config = geminiConfig(body, format === 'json_object' || format === 'json_schema');
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/**
 * An Anthropic Messages request as a Gemini generateContent body. The Anthropic wire carries no
 * JSON mode, so generationConfig never names one here.
 */
export function anthropicToGeminiRequest(body: Json): Json {
  const out: Json = {};
  const systemText = textOf(body.system);
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      const content = raw.content;
      if (Array.isArray(content)) {
        for (const block of content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
          else if (block.type === 'tool_use') {
            parts.push({ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } });
          }
        }
      } else if (typeof content === 'string' && content !== '') {
        parts.push({ text: content });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    const parts: Json[] = [];
    const content = raw.content;
    if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
        else if (block.type === 'image' && isRecord(block.source)) {
          const s = block.source;
          if (s.type === 'base64' && typeof s.data === 'string') {
            parts.push({ inlineData: { mimeType: typeof s.media_type === 'string' ? s.media_type : 'image/png', data: s.data } });
          } else if (s.type === 'url' && typeof s.url === 'string') {
            parts.push({ fileData: { fileUri: s.url } });
          }
        } else if (block.type === 'tool_result') {
          const text = typeof block.content === 'string' ? block.content : textOf(block.content);
          parts.push({ functionResponse: { name: typeof block.tool_use_id === 'string' ? block.tool_use_id : '', response: { result: text } } });
        }
      }
    } else if (typeof content === 'string' && content !== '') {
      parts.push({ text: content });
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  out.contents = contents;
  const tools = toolsToGemini(body.tools, 'anthropic');
  if (tools.length > 0) out.tools = [{ functionDeclarations: tools }];
  const config = geminiConfig(body, false);
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** The text of a Gemini answer's candidates, joined in order. */
export function geminiTextOf(body: Json): string {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const parts: string[] = [];
  for (const candidate of candidates) {
    if (!isRecord(candidate) || !isRecord(candidate.content) || !Array.isArray(candidate.content.parts)) continue;
    for (const part of candidate.content.parts) {
      if (isRecord(part) && typeof part.text === 'string') parts.push(part.text);
    }
  }
  return parts.join('');
}

/** A Gemini answer's usage, from its usageMetadata; zeros when it carries none. */
export function geminiUsageOf(body: Json): { input: number; output: number } {
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return { input: numberOrZero(meta.promptTokenCount), output: numberOrZero(meta.candidatesTokenCount) };
}

/** A Gemini JSON answer as a provider answer. */
export function parseGeminiJson(body: Json): ProviderAnswer {
  const usage = geminiUsageOf(body);
  return { text: geminiTextOf(body), inputTokens: usage.input, outputTokens: usage.output };
}

/** An Anthropic JSON answer as a provider answer. */
export function parseAnthropicJson(body: Json): ProviderAnswer {
  const usage = isRecord(body.usage) ? body.usage : {};
  return { text: textOf(body.content), inputTokens: numberOrZero(usage.input_tokens), outputTokens: numberOrZero(usage.output_tokens) };
}

/** An OpenAI JSON answer as a provider answer. */
export function parseOpenAIJson(body: Json): ProviderAnswer {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const usage = isRecord(body.usage) ? body.usage : {};
  return { text: typeof message.content === 'string' ? message.content : textOf(message.content), inputTokens: numberOrZero(usage.prompt_tokens), outputTokens: numberOrZero(usage.completion_tokens) };
}

/** A provider answer as an OpenAI Chat Completions object (finish_reason "stop"). */
export function toOpenAICompletion(answer: ProviderAnswer, model: string): Json {
  return {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion',
    created: 1767225600,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: 'stop', logprobs: null }],
    usage: { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens },
  };
}

/** A provider answer as an Anthropic Messages object (stop_reason "end_turn"). */
export function toAnthropicMessage(answer: ProviderAnswer, model: string): Json {
  return {
    id: 'msg_aggregate',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: 'end_turn',
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

/** One SSE frame: its event name (null when the line has none) and its data payload. */
export interface SseFrame {
  event: string | null;
  data: string;
}

/** Parse SSE text into frames, skipping blank lines and comment lines. */
export function parseSseFrames(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  const chunks = text.split(/\n\s*\n/);
  for (const chunk of chunks) {
    const lines = chunk.split('\n');
    let event: string | null = null;
    const dataLines: string[] = [];
    let sawAnything = false;
    for (const line of lines) {
      if (line.startsWith('event:')) {
        event = line.slice(6).trim();
        sawAnything = true;
      } else if (line.startsWith('data:')) {
        dataLines.push(line.slice(5).trimStart());
        sawAnything = true;
      } else if (line.startsWith(':')) {
        sawAnything = true;
      }
    }
    if (sawAnything && dataLines.length > 0) frames.push({ event, data: dataLines.join('\n') });
    else if (dataLines.length > 0) frames.push({ event, data: dataLines.join('\n') });
  }
  return frames;
}

/** Read a fetch Response body as SSE frames, yielding each frame as it arrives. */
export async function* readSseFrames(stream: ReadableStream<Uint8Array>): AsyncGenerator<SseFrame> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  while (true) {
    const next = await reader.read();
    if (next.done) {
      buffer += decoder.decode();
      for (const frame of parseSseFrames(buffer)) yield frame;
      break;
    }
    buffer += decoder.decode(next.value, { stream: true });
    const parts = buffer.split(/\n\s*\n/);
    buffer = parts.pop() ?? '';
    for (const part of parts) {
      for (const frame of parseSseFrames(part + '\n\n')) yield frame;
    }
  }
}

/** Parse one SSE frame's data as JSON; null when it is [DONE] or not JSON. */
function frameJson(frame: SseFrame): Json | null {
  const data = frame.data.trim();
  if (data === '' || data === '[DONE]') return null;
  try {
    const parsed: unknown = JSON.parse(data);
    return isRecord(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

/** One OpenAI chat-completion chunk carrying a piece of content. */
function openAIChunk(model: string, text: string): string {
  return sseEvent(null, {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion.chunk',
    created: 1767225600,
    model,
    choices: [{ index: 0, delta: { role: 'assistant', content: text }, finish_reason: null }],
  });
}

/** The OpenAI chunk that closes a stream: finish_reason "stop", no usage. */
function openAIStopChunk(model: string): string {
  return sseEvent(null, {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion.chunk',
    created: 1767225600,
    model,
    choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
  });
}

/** The OpenAI usage chunk the client's parser reads (emitted right before [DONE]). */
function openAIUsageChunk(model: string, input: number, output: number): string {
  return sseEvent(null, {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion.chunk',
    created: 1767225600,
    model,
    choices: [],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  });
}

/** An Anthropic stream's opening events for `model` (input tokens 0: the check ignores them). */
function anthropicStart(model: string): string[] {
  return [
    sseEvent('message_start', { type: 'message_start', message: { id: 'msg_aggregate', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: 0, output_tokens: 0 } } }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
}

/** An Anthropic stream's closing events for the output token count (stop_reason "end_turn"). */
function anthropicStop(model: string, output: number): string[] {
  void model;
  return [
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: 'end_turn', stop_sequence: null }, usage: { output_tokens: output } }),
    sseEvent('message_stop', { type: 'message_stop' }),
  ];
}

/**
 * An Anthropic provider stream as OpenAI SSE chunks. Input tokens come from message_start, the
 * output count from message_delta; the usage chunk closes the stream before [DONE].
 */
export async function* anthropicEventsToOpenAI(stream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let input = 0;
  let output = 0;
  yield sseEvent(null, {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion.chunk',
    created: 1767225600,
    model,
    choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }],
  });
  for await (const frame of readSseFrames(stream)) {
    const body = frameJson(frame);
    if (body === null) continue;
    if (frame.event === 'message_start' && isRecord(body.message) && isRecord(body.message.usage)) {
      input = numberOrZero(body.message.usage.input_tokens);
    } else if (frame.event === 'content_block_delta' && isRecord(body.delta) && typeof body.delta.text === 'string' && body.delta.text !== '') {
      yield openAIChunk(model, body.delta.text);
    } else if (frame.event === 'message_delta') {
      if (isRecord(body.usage)) output = numberOrZero(body.usage.output_tokens);
    }
  }
  yield openAIStopChunk(model);
  yield openAIUsageChunk(model, input, output);
  yield 'data: [DONE]\n\n';
}

/**
 * A Gemini provider stream as OpenAI SSE chunks. Text pieces come from each frame's candidate
 * parts; usage arrives with the final frame's usageMetadata.
 */
export async function* geminiEventsToOpenAI(stream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let input = 0;
  let output = 0;
  yield sseEvent(null, {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion.chunk',
    created: 1767225600,
    model,
    choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }],
  });
  for await (const frame of readSseFrames(stream)) {
    const body = frameJson(frame);
    if (body === null) continue;
    const text = geminiTextOf(body);
    if (text !== '') yield openAIChunk(model, text);
    const usage = geminiUsageOf(body);
    if (usage.input !== 0 || usage.output !== 0) {
      input = usage.input;
      output = usage.output;
    }
  }
  yield openAIStopChunk(model);
  yield openAIUsageChunk(model, input, output);
  yield 'data: [DONE]\n\n';
}

/**
 * A Gemini provider stream as Anthropic SSE events. The opening events carry input tokens 0
 * (the client's check ignores them on streams); the closing message_delta carries the real
 * output count from the final frame's usageMetadata.
 */
export async function* geminiEventsToAnthropic(stream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let output = 0;
  for (const event of anthropicStart(model)) yield event;
  for await (const frame of readSseFrames(stream)) {
    const body = frameJson(frame);
    if (body === null) continue;
    const text = geminiTextOf(body);
    if (text !== '') yield sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text } });
    const usage = geminiUsageOf(body);
    if (usage.output !== 0) output = usage.output;
  }
  for (const event of anthropicStop(model, output)) yield event;
}
