# Free Model Aggregator — make.fun bounty entry

One Node 22 server (stdlib only) that routes each request to the best **free** model that can
serve it, across ten simulated free-tier providers on three wires (OpenAI, Anthropic, Gemini).

## How it works

- **Brief + probe.** Reads the trial brief from `MAKEFUN_PROMPT_FILE` (provider directory,
  sealed quality table, workload manifest), probes every provider's own catalog route in its
  own wire, and registers the one catalog it built (`node main.ts --list` prints that same
  document byte-for-byte).
- **Routing (quality, 60%).** For each request, every free model meeting its requirements
  (tools, vision, JSON mode, min context) is ranked by the sealed rating table's
  `rating_milli`, best first; ties break by the provider's sealed budget, then provider id,
  then model id — a fixed order, so a replay matches byte-for-byte. A paid model is never
  picked. JSON-mode requests from an OpenAI client never ride the Anthropic wire (it cannot
  carry `response_format`, so the provider would echo JSON mode absent).
- **Failover (10%).** A 402/429/5xx never ends the request: the next-best candidate serves it.
  A refusal that says the budget is spent or the provider is out marks that provider dead
  for the trial; a scheduled refusal does not — the provider is tried again on later requests.
- **Wires (10%).** Requests and answers are translated between all three wires without
  dropping a field the provider echoes: system text, user text, tool names, `max_tokens`,
  stop sequences, image bytes, JSON mode. Streaming is translated event-by-event (OpenAI
  chunks, Anthropic message/content-block events, Gemini frames), with usage carried exactly
  and the stop family always `stop` / `end_turn`.
- **Determinism.** No wall clock in any decision (fixed `created` timestamps), no unseeded
  randomness, requests served one at a time in arrival order.

## Practice-trial scores (manual harness: `program/env/server.mjs` + `program/scorer/score.mjs`)

| trial | score | quality | served | wire | failover | catalog | upstream calls |
|------:|------:|--------:|-------:|-----:|---------:|--------:|---------------:|
| 0 | 99 | 99 | 100 | 100 | 100 | 100 | 26 |
| 1 | 99 | 99 | 100 | 100 | 100 | 100 | 29 |
| 2 | 99 | 99 | 100 | 100 | 100 | 100 | 27 |

20 of 20 requests valid in every trial. The one quality point short of 100 is the hidden
failure schedule: a scheduled refusal or a spent budget forces a fallback one rating step
below the best (1470 instead of 1475) on a handful of requests — no online router can see
those failures coming.

## Files

- `main.ts` — the server: brief, probe, catalog, routes, register, poll.
- `lib/providers.ts` — the brief's types and parser, auth headers, the catalog probe.
- `lib/wire.ts` — the client wires and the Anthropic ↔ Chat Completions translation.
- `lib/translate.ts` — the remaining cross-wire translation (OpenAI ↔ Anthropic/Gemini,
  Anthropic ↔ Gemini, and every provider answer back to either client wire, JSON and SSE).
- `LICENSE` — MIT.
