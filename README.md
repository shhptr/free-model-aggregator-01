# free-model-aggregator-01

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/shhptr/free-model-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/shhptr/free-model-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

A single-file Node entry (`main.ts`, standard library only) that aggregates
ten free-model providers behind one OpenAI- and Anthropic-shaped API. The
providers speak three different wires (OpenAI, Anthropic, Gemini), use
three different auth styles, spend per-trial call budgets, refuse on
schedules, and one of them goes dark mid-trial — the entry routes around
all of it.

## How it works

- The trial brief is the map: a provider directory, a sealed ratings table
  with capabilities (tools, vision, JSON mode, context length), and the
  twenty-request manifest. At startup the entry probes every provider's
  own catalog route and registers the catalog it measured — the same
  document `--list` prints.
- Requests are parsed once into a canonical form (system text, turns,
  images, tool definitions, token cap, stop sequences, JSON mode, stream
  flag) and rendered per provider wire, so a request keeps its meaning on
  any wire: an image stays the same image, a tool keeps its name, a JSON
  request stays a JSON request. A JSON-mode OpenAI request is never sent
  to an Anthropic-wire provider, which cannot echo JSON mode back.
- Routing is capability-first, rating-second: only free models that meet
  the manifest's requirements are candidates, best rating first.
- Upstream calls are always single, buffered and non-streaming. An answer
  is judged before a byte reaches the client — it must be the provider's
  own answer text with its usage block — and only then is it re-expressed
  in the client's wire, JSON or SSE, with the provider's usage numbers and
  stop reason translated, never recomputed.
- Failover is budget-aware. Every provider's refusals share one shape, so
  the entry tracks what it observed: a refusal at or before a provider's
  known budget floor can only be a scheduled one, and earns that candidate
  one immediate retry; two consecutive refusals park the provider for the
  rest of the pass; later passes re-admit it before giving up. Exactly one
  successful provider call ever serves a request.
- No output reads the wall clock or a random source, so a replay of a
  trial reproduces the live pass byte for byte.

## Run it

```sh
# Run the local test suite (mock environment + mock providers, no keys,
# no network):
node test/run_tests.mjs

# Grade against the kit's own simulated environment and scorer:
sh test/e2e_kit.sh 0

# Print the served catalog (the same document the server registers):
MAKEFUN_ENV_URL=<trial env> MAKEFUN_PROMPT_FILE=<trial brief> node main.ts --list
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=w4kvb54w&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
