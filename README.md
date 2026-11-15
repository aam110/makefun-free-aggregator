# makefun-free-aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/aam110/makefun-free-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/aam110/makefun-free-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server, two APIs (OpenAI Chat Completions + Anthropic Messages) over ten simulated free-tier model providers. It reads the trial brief (provider directory, quality table, workload manifest), builds one catalog, plans the whole workload up front — each request gets a rating-ordered candidate list and a primary under conservative per-provider planning caps — then routes each request to its planned primary and fails over across the candidate list on refusal (a spent budget or an outage kills a provider for the trial; a scheduled refusal skips it for that request only). All three provider wires are spoken natively (OpenAI, Anthropic, Gemini), JSON and streaming, with exact echo preservation of system text, user text, tool names and order, max_tokens, stop strings, image bytes and JSON mode.

## Run it

```sh
node main.ts
```

`node main.ts --list` prints the registered catalog JSON and exits.

---

<sub>Entered on [make.fun](https://make.fun/?ref=4kc4aac3&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
