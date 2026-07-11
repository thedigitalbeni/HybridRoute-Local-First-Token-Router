# Track 1 — Hybrid Token-Efficient Routing Agent

A containerized agent that routes each task to either a **local** quantized
model (zero Fireworks tokens) or a **remote** Fireworks model, depending on
which of the 8 published capability categories the task falls into.

## How it works

1. `main.py` reads `/input/tasks.json`, classifies each prompt with
   `router.py`, and routes it local or remote per `ROUTE_MAP`.
2. **Local** (`local_infer.py`): factual knowledge, sentiment classification,
   summarization, named entity recognition. Handled by a small quantized
   model bundled directly in the image via `llama-cpp-python`. Costs zero
   Fireworks tokens.
3. **Remote** (`remote_infer.py`): mathematical reasoning, logical/deductive
   reasoning, code debugging, code generation. Sent through
   `FIREWORKS_BASE_URL` using a model from `ALLOWED_MODELS`. These categories
   need precise multi-step reasoning or syntactic correctness that a 1.5-3B
   quantized model is too risky to trust against the 80% accuracy gate.
4. If a local answer comes back empty (model failure), the agent
   automatically escalates that task to remote as a safety net — costs a
   few tokens but protects you from a hard zero on that task.
5. Results are written to `/output/results.json` as
   `[{"task_id": ..., "answer": ...}, ...]`.

## Setup (run these in order)

```bash
# 1. Download the local model (needs real internet access -- huggingface.co)
./download_model.sh

# 2. Sanity check the router against the 8 practice tasks (no model needed)
python3 -c "
import json
from router import classify_task, ROUTE_MAP
for t in json.load(open('practice_tasks.json')):
    cat = classify_task(t['prompt'])
    print(t['task_id'], '->', cat, '(' + ROUTE_MAP[cat] + ')')
"

# 3. Install deps locally and run the real practice loop
pip install -r requirements.txt
export FIREWORKS_API_KEY="<your team's key from https://app.fireworks.ai/fire-pass>"
export ALLOWED_MODELS="<comma-separated list published on launch day>"
./run_practice.sh
```

Read `practice_output.json` and actually judge the answers yourself —
especially the 4 "local" categories. If the small model is shaky on any of
them, flip that category to `"remote"` in `router.py`'s `ROUTE_MAP`. Losing
some token efficiency beats failing the 80% accuracy gate.

## Build and submit

```bash
# Edit REGISTRY_USER / IMAGE_NAME at the top of build_and_push.sh first
./build_and_push.sh
```

Then on the lablab.ai submission page, paste in the public image reference
(e.g. `ghcr.io/you/track1-agent:latest`). Double check the GHCR package
visibility is **public** — a private package will PULL_ERROR and score zero.

## The actual competitive math (read this before tuning anything)

19 fixed tasks, 80% accuracy gate. **16/19 = 84.2% is the exact minimum
passing score** — 15/19 = 78.9% fails. On the live dashboard, every team in
the top 6 is tied at exactly 84.2%, which means the entire competitive field
has converged on the same strategy: hit the bare minimum to clear the gate,
then compete purely on tokens. Chasing extra accuracy above the gate (two
teams hit 89.5%) cost 6,192 and 10,522 tokens — far more than the 4,268-token
leader, for a score the ranking doesn't reward once you've already passed.

Two implications:
- **The real optimization target is tokens, not accuracy, once you're
  reliably above the gate.** Don't over-engineer accuracy past that point.
- **Being exactly at 16/19 is risky, not efficient.** The judge isn't
  perfectly deterministic run-to-run — sitting exactly on the gate line
  means a re-score could drop you below it. Aim for a small buffer (17/19)
  rather than the bare minimum; a few hundred extra tokens is cheap
  insurance against getting zeroed out entirely.

**Token count does not care which model you use, only how many tokens you
send/receive.** `nvfp4` vs full-precision Gemma is a dollar-cost lever for
your $50 Fireworks budget, not a leaderboard-token lever — don't pick a
quantized model thinking it saves you leaderboard points. Pick whichever
allowed model is most *accurate per token* for a category; use
`compare_models.py` below to actually check this instead of guessing.

## Deciding which model wins each category — use compare_models.py

`CATEGORY_MODEL_PREFERENCE` in `remote_infer.py` is currently a starting
hypothesis (kimi for code is a safe bet given its name; gemma vs minimax for
math/logic is genuinely unverified). Before you lock it in:

```bash
export FIREWORKS_API_KEY="<real key>"
export FIREWORKS_BASE_URL="https://api.fireworks.ai/inference/v1"
export ALLOWED_MODELS="minimax-m3,kimi-k2p7-code,gemma-4-31b-it,gemma-4-26b-a4b-it,gemma-4-31b-it-nvfp4"

python3 compare_models.py math
python3 compare_models.py logic
python3 compare_models.py code_debug
python3 compare_models.py code_gen
```

This calls every allowed model on the same prompt and prints answer + token
cost side by side. Judge correctness yourself first, then prefer the
cheapest model among the ones that actually got it right. Update
`CATEGORY_MODEL_PREFERENCE` based on what you actually see, not on model
names sounding impressive. This costs 5 calls per category (20 total) —
deliberate, not repeated.



```
minimax-m3
kimi-k2p7-code
gemma-4-31b-it
gemma-4-26b-a4b-it
gemma-4-31b-it-nvfp4
```

`remote_infer.py` routes by category preference, matched by substring so it
doesn't matter whether the harness publishes short names or full
`accounts/fireworks/models/...` paths:

| Category | 1st choice | Fallback |
|---|---|---|
| `code_debug` / `code_gen` | `kimi-k2p7-code` | `minimax-m3` |
| `math` / `logic` | `gemma-4-31b-it` | `gemma-4-31b-it-nvfp4` → `minimax-m3` |

**Before your first remote test run**, go deploy at least
`gemma-4-31b-it` at https://app.fireworks.ai/models — Gemma models are
on-demand and return a 404 until deployed (see the credits section below).
If you skip this, `RemoteModel` will silently fall through to
`gemma-4-31b-it-nvfp4` and eventually `minimax-m3` instead of failing the
task outright — which is the safety net working as intended, but you'll get
better accuracy from the deployed dense model, so don't rely on the fallback
by default.

`gemma-4-26b-a4b-it` is almost certainly the "Gemma 4 E4B" the organizers
called out as the cheapest deploy option (~$7/hr idle billing) — if you want
to chase the $1,000 Gemma bonus for Track 1 cheaply, that's the one to
deploy and route through deliberately, then undeploy right after testing.

## Credits & access (per the team's Discord clarifications)

- **Fireworks credits are per-team**, not per-person: one coupon code sent to
  your team's registered email within 2-3 business days of team
  registration. Redeem at https://app.fireworks.ai/fire-pass. Check spam.
- **There is no separate "AMD cloud credit"** — your compute is your team's
  Jupyter instance at `notebooks.amd.com/hackathon` (8h/day allowance). Use
  that for iterating and testing, not for the final scored run (the judging
  environment is separate — 4GB RAM / 2 vCPU).
- **Gemma is on-demand, not free-standing**: deploy it first at
  https://app.fireworks.ai/models (a 404 there means "not deployed yet," not
  "banned"). The cheapest option, Gemma 4 E4B, bills roughly **$7/hour even
  while idle**. If you want the Gemma bonus prize, deploy it, test quickly,
  and **undeploy immediately after** — otherwise it burns your $50 fast for
  no reason. You do not need Gemma to pass the accuracy gate.

## Scoring, per the official clarification

- **Accuracy gate: 80%.** Below that you're excluded from the leaderboard
  regardless of token count — there are exactly 19 fixed tasks, so every
  score is `n/19`.
- Once you clear 80%, ranking is by **total tokens** — fewer is better.
- Scores can vary slightly run-to-run on identical code: the LLM judge isn't
  perfectly deterministic. Not a bug on your end.
- `ZERO_API_CALLS` shown alongside a result is **not a failure** — it just
  means that submission made zero remote calls (fully local), which is a
  valid, in fact ideal, strategy per the token-scoring rules.

## Troubleshooting

| Status | Fix |
|---|---|
| `PULL_ERROR` | Confirm the image is public and has a `linux/amd64` manifest. Rebuild with `docker buildx build --platform linux/amd64 ...` if you're on Apple Silicon. |
| `RUNTIME_ERROR` | Check container logs locally — something in agent code crashed. |
| `TIMEOUT` | Check for hangs / retries; hard limit is 10 minutes. |
| `OUTPUT_MISSING` | Confirm `/output/results.json` is written before exit — `main.py` always writes, even on partial failure. |
| `INVALID_RESULTS_SCHEMA` | Every entry needs both `task_id` and `answer` — already enforced by `main.py`. |
| `MODEL_VIOLATION` | You called a model not in `ALLOWED_MODELS`. `remote_infer.py` only ever reads models from that env var — don't hardcode a model string elsewhere. |
| `IMAGE_TOO_LARGE` | Over 10GB compressed. The local model weights are almost certainly the culprit — use a smaller quant. |
| `ACCURACY_GATE_FAILED` | Answer quality issue, not infra. Revisit which categories are routed local vs remote in `router.py`. |

## Tuning ideas if you have spare time

- **Local model size is a free lever.** Since local answers cost zero
  leaderboard tokens regardless of size, `download_model.sh` now defaults to
  Qwen2.5-3B (up from 1.5B) — more reliable on factual/sentiment/summary/ner
  for the same token cost: zero. If you have RAM headroom left after testing
  (check actual usage under `docker run --memory=4g`), there's no reason not
  to run the biggest model that reliably fits.
- The router is pure regex/keyword matching — fast and free, but not
  perfect. If you have time, log the classified category next to each
  answer during practice runs and manually verify a few misclassifications
  don't slip through.
- `remote_infer.py`'s `MAX_TOKENS_BY_CATEGORY` values are estimates sized to
  what each answer type actually needs. If `compare_models.py` or
  `run_practice.sh` shows a truncated code answer (missing closing bracket,
  cut off mid-line), bump that category's cap — a truncated answer failing
  accuracy costs the whole task, far worse than a slightly larger cap.
- **Advanced/optional, only if the core submission is solid and time
  remains:** batching same-category tasks into a single API call (with clear
  per-task delimiters, asking for one JSON array of answers back) would
  amortize the system-prompt token cost across multiple tasks instead of
  paying it once per task. This meaningfully cuts tokens on paper, but adds
  real accuracy risk (cross-task confusion, harder parsing) for a field where
  a single failed task can drop you below the gate. Don't attempt this until
  the simple one-task-per-call version is tested and safely above 16/19.
