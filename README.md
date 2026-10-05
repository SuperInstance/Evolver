# evolver

Overnight prompt evolution. Give it a task, test scenarios, and a judge —
come back to a better prompt.

## What it is

A small evolutionary loop: a population of candidate prompts is tested
against fixed scenarios on a local model, scored by a cheap typed judge
(JEV), and the winners are mutated into the next generation. Selection
pressure does the work. No gradients, no training runs, no GPU.

## Why it works

- **The judge is cheap.** JEV answers typed yes/no questions in ~1 second
  for fractions of a cent. You can afford thousands of judgments —
  enough for real selection pressure.
- **The model is local.** Generation runs on ollama (your machine or a
  cheap cloud box). No API bills, no rate limits, runs all night.
- **Smallness is the instrument.** Narrow task, fixed scenarios, one
  number per prompt. Evolution doesn't need scale; it needs iterations.

## Overnight proof

2026-10-05: 13 generations evolving GO/NO-GO verdict prompts on a 1.2B
local model, judged by JEV.

| gen | best | mean |
|-----|------|------|
| 0   | 0.740 | 0.596 |
| 4   | 0.923 | 0.773 |
| 11  | 0.890 | 0.804 |

Winner: *"Give a GO or NO-GO verdict. One line of reasoning. Be specific,
no hedging. Consider the downside first."* — 0.923.

(One incident: a VM replacement wiped SSH host keys mid-run and six
generations scored 0.000. The fix — workspace-pinned host keys — is
baked into the ssh-ollama backend. Rock charted.)

## Quick start

```bash
pip install pyyaml
cp config.example.yaml config.yaml
# edit config.yaml: your task, scenarios, seeds, judge question
python evolve.py --config config.yaml --generations 20
```

Or tick it from cron, one generation per run (how the overnight proof ran):

```bash
*/30 * * * * cd ~/evolver && python evolve.py --config config.yaml --once
```

## Config

Everything lives in the YAML — no hardcoded tasks:

- `task.prompt_template` — `{prompt}` and `{scenario}` slots.
- `scenarios` — fixed test cases every prompt is judged on.
- `seeds` — generation-0 population.
- `population`, `keep` — population size, survivors per generation.
- `judge` — JEV question plus true/false labels. This is the fitness
  function; write it carefully.
- `mutations` — `append`, `prepend`, `replace`, `crossover`.
- `model.backend` — `local-ollama` (zero-config) or `ssh-ollama`
  (remote box; pin `known_hosts` in the workspace).

## Ledger

Every scored prompt appends one JSONL row. Fields follow the stacking
convention so loops can be compared:

`loop, gen, dimension, score, movement (new/up/down/flat), ts`,
plus `prompt` and `details` (mutation used, parent score, per-scenario
scores).

```bash
python evolve.py --config config.yaml --report   # best per gen + winner
```

## The shape

This is one instance of a general pattern: **generate broadly, select
narrowly, distill.** The evolver is the selection half. Pair it with a
researcher (topics in, briefs out) and a curator (scores in, taste out)
and the loop compounds.
