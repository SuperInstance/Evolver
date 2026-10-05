#!/usr/bin/env python3
"""evolver — overnight prompt evolution, as a tool.

Given a task description, test scenarios, and a JEV judge, evolve better
prompts through selection and mutation. Designed to run unattended:
one invocation = N generations, or a single generation per cron tick.

Proven overnight 2026-10-05: 13 generations on a local 1.2B model,
mean JEV score 0.60 -> 0.80, winner 0.923.

Usage:
    python evolve.py --config config.yaml
    python evolve.py --config config.yaml --generations 10
    python evolve.py --config config.yaml --once          # single generation (cron mode)
    python evolve.py --config config.yaml --report       # ledger summary, no run
    python evolve.py --config config.yaml --dry-run      # show the plan, touch nothing

Ledger format follows the stacking convention (see STACKING.md):
each row carries loop, gen, dimension, score, movement, ts.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time

try:
    import yaml
except ImportError:
    sys.exit("evolver needs PyYAML: pip install pyyaml")


# ---------------------------------------------------------------- backends

def _run(cmd, input_bytes=None, timeout=120):
    r = subprocess.run(cmd, input=input_bytes, capture_output=True,
                       timeout=timeout)
    return r.stdout.decode(errors="replace")


class SshOllama:
    """Generate via ollama on a remote host over SSH.

    The known_hosts file should live next to the config (workspace-pinned),
    not in ~/.ssh — worker-VM replacement wipes ephemeral home directories.
    """
    def __init__(self, cfg, workdir):
        s = cfg["ssh"]
        kh = s.get("known_hosts", "")
        if kh and not os.path.isabs(kh):
            kh = os.path.join(workdir, kh)
        self.ssh = ["ssh"]
        if s.get("proxy"):
            self.ssh += ["-o", f"ProxyCommand={s['proxy']}"]
        self.ssh += [
            "-i", os.path.expanduser(s["key"]),
            "-o", "ConnectTimeout=15",
            "-o", "StrictHostKeyChecking=yes",
        ]
        if kh:
            self.ssh += ["-o", f"UserKnownHostsFile={kh}"]
        self.ssh.append(s["host"])
        self.model = cfg["model"]
        self.num_predict = cfg.get("num_predict", 120)

    def generate(self, full_prompt):
        payload = json.dumps({
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {"num_predict": self.num_predict},
        })
        try:
            out = _run(self.ssh + ["curl -s localhost:11434/api/generate -d @-"],
                       input_bytes=payload.encode(), timeout=150)
            return json.loads(out).get("response", "").strip()
        except Exception:
            return ""


class LocalOllama:
    """Generate via ollama on localhost:11434. No SSH needed."""
    def __init__(self, cfg, workdir):
        self.model = cfg["model"]
        self.num_predict = cfg.get("num_predict", 120)
        self.url = cfg.get("url", "http://localhost:11434/api/generate")

    def generate(self, full_prompt):
        import urllib.request
        payload = json.dumps({
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {"num_predict": self.num_predict},
        }).encode()
        try:
            req = urllib.request.Request(self.url, data=payload,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=150) as resp:
                return json.loads(resp.read().decode()).get("response", "").strip()
        except Exception:
            return ""


BACKENDS = {"ssh-ollama": SshOllama, "local-ollama": LocalOllama}


# ---------------------------------------------------------------- judge

class JevJudge:
    """Score model output with a JEV noul() question. Cheap, typed, fast."""
    def __init__(self, cfg, workdir):
        cmd = cfg.get("command", "")
        if cmd and not os.path.isabs(cmd):
            # allow ~ and relative-to-workdir paths
            cmd = os.path.expanduser(cmd)
            if not os.path.isabs(cmd):
                cmd = os.path.join(workdir, cmd)
        self.cmd = cmd
        self.question = cfg["question"]
        self.true_label = cfg["true_label"]
        self.false_label = cfg["false_label"]

    def score(self, output):
        if not output:
            return 0.0
        try:
            out = _run([self.cmd, "noul", output, self.question,
                        "--true", self.true_label,
                        "--false", self.false_label], timeout=90)
            return float(json.loads(out).get("noul", 0.0))
        except Exception:
            return 0.0


# ---------------------------------------------------------------- mutations

def apply_mutation(op, prompt, rng, other=None):
    kind = op["op"]
    if kind == "append":
        return prompt + op["text"]
    if kind == "prepend":
        return op["text"] + prompt
    if kind == "replace":
        return prompt.replace(op["old"], op["new"])
    if kind == "crossover" and other:
        # first half of one winner, second half of another
        a, b = prompt.split(), other.split()
        cut = max(1, len(a) // 2)
        return " ".join(a[:cut] + b[cut:])
    return prompt


# ---------------------------------------------------------------- ledger

def load_ledger(path):
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return rows


def append_ledger(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def movement_of(score, prev_mean, gen):
    if gen == 0:
        return "new"
    d = score - prev_mean
    if abs(d) < 0.02:
        return "flat"
    return "up" if d > 0 else "down"


# ---------------------------------------------------------------- core

def run_generation(cfg, workdir, ledger_path, backend, judge, rng, gen):
    loop = cfg["loop"]
    pop_size = cfg.get("population", 6)
    keep = cfg.get("keep", 3)
    template = cfg["task"]["prompt_template"]
    scenarios = cfg["scenarios"]
    mutations = cfg.get("mutations", [])

    rows = load_ledger(ledger_path)

    if gen == 0:
        pop = [(s, None, None) for s in cfg["seeds"][:pop_size]]
    else:
        last = [r for r in rows if r["gen"] == gen - 1]
        last.sort(key=lambda r: r["score"], reverse=True)
        winners = [(r["prompt"], r["score"]) for r in last[:keep]]
        pop = [(p, None, s) for p, s in winners]
        while len(pop) < pop_size:
            parent, pscore = winners[rng.randrange(len(winners))]
            op = rng.choice(mutations) if mutations else {"op": "identity"}
            if op.get("op") == "identity":
                child = parent
            else:
                mate, _ = winners[rng.randrange(len(winners))]
                child = apply_mutation(op, parent, rng, other=mate)
            pop.append((child, op.get("op", "identity"), pscore))

    prev_mean = None
    if gen > 0:
        prev = [r["score"] for r in rows if r["gen"] == gen - 1]
        prev_mean = sum(prev) / len(prev) if prev else 0.0

    for i, (prompt, mut_op, parent_score) in enumerate(pop):
        per_scenario = []
        for sc in scenarios:
            full = template.format(prompt=prompt, scenario=sc)
            out = backend.generate(full)
            time.sleep(cfg.get("pause_secs", 1))
            per_scenario.append(judge.score(out))
        score = sum(per_scenario) / len(per_scenario) if per_scenario else 0.0
        row = {
            "loop": loop,
            "gen": gen,
            "dimension": f"prompt:{i:02d}",
            "score": round(score, 3),
            "movement": movement_of(score, prev_mean or 0.0, gen),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "prompt": prompt,
            "details": {
                "mutation": mut_op,
                "parent_score": parent_score,
                "per_scenario": [round(s, 3) for s in per_scenario],
            },
        }
        append_ledger(ledger_path, row)
        print(f"[{loop}] gen {gen} {score:.3f} ({row['movement']}): "
              f"{prompt[:64]}...", flush=True)
    return gen


def report(ledger_path):
    rows = load_ledger(ledger_path)
    if not rows:
        print("ledger empty — nothing evolved yet.")
        return
    loop = rows[0].get("loop", "?")
    print(f"loop: {loop}  ({len(rows)} rows)")
    gens = sorted(set(r["gen"] for r in rows))
    for g in gens:
        grp = [r for r in rows if r["gen"] == g]
        best = max(grp, key=lambda r: r["score"])
        mean = sum(r["score"] for r in grp) / len(grp)
        print(f"  gen {g}: n={len(grp)} best={best['score']:.3f} "
              f"mean={mean:.3f}")
    w = max(rows, key=lambda r: r["score"])
    print(f"\nwinner (gen {w['gen']}, {w['score']:.3f}):\n  {w['prompt']}")


def main():
    ap = argparse.ArgumentParser(description="evolver — evolve better prompts overnight")
    ap.add_argument("--config", required=True, help="YAML config file")
    ap.add_argument("--generations", type=int, default=None,
                    help="run this many generations, then exit (default: from config)")
    ap.add_argument("--once", action="store_true",
                    help="run a single generation (cron mode)")
    ap.add_argument("--ledger", default=None, help="override ledger path")
    ap.add_argument("--dry-run", action="store_true",
                    help="show the plan without calling model or judge")
    ap.add_argument("--report", action="store_true",
                    help="print ledger summary and exit")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed (default: random)")
    args = ap.parse_args()

    cfg_path = os.path.abspath(args.config)
    workdir = os.path.dirname(cfg_path)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    ledger_path = args.ledger or cfg.get("ledger", "./ledger.jsonl")
    if not os.path.isabs(ledger_path):
        ledger_path = os.path.join(workdir, ledger_path)

    if args.report:
        report(ledger_path)
        return

    rng = random.Random(args.seed)

    backend_name = cfg["model"]["backend"]
    if backend_name not in BACKENDS:
        sys.exit(f"unknown model backend: {backend_name} "
                 f"(choose from {list(BACKENDS)})")
    backend = BACKENDS[backend_name](cfg["model"], workdir)
    judge = JevJudge(cfg["judge"], workdir)

    rows = load_ledger(ledger_path)
    start_gen = max((r["gen"] for r in rows), default=-1) + 1

    if args.dry_run:
        n = 1 if args.once else (args.generations or cfg.get("generations", 1))
        print(f"would run {n} generation(s) from gen {start_gen}:")
        print(f"  loop={cfg['loop']} backend={backend_name} "
              f"model={cfg['model']['model']}")
        print(f"  population={cfg.get('population', 6)} "
              f"scenarios={len(cfg['scenarios'])} "
              f"mutations={len(cfg.get('mutations', []))}")
        print(f"  ledger={ledger_path}")
        return

    n = 1 if args.once else (args.generations
                             if args.generations is not None
                             else cfg.get("generations", 1))
    # generations: 0 in config (or --once) means "exactly one, then exit"
    # so cron can tick it; a positive N runs N and exits.
    n = max(1, n)
    for g in range(start_gen, start_gen + n):
        run_generation(cfg, workdir, ledger_path, backend, judge, rng, g)


if __name__ == "__main__":
    main()
