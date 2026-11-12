"""Root-cause the failures: replay every episode of a run and classify each action.

Per action (replayed in a fresh env, so the classification uses exactly the evidence the model had):
    contradicts   the action is impossible given earlier feedback (evidence_check, or for BugFix a patch
                  that fails the visible tests the model was shown; for Circuit an answer contradicted by probes)
    repeat        identical to an earlier action (RUN excluded)
    invalid       the env could not parse it
Per episode: did the model ever form a consistent answer; did its own Thought contain a correct value
it then did not use (BugFix/HiddenRule: arithmetic in the thought vs the action).

    python diagnose.py runs/q7_plain
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import agents
import envs


def fits_visible(env: envs.BugFix, n: int, expr: str) -> bool:
    try:
        return all(env.f(expr, x) == want for x, want in env.tests[n])
    except Exception:
        return False


def circuit_expected(env: envs.Circuit):
    v, out = env.x0, []
    for _, fn, c in env.stages:
        v = fn(v, c)
        out.append(v)
    return out


def classify(ep: dict) -> list[str]:
    env = envs.ALL[ep["env"]](ep["seed"])
    env.first_observation()
    labels, seen, probes = [], set(), {}
    for m in ep["messages"]:
        if m["role"] != "assistant" or env.finished:
            continue
        act = agents.action_of(m["content"])
        line = act.strip("`*> ").strip()
        env.line = line
        parts = line.split()
        verb = parts[0].upper().rstrip(":.") if parts else ""
        args = [a.strip("[](),") for a in parts[1:]]
        key = " ".join(line.upper().split())
        label = "ok"
        note, _ = env.evidence_check(verb, args)
        if note:
            label = "contradicts"
        if isinstance(env, envs.BugFix) and verb == "PATCH" and args:
            n = envs._int(args[0])
            rest = line.split(None, 2)
            expr = envs._expr(rest[2]) if len(rest) == 3 else ""
            if n in env.funcs and expr and not fits_visible(env, n, expr):
                label = "contradicts"
        if isinstance(env, envs.Circuit):
            exp = circuit_expected(env)
            k = envs._int(args)
            if verb == "ANSWER" and k:
                # stage k is faulty only if every probe before k matched and every probe at/after k differed
                bad = any((p < k and v != exp[p - 1]) or (p >= k and v == exp[p - 1]) for p, v in probes.items())
                if bad:
                    label = "contradicts"
            if verb == "PROBE" and k and 1 <= k <= env.n:
                probes[k] = env.value(k, True)
        if label == "ok" and key in seen and verb != "RUN":
            label = "repeat"
        seen.add(key)
        obs = env.step(act)
        if obs.startswith("Invalid action"):
            label = "invalid"
        labels.append(label)
    return labels


def main():
    run = Path(sys.argv[1])
    rows = [json.loads(x) for x in (run / "episodes.jsonl").open()]
    by_env = defaultdict(Counter)
    ep_any = defaultdict(Counter)
    for ep in rows:
        labs = classify(ep)
        by_env[ep["env"]].update(labs)
        ep_any[ep["env"]]["episodes"] += 1
        ep_any[ep["env"]]["success"] += ep["success"]
        for kind in ("contradicts", "repeat", "invalid"):
            ep_any[ep["env"]][f"ep_with_{kind}"] += kind in labs
    out = {}
    for e, c in sorted(by_env.items()):
        tot = sum(c.values())
        out[e] = {"actions": tot, **{k: round(v / tot, 2) for k, v in c.items()},
                  **{k: v for k, v in ep_any[e].items()}}
    print(json.dumps(out, indent=1))
    (run / "diagnosis.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
