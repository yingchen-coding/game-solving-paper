"""Compact process-verified worked teacher for TRAIN_GAMES ONLY (GuessNumber, CodeBreak, HiddenRule, OddCoin).

Strict rules (PLAN.md):
1. Reads ONLY the visible transcript `messages` and public task parameters (never hidden state).
2. Implements ONLY `envs.TRAIN_GAMES` (raises ValueError on any held-out or near-transfer env).
3. Every Thought executes compact, universal domain-general steps (~180-260 chars/turn):
   FACTS: concrete observations & step-by-step expected arithmetic; remaining unknowns/candidates.
   CHECK: plug EVERY observation into explicit candidate(s) with exact arithmetic; write REJECT on mismatch; never repeat.
   PLAN:  pick one valid test (equal-size split / single-variable probe) or commit when verified.
"""
from __future__ import annotations

import re

import agents
import envs


def _obs_and_acts(messages: list[dict]) -> tuple[list[str], list[str]]:
    obs = [m["content"] for m in messages if m["role"] == "user"]
    acts = [agents.action_of(m["content"]) for m in messages if m["role"] == "assistant"]
    return obs, acts


def _guessnumber_step(messages: list[dict], env: envs.GuessNumber) -> str:
    obs, acts = _obs_and_acts(messages)
    lo, hi = 1, env.hi
    notes = []
    for a, o in zip(acts, obs[1:]):
        g = int(a.split()[1])
        if o.startswith("Higher"):
            lo = max(lo, g + 1)
            notes.append(f"{g}->H")
        elif o.startswith("Lower"):
            hi = min(hi, g - 1)
            notes.append(f"{g}->L")
    mid = (lo + hi) // 2
    span = hi - lo + 1
    hist = ", ".join(notes) if notes else "none"
    facts = f"FACTS: {hist} => active range [{lo}, {hi}] ({span} left)."
    if lo == hi:
        last_g = int(acts[-1].split()[1])
        rel = ">" if mid > last_g else "<"
        check = f"CHECK: Candidate {mid}: {mid} {rel} {last_g} matches latest bound, all prior bounds satisfied."
        plan = f"PLAN: Unique verified value {mid}; commit."
    else:
        check = f"CHECK: {span} values left, do not guess early. Mid = ({lo}+{hi})//2 = {lo+hi}//2 = {mid} (untried)."
        plan = f"PLAN: Probe {mid} to split [{lo}, {hi}] into [{lo}, {mid-1}] vs [{mid+1}, {hi}]."
    return f"Thought:\n{facts}\n{check}\n{plan}\nAction: GUESS {mid}"


def _explain_score(cand: list[int], guess: list[int]) -> tuple[int, int, str]:
    e = sum(a == b for a, b in zip(cand, guess))
    shared = sum(min(cand.count(d), guess.count(d)) for d in set(guess))
    m = shared - e
    return e, m, f"E={e},M={m}"


def _codebreak_step(messages: list[dict], env: envs.CodeBreak) -> str:
    obs, _ = _obs_and_acts(messages)
    past: list[tuple[list[int], int, int]] = []
    for o in obs[1:]:
        m = re.match(r"([\d ]+) -> EXACT=(\d) MISPLACED=(\d)", o)
        if m:
            g = [int(d) for d in m.group(1).split()]
            past.append((g, int(m.group(2)), int(m.group(3))))

    if not past:
        facts = f"FACTS: 0 observations; {env.k} slots in 1..{env.digits}."
        check = "CHECK: All combinations open; distinct values test 3 symbols at once."
        plan = "PLAN: Start with 1 2 3."
        return f"Thought:\n{facts}\n{check}\n{plan}\nAction: GUESS 1 2 3"

    all_cands = [c for c in envs.all_codes(env.k, env.digits) if all(env.score(c, g) == (e, mis) for g, e, mis in past)]
    best_cand = all_cands[0]
    if len(all_cands) > 1:
        best_key = (9999, 9999)
        for c in all_cands:
            buckets: dict[tuple[int, int], int] = {}
            for secret in all_cands:
                fb = env.score(secret, c)
                buckets[fb] = buckets.get(fb, 0) + 1
            key = (max(buckets.values()), -len(buckets))
            if key < best_key:
                best_key, best_cand = key, c

    hist = "; ".join(f"{' '.join(map(str, g))}->{e},{mis}" for g, e, mis in past)
    facts = f"FACTS: {hist}."

    parts = []
    for d in range(1, env.digits + 1):
        bad = [best_cand[0], best_cand[1], d]
        if bad != best_cand and bad not in all_cands and bad != past[-1][0]:
            for g, e, mis in past:
                ge, gm, tag = _explain_score(bad, g)
                if (ge, gm) != (e, mis):
                    parts.append(f"Try {' '.join(map(str, bad))} vs {' '.join(map(str, g))}: {tag}!={e},{mis} (REJECT).")
                    break
            break

    cand_s = " ".join(map(str, best_cand))
    vers = [f"vs {' '.join(map(str, g))}->{e},{mis} OK" for g, e, mis in past]
    parts.append(f"Candidate {cand_s}: " + ", ".join(vers) + " (untried).")
    check = "CHECK: " + " ".join(parts)
    plan = f"PLAN: {'Commit unique' if len(all_cands) == 1 else 'Test consistent'} candidate {cand_s}."
    return f"Thought:\n{facts}\n{check}\n{plan}\nAction: GUESS {cand_s}"


def _fmt_poly(a: int, b: int, c: int) -> str:
    terms = []
    if a != 0:
        terms.append("x * x" if a == 1 else ("-1 * x * x" if a == -1 else f"{a} * x * x"))
    if b != 0:
        terms.append("x" if b == 1 else ("-1 * x" if b == -1 else f"{b} * x"))
    if c != 0 or not terms:
        terms.append(str(c))
    expr = terms[0]
    for t in terms[1:]:
        expr += (" - " + t[1:].lstrip()) if t.startswith("-") else (" + " + t)
    return expr


def _hiddenrule_step(messages: list[dict], env: envs.HiddenRule) -> str:
    obs, acts = _obs_and_acts(messages)
    pts: list[tuple[int, int]] = []
    for a, o in zip(acts, obs[1:]):
        m_x = re.search(r"-?\d+", a)
        m_v = re.search(r"= (-?\d+)", o)
        if m_x and m_v:
            pts.append((int(m_x.group(0)), int(m_v.group(1))))

    if len(pts) == 0:
        return "Thought:\nFACTS: 0 pts known; a*x^2+b*x+c has 3 unknowns.\nCHECK: Need 3 pts before answering.\nPLAN: Query x=0 to get c.\nAction: QUERY 0"
    if len(pts) == 1:
        _, y0 = pts[0]
        return f"Thought:\nFACTS: f(0)={y0} => c={y0}; 2 unknowns (a,b) left.\nCHECK: 1 pt cannot pin down 3 unknowns.\nPLAN: Query x=1.\nAction: QUERY 1"
    if len(pts) == 2:
        (_, y0), (_, y1) = pts
        ab = y1 - y0
        nx = 2 if (abs(y0 + y1) % 2 == 0) else -1
        lin = _fmt_poly(0, ab, y0)
        return (
            f"Thought:\nFACTS: f(0)={y0}, f(1)={y1} => c={y0}, a+b={ab}; 1 unknown left.\n"
            f"CHECK: ({lin}) fits x=0,1, but quadratic may differ; need 3rd point.\n"
            f"PLAN: Query x={nx}.\nAction: QUERY {nx}"
        )

    pm = dict(pts)
    c, y1 = pm[0], pm[1]
    ab = y1 - c
    if 2 in pm:
        tx, ty = 2, pm[2]
        tab = (ty - c) // 2
        a = tab - ab
        b = ab - a
        sol = f"c={c}, a+b={ab}, 2a+b=({ty}-({c}))//2={tab} => a={a}, b={b}"
    else:
        tx, ty = -1, pm[-1]
        amb = ty - c
        a = (ab + amb) // 2
        b = ab - a
        sol = f"c={c}, a+b={ab}, a-b=({ty}-({c}))={amb} => a={a}, b={b}"

    facts = "FACTS: " + ", ".join(f"f({x})={y}" for x, y in pts) + " (3 pts, 3 unknowns)."
    parts = []
    if a != 0:
        lv = ab * tx + c
        lin = _fmt_poly(0, ab, c)
        parts.append(f"Try ({lin}): x={tx} -> {lv}!={ty} (REJECT).")
    expr = _fmt_poly(a, b, c)
    vers = ", ".join(f"x={x}->{a*x*x + b*x + c}=={y} OK" for x, y in pts)
    parts.append(f"Solve: {sol} => ({expr}). Check all: {vers}.")
    return f"Thought:\n{facts}\nCHECK: {' '.join(parts)}\nPLAN: All 3 match; commit.\nAction: ANSWER {expr}"


def _oddcoin_step(messages: list[dict], env: envs.OddCoin) -> str:
    obs, acts = _obs_and_acts(messages)
    cands = list(range(1, env.n + 1))
    hist: list[tuple[list[int], list[int], str]] = []
    for a, o in zip(acts, obs[1:]):
        left, right = [[int(x) for x in re.findall(r"\d+", s)] for s in a.split(None, 1)[1].split("|")]
        res = "LEFT" if "LEFT" in o else ("RIGHT" if "RIGHT" in o else "BALANCED")
        hist.append((left, right, res))
        cands = [c for c in cands if (c in left if res == "LEFT" else (c in right if res == "RIGHT" else (c not in left and c not in right)))]

    facts = f"FACTS: Suspects S={cands} ({len(cands)} left)." if not hist else f"FACTS: Last={hist[-1][2]} => suspects S={cands} ({len(cands)} left)."
    if len(cands) == 1:
        ans = cands[0]
        vers = ", ".join(f"w{i}:{'L' if ans in l else ('R' if ans in r else 'B')}=={res[0]} OK" for i, (l, r, res) in enumerate(hist, 1))
        return f"Thought:\n{facts}\nCHECK: Candidate {ans}: {vers}.\nPLAN: Unique verified coin {ans}; commit.\nAction: ANSWER {ans}"

    k = (len(cands) + 2) // 3
    L, R, U = cands[:k], cands[k:2 * k], cands[2 * k:]
    ls, rs = " ".join(map(str, L)), " ".join(map(str, R))
    return (
        f"Thought:\n{facts}\n"
        f"CHECK: {len(cands)} left; compare equal pan sizes {k} vs {k} (never unequal counts).\n"
        f"PLAN: Weigh [{ls}] vs [{rs}] (off={U}).\nAction: WEIGH {ls} | {rs}"
    )


def process_worked(messages: list[dict], env: envs.Env) -> str:
    if type(env) not in envs.TRAIN_GAMES:
        raise ValueError(f"process_worked forbidden outside TRAIN_GAMES: got {env.name}")
    if isinstance(env, envs.GuessNumber):
        return _guessnumber_step(messages, env)
    if isinstance(env, envs.CodeBreak):
        return _codebreak_step(messages, env)
    if isinstance(env, envs.HiddenRule):
        return _hiddenrule_step(messages, env)
    if isinstance(env, envs.OddCoin):
        return _oddcoin_step(messages, env)
    raise ValueError(env)
