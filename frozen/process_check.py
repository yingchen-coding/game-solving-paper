"""Process-verified worked examples on 8 practice games, with the check made an ACTION.

The thoughts show the arithmetic of every check (FACTS / CHECK / PLAN, the other agent's process_teacher for the
four original games, extended here to the four extra games), and before every committing move the expert calls
the free CHECK tool on that move. Tests whether practising both the computation and the tool call, on more kinds
of problem, transfers to problems never practised. Reads only the visible transcript plus evidence-derived state.
"""
from __future__ import annotations

import re

import agents
import envs
import process_teacher

COMMIT = {"GuessNumber": "GUESS", "CodeBreak": "GUESS", "HiddenRule": "ANSWER", "OddCoin": "ANSWER",
          "Heaviest": "ANSWER", "Radar": "ANSWER", "Boxes": "ANSWER", "Switches": "ANSWER"}


def _heaviest(env):
    alive = [x for x in envs.ITEMS if x not in env.lost]
    wins = re.findall(r"([A-F]) vs ([A-F]): ([A-F]) is heavier", "\n".join(env.evidence))
    hist = "; ".join(f"{a} vs {b} -> {h} heavier, so {a if h == b else b} is out" for a, b, h in wins) or "none"
    facts = f"FACTS: {hist}. Never lost: {', '.join(alive)} ({len(alive)} left)."
    if len(alive) == 1:
        x = alive[0]
        losses = ", ".join(f"{a} vs {b}: {x} {'won' if h == x else 'not in it'}" for a, b, h in wins)
        return (f"{facts}\nCHECK: Candidate {x}: {losses}; {x} never lost, every other box lost once.\n"
                f"PLAN: Unique verified box {x}; commit."), f"ANSWER {x}"
    seen = {c for a, b, _ in wins for c in (a, b)}
    champ = next((x for x in alive if x in seen), alive[0])
    other = next(x for x in alive if x != champ)
    return (f"{facts}\nCHECK: {len(alive)} boxes could still be heaviest; answering now would be a guess.\n"
            f"PLAN: Compare {champ} with {other}; the loser is ruled out."), f"COMPARE {champ} {other}"


def _radar(env):
    cands = env.cands()
    hist = "; ".join(f"PING {r} {c} -> {d} means |row-{r}|+|col-{c}|={d}" for r, c, d in env.pings) or "none"
    facts = f"FACTS: {hist}. Cells fitting all pings: {len(cands)}."
    if len(cands) == 1:
        r, c = cands[0]
        vers = ", ".join(f"|{r}-{pr}|+|{c}-{pc}|={abs(r - pr) + abs(c - pc)}=={d} OK" for pr, pc, d in env.pings)
        return f"{facts}\nCHECK: Candidate ({r},{c}): {vers}.\nPLAN: Unique verified cell; commit.", f"ANSWER {r} {c}"
    q = agents._best_question(env, cands)
    shown = ", ".join(f"({r},{c})" for r, c in cands[:6]) + (" ..." if len(cands) > 6 else "")
    return (f"{facts}\nCHECK: Still possible: {shown}; more than one cell fits, so I do not answer.\n"
            f"PLAN: {q} splits these cells into the smallest worst-case group."), q


def _perm_problem(env, label, unit):
    cands = env.cands()
    facts = f"FACTS: {len(env.evidence)} answers so far; arrangements fitting all of them: {len(cands)} of 24."
    if len(cands) == 1:
        q = cands[0]
        if isinstance(env, envs.Boxes):
            vers = ", ".join(f"{envs.TOKENS[i]} in {b}? {'YES' if y else 'NO'}: arrangement says "
                             f"{'YES' if q[i] == b else 'NO'} OK" for i, b, y in env.facts)
        else:
            vers = ", ".join(f"FLIP {' '.join(map(str, sorted(on)))}: predicts lamps "
                             f"{' '.join(map(str, sorted(q[s - 1] for s in on)))} == seen OK" for on, _ in env.flips)
        arg = " ".join(map(str, q))
        return f"{facts}\nCHECK: Candidate {arg}: {vers}.\nPLAN: Unique verified {label}; commit.", f"ANSWER {arg}"
    act = agents._best_question(env, cands)
    return (f"{facts}\nCHECK: {len(cands)} {label}s still fit; answering now would be a guess.\n"
            f"PLAN: {act} has the smallest worst-case group of remaining {unit}."), act


def _step(messages, env):
    if type(env) in envs.TRAIN_GAMES:
        thought, _, act = process_teacher.process_worked(messages, env).partition("\nAction: ")
        return thought.removeprefix("Thought:\n"), act.strip()
    if isinstance(env, envs.Heaviest):
        return _heaviest(env)
    if isinstance(env, envs.Radar):
        return _radar(env)
    if isinstance(env, envs.Boxes):
        return _perm_problem(env, "arrangement", "arrangements")
    if isinstance(env, envs.Switches):
        return _perm_problem(env, "pairing", "pairings")
    raise ValueError(f"process_check is only defined on practice games, got {env.name}")


def process_check(messages, env):
    kept, skip = [], False
    for m in messages:
        if skip:
            skip = False
            continue
        if m["role"] == "assistant" and agents.action_of(m["content"]).upper().startswith("CHECK"):
            skip = True
            continue
        kept.append(m)
    thought, act = _step(kept, env)
    verb, _, arg = act.partition(" ")
    if verb != COMMIT[type(env).__name__]:
        return f"Thought:\n{thought}\nAction: {act}"
    last = next((agents.action_of(m["content"]) for m in reversed(messages) if m["role"] == "assistant"), "")
    if last != f"CHECK {arg}":
        return f"Thought:\n{thought}\nBefore committing I run the free CHECK on {arg}.\nAction: CHECK {arg}"
    return f"Thought:\nThe CHECK tool also says {arg} is consistent with all the evidence; commit.\nAction: {act}"
