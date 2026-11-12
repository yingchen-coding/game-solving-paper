"""Policies that play the environments in envs.py, and the episode loop.

    oracle          scripted solver; proves every seed is solvable inside its step budget
    mlx:<repo>      local MLX model, optional LoRA adapter (the model we train)
    ollama:<name>   local Ollama model (larger untrained reference models)
"""
from __future__ import annotations

import json
import itertools
import re
import urllib.request

import envs

SYSTEM = {
    "plain": ("You are an agent solving a problem in a text environment. Each turn, think briefly, then give "
              "exactly one action on a final line that starts with 'Action:'."),
    "selfcheck": ("You are an agent solving a problem in a text environment. Each turn, think briefly, then give "
                  "exactly one action on a final line that starts with 'Action:'. Before every action, check your "
                  "candidate against EVERY observation so far, one line per observation, and act only if all agree."),
    "opposite": ("You are an agent solving a problem in a text environment. Each turn, think briefly, then give "
                 "exactly one action on a final line that starts with 'Action:'. Before committing to any candidate, "
                 "consider the opposite: say which observation would prove it wrong, and look for that observation in "
                 "what you have already seen."),
    "paraphrase": ("You are an agent solving a problem in a text environment. Each turn, think briefly, then give "
                   "exactly one action on a final line that starts with 'Action:'. Start every Thought by restating "
                   "exactly what the latest observation says and what it does NOT tell you, before drawing conclusions."),
    "strategy": ("You are an agent solving a problem in a text environment. Each turn, think briefly, then give "
                 "exactly one action on a final line that starts with 'Action:'. Solve it like a careful human: keep "
                 "track of what each observation rules out, choose the action that tells you the most, test a "
                 "hypothesis before committing to it, and only answer when the evidence pins it down."),
}


def action_of(reply: str) -> str:
    """The action is the text after the last 'Action:'; a reply without one is taken whole (and is usually invalid)."""
    hits = re.findall(r"(?im)^\s*\**action\**\s*:\s*(.+)$", reply)
    return hits[-1].strip() if hits else reply.strip()


def run_episode(env: envs.Env, policy, system: str = "plain", playbook: str | None = None) -> dict:
    sys_text = SYSTEM[system] + (f"\n\nLessons you learned from earlier problems:\n{playbook}" if playbook else "")
    messages = [{"role": "system", "content": sys_text},
                {"role": "user", "content": env.first_observation()}]
    while not env.finished:
        reply = policy(messages, env)
        messages.append({"role": "assistant", "content": reply})
        obs = env.step(action_of(reply))
        messages.append({"role": "user", "content": obs})
    rec = env.rec
    return {"env": env.name, "messages": messages, "success": rec.success, "asks": rec.asks,
            "invalid": rec.invalid, "steps": rec.steps, "budget": env.max_steps}


# ---------------------------------------------------------------- oracle
# Honest experts: they see only what the agent sees (the transcript and the opening text), never hidden state.

def _observations(messages):
    return [m["content"] for m in messages if m["role"] == "user"]


def oracle(messages, env):
    obs = _observations(messages)
    acts = [action_of(m["content"]) for m in messages if m["role"] == "assistant"]
    if isinstance(env, envs.GuessNumber):
        lo, hi = 1, env.hi
        for a, o in zip(acts, obs[1:]):
            g = int(a.split()[1])
            lo, hi = (g + 1, hi) if o.startswith("Higher") else (lo, g - 1)
        return f"GUESS {(lo + hi) // 2}"
    if isinstance(env, envs.Lock):
        cands = envs.all_codes(env.k, env.digits)
        for o in obs[1:]:
            m = re.match(r"([A-E ]+) -> IN PLACE=(\d) ELSEWHERE=(\d)", o)
            g = [envs.LETTERS.index(c) + 1 for c in m.group(1).split()]
            cands = [c for c in cands if envs.CodeBreak.score(c, g) == (int(m.group(2)), int(m.group(3)))]
        return "TRY " + " ".join(envs.LETTERS[d - 1] for d in cands[0])
    if isinstance(env, envs.Machine):
        vals = [int(re.search(r"= (-?\d+)", o).group(1)) for o in obs[1:]]
        if len(vals) < 3:
            return f"INPUT {len(vals)}"
        c = vals[0]
        a = (vals[2] - 2 * vals[1] + c) // 2
        b = vals[1] - c - a
        return f"SUBMIT {a} * n * n + {b} * n + {c}"
    if isinstance(env, envs.CodeBreak):
        cands = envs.all_codes(env.k, env.digits)
        for o in obs[1:]:
            m = re.match(r"([\d ]+) -> EXACT=(\d) MISPLACED=(\d)", o)
            g, e, mis = [int(d) for d in m.group(1).split()], int(m.group(2)), int(m.group(3))
            cands = [c for c in cands if envs.CodeBreak.score(c, g) == (e, mis)]
        return "GUESS " + " ".join(map(str, cands[0]))
    if isinstance(env, envs.HiddenRule):
        vals = [int(re.search(r"= (-?\d+)", o).group(1)) for o in obs[1:]]
        if len(vals) < 3:
            return f"QUERY {len(vals)}"
        c = vals[0]
        a = (vals[2] - 2 * vals[1] + c) // 2
        b = vals[1] - c - a
        return f"ANSWER {a} * x * x + {b} * x + {c}"
    if isinstance(env, envs.OddCoin):
        cands = list(range(1, env.n + 1))
        for a, o in zip(acts, obs[1:]):
            left, right = [[int(x) for x in re.findall(r"\d+", s)] for s in a.split(None, 1)[1].split("|")]
            cands = [c for c in cands if (c in left) == ("LEFT" in o) and (c in right) == ("RIGHT" in o)]
        if len(cands) == 1:
            return f"ANSWER {cands[0]}"
        k = (len(cands) + 2) // 3
        return f"WEIGH {' '.join(map(str, cands[:k]))} | {' '.join(map(str, cands[k:2 * k]))}"
    if isinstance(env, envs.Circuit):
        spec = re.findall(r"stage \d+: (add|multiply by|subtract) (\d+)", obs[0])
        x0 = int(re.search(r"Input = (\d+)", obs[0]).group(1))
        ops = {"add": lambda v, c: v + c, "multiply by": lambda v, c: v * c, "subtract": lambda v, c: v - c}
        expect, v = [], x0
        for name, c in spec:
            v = ops[name](v, int(c))
            expect.append(v)
        lo, hi = 1, len(spec)  # the faulty stage is in [lo, hi]
        for o in obs[1:]:
            m = re.search(r"value after stage (\d+) = (-?\d+)", o)
            k, val = int(m.group(1)), int(m.group(2))
            lo, hi = (lo, k) if val != expect[k - 1] else (k + 1, hi)
        return f"ANSWER {lo}" if lo == hi else f"PROBE {(lo + hi) // 2}"
    if isinstance(env, envs.BugFix):
        return _bugfix_expert(acts, obs)
    if isinstance(env, envs.Heaviest):
        alive = [x for x in envs.ITEMS if x not in env.lost]
        seen = {box for pair in re.findall(r"([A-F]) vs ([A-F])", "\n".join(obs)) for box in pair}
        if len(alive) == 1:
            return f"ANSWER {alive[0]}"
        champ = next((x for x in alive if x in seen), alive[0])
        other = next(x for x in alive if x != champ)
        return f"COMPARE {champ} {other}"
    if isinstance(env, (envs.Radar, envs.Boxes, envs.Switches)):
        cands = env.cands()
        if len(cands) == 1:
            c = cands[0]
            return "ANSWER " + " ".join(map(str, c))
        return _best_question(env, cands)
    raise ValueError(env)


def _best_question(env, cands):
    """The question whose worst-case answer leaves the fewest candidates (minimax split)."""
    if isinstance(env, envs.Radar):
        qs = [((r, c), "PING %d %d" % (r, c)) for r in range(1, 7) for c in range(1, 7)]
        outcome = lambda q, t: abs(q[0] - t[0]) + abs(q[1] - t[1])
    elif isinstance(env, envs.Boxes):
        qs = [((i, b), "IS %s %d" % (envs.TOKENS[i], b)) for i in range(4) for b in range(1, 5)]
        outcome = lambda q, t: t[q[0]] == q[1]
    else:
        qs = [(on, "FLIP " + " ".join(map(str, on))) for k in (1, 2) for on in itertools.combinations(range(1, 5), k)]
        outcome = lambda q, t: frozenset(t[s - 1] for s in q)
    def worst(q):
        counts = {}
        for t in cands:
            o = outcome(q, t)
            counts[o] = counts.get(o, 0) + 1
        return max(counts.values())
    return min(qs, key=lambda q: worst(q[0]))[1]


def worked_example(messages, env):
    """The honest expert, with its reasoning written out (a worked example in Sweller's sense). The
    explanation uses only evidence-derived state, and every answer is preceded by an explicit check."""
    act = oracle(messages, env)
    if isinstance(env, envs.GuessNumber):
        why = (f"The feedback so far says the number is between {env.lo_seen} and {env.hi_seen}. Guessing the middle "
               "halves what is left.")
    elif isinstance(env, envs.CodeBreak):
        cands = [c for c in envs.all_codes(env.k, env.digits) if all(env.score(c, g) == fb for g, fb in env.past)]
        why = (f"{len(cands)} codes are still consistent with every piece of feedback. I check my next guess against "
               "each earlier guess before playing it, and only guess a code that could still be right.")
    elif isinstance(env, envs.HiddenRule):
        if act.startswith("QUERY"):
            why = (f"A polynomial of degree at most 2 has 3 unknown coefficients, so I need 3 points; I have "
                   f"{len(env.points)}.")
        else:
            checks = ", ".join(f"f({x}) = {v}" for x, v in env.points.items())
            why = (f"Solving with c = f(0), a + b = f(1) - c and 4a + 2b = f(2) - c. Check against every point: {checks} "
                   "all agree with the formula, so I answer.")
    elif isinstance(env, envs.OddCoin):
        cands = sorted(env.cands)
        why = (f"Coins still possible: {cands}. " + ("Only one is left and it agrees with every weighing." if len(cands) == 1
               else "Splitting them into three equal groups means any result removes two thirds."))
    elif isinstance(env, envs.Heaviest):
        alive = [x for x in envs.ITEMS if x not in env.lost]
        why = (f"Boxes that have never lost a comparison: {', '.join(alive)}. " + ("Only one is left, so it is the "
               "heaviest." if len(alive) == 1 else "I compare two of them; the lighter one can never be the heaviest."))
    elif isinstance(env, (envs.Radar, envs.Boxes, envs.Switches)):
        n = len(env.cands())
        why = (f"{n} possibilit{'y' if n == 1 else 'ies'} still fit{'s' if n == 1 else ''} every observation. " + ("Only one is left and it agrees with everything I "
               "saw." if n == 1 else "I ask the question whose worst answer still rules out the most possibilities."))
    else:
        why = "Following the evidence."
    return f"Thought: {why}\nAction: {act}"


COMMIT_VERBS = {"GuessNumber": "GUESS", "CodeBreak": "GUESS", "HiddenRule": "ANSWER", "OddCoin": "ANSWER",
                "Heaviest": "ANSWER", "Radar": "ANSWER", "Boxes": "ANSWER", "Switches": "ANSWER"}


def worked_check(messages, env):
    """Worked example that makes the check an ACTION: before every committing move it calls the free CHECK tool on
    the move it is about to make. The habit is anchored to a tool the held-out tasks also have, not to prose."""
    kept, skip = [], False
    for m in messages:
        if skip:
            skip = False
            continue
        if m["role"] == "assistant" and action_of(m["content"]).upper().startswith("CHECK"):
            skip = True
            continue
        kept.append(m)
    thought, _, act = worked_example(kept, env).partition("\nAction: ")
    verb, _, arg = act.partition(" ")
    if verb != COMMIT_VERBS.get(type(env).__name__):
        return f"{thought}\nAction: {act}"
    last = next((action_of(m["content"]) for m in reversed(messages) if m["role"] == "assistant"), "")
    if last != f"CHECK {arg}":
        return (f"{thought} Before I commit, I test {arg} against all the evidence with the free check.\n"
                f"Action: CHECK {arg}")
    return f"Thought: The check says {arg} is consistent with all the evidence, so I commit.\nAction: {act}"


def _bugfix_expert(acts, obs):
    if not acts:
        return "RUN"
    last = obs[-1]
    fails = sorted({int(n) for n in re.findall(r"FAIL f(\d+)\(", "\n".join(obs))})
    shown = {int(m.group(1)): m.group(2) for o in obs for m in [re.search(r"def f(\d+)\(x\):\n    return (.+)", o)] if m}
    patched = {int(re.search(r"PATCH (\d+)", a).group(1)) for a in acts if a.startswith("PATCH")}
    if acts[-1].startswith("PATCH") and not acts[-1].startswith("RUN"):
        return "RUN"
    if "visible tests pass." in last and "FAIL" not in last and acts[-1] == "RUN":
        return "DONE"
    latest_run = next(o for o in reversed(obs) if "visible tests pass" in o)
    failing_now = sorted({int(n) for n in re.findall(r"FAIL f(\d+)\(", latest_run)})
    for n in failing_now:
        if n not in shown:
            return f"SHOW {n}"
        tests = [(int(x), int(w)) for x, w in re.findall(rf"f{n}\((-?\d+)\): expected (-?\d+)", "\n".join(obs))]
        return f"PATCH {n} {_repair(shown[n], tests)}"
    return "RUN"


def _repair(body, tests):
    """Smallest edit of the body (by token difference) among bug-template repairs that fits the visible tests."""
    def fits(expr):
        try:
            return all(eval(expr, {"__builtins__": {}}, {"x": x}) == want for x, want in tests)
        except Exception:
            return False
    best = None
    for good_t, _ in envs.BUGS:
        for a in range(1, 13):
            for b in range(1, 13):
                expr = good_t.format(a=a, b=b)
                if fits(expr):
                    d = len(set(expr.split()) ^ set(body.split()))
                    if best is None or d < best[0]:
                        best = (d, expr)
    return best[1] if best else body


# ---------------------------------------------------------------- MLX

class MLXPolicy:
    def __init__(self, repo: str, adapter: str | None = None, temp: float = 0.0, max_tokens: int = 256):
        from mlx_lm import load
        from mlx_lm.sample_utils import make_sampler
        self.model, self.tok = load(repo, adapter_path=adapter)
        self.sampler = make_sampler(temp=temp)
        self.max_tokens = max_tokens
        self.use_cache = True
        self.prefix = "Thought:"
        self._env, self._ctx, self._cache = None, [], None

    def __call__(self, messages, env):
        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
        toks = self.tok.apply_chat_template(messages, add_generation_prompt=True)
        # Small models skip the reasoning the system prompt asks for; starting the reply with "Thought:"
        # makes them reason before acting. The prefix is part of the returned reply (and of training data).
        toks = toks + self.tok.encode(self.prefix, add_special_tokens=False)
        # A chat transcript only grows, so the KV cache of the previous turn's prompt is a prefix of this
        # turn's prompt; only the new tokens are prefilled. Without this a 40-turn episode costs minutes.
        if not self.use_cache or self._env is not env or toks[:len(self._ctx)] != self._ctx:
            self._env, self._ctx, self._cache = env, [], make_prompt_cache(self.model)
        new = toks[len(self._ctx):]
        out = ""
        for chunk in stream_generate(self.model, self.tok, prompt=mx.array(new), max_tokens=self.max_tokens,
                                     sampler=self.sampler, prompt_cache=self._cache):
            out += chunk.text
            if re.search(r"(?im)^\s*\**action\**\s*:.+\n", out):
                break
        extra = self._cache[0].offset - len(toks)
        if extra > 0:
            trim_prompt_cache(self._cache, extra)
        self._ctx = toks if self._cache[0].offset == len(toks) else []
        return (self.prefix + out).strip()


def complete(policy, messages, max_tokens=300) -> str:
    """Free-form text from an MLX policy (used by the reflection learner), without the action stop rule."""
    from mlx_lm import generate
    prompt = policy.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return generate(policy.model, policy.tok, prompt=prompt, max_tokens=max_tokens, sampler=policy.sampler).strip()


# ---------------------------------------------------------------- Ollama

class OllamaPolicy:
    def __init__(self, name: str, temp: float = 0.0):
        self.name, self.temp = name, temp

    def __call__(self, messages, env):
        body = json.dumps({"model": self.name, "messages": messages, "stream": False,
                           "options": {"temperature": self.temp, "num_predict": 200}}).encode()
        req = urllib.request.Request("http://localhost:11434/api/chat", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            out = json.load(r)["message"]["content"].strip()
        return out


def make_policy(spec: str, adapter: str | None = None, temp: float = 0.0):
    if spec == "oracle":
        return oracle
    if spec == "worked":
        return worked_example
    if spec == "worked_check":
        return worked_check
    if spec == "process_check":
        import process_check  # imports agents itself, so it cannot be imported at module top
        return process_check.process_check
    kind, _, name = spec.partition(":")
    if kind == "mlx":
        return MLXPolicy(name, adapter, temp)
    if kind == "ollama":
        return OllamaPolicy(name, temp)
    raise ValueError(spec)
