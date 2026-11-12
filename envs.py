"""Interactive problems that need reasoning from feedback: deduce, test a hypothesis, diagnose.

Training games (the model may learn from these):
    GuessNumber   find a hidden number from higher/lower feedback          (search)
    CodeBreak     Mastermind: deduce a hidden code from exact/misplaced     (deduction)
    HiddenRule    query a black-box function, then state its formula        (induction)
    OddCoin       find the heavy coin with a balance in few weighings        (information-efficient tests)

Held-out problems (never used for training or prompt tuning):
    Circuit       a pipeline of stages, one is faulty; probe to localize it  (diagnosis, new mechanic)
    BugFix        a tiny repo with failing tests; read, patch, rerun; graded on HIDDEN tests
                  (root-cause debugging: the shape of a real PR)

Each turn the agent emits one action line. Every problem's verifier is exact; nothing here calls a model.
"""
from __future__ import annotations

import itertools
import random
import re
from dataclasses import dataclass


@dataclass
class Record:
    success: bool = False
    answered: bool = False
    asks: int = 0
    invalid: int = 0
    steps: int = 0
    coach_notes: int = 0


class Env:
    name = "env"
    actions_help = ""
    check_help = "<candidate>"

    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.rec = Record()
        self.finished = False
        self.coach = False  # practice-only scaffold (H6); never on at test time
        self.teacher: str | None = None  # one of TEACHERS; adds a teacher line to every observation
        self.last_note: str | None = None
        self.ledger = False       # show all evidence so far every turn (memory aid)
        self.allow_check = False  # CHECK <candidate>: free consistency test against the evidence (verifier tool)
        self.checks = 0
        self.evidence: list[str] = []
        self.seen_actions: set[str] = set()
        self.refused: set[str] = set()
        self.setup()
        self.max_steps = self.budget()

    def setup(self): ...
    def budget(self) -> int: ...
    def goal(self) -> str: ...
    def opening(self) -> str:
        return ""
    def solved(self) -> bool: ...
    def reveal(self) -> str:
        """The hidden answer, shown to the learner after an episode (as a person sees it after a game)."""
        return ""
    def apply(self, verb: str, args: list[str]) -> str | None:
        """Return feedback text, or None if the action is invalid. May set self.finished."""

    def evidence_check(self, verb: str, args: list[str]) -> tuple[str | None, bool]:
        """Per-game coach: (fact the action contradicts or ignores, whether to refuse it). Facts only."""
        return None, False

    def check(self, args: list[str]) -> str:
        """Verifier tool: is the candidate consistent with all evidence so far? Never reveals the answer."""
        return "not available here."

    def _coach(self, verb, args):
        key = " ".join(self.line.upper().split())
        note, refuse = self.evidence_check(verb, args)
        if note is None and key in self.seen_actions and verb not in ("RUN", "ASK"):
            note = "this action repeats one you already took; it gives no new information."
        self.seen_actions.add(key)
        if refuse and key in self.refused:
            refuse = False
        if refuse:
            self.refused.add(key)
        if note:
            self.rec.coach_notes += 1
        return note, refuse

    def first_observation(self) -> str:
        tool = (f"\nTool: CHECK {self.check_help} tests a candidate against all evidence so far; it is free "
                "(max 12) and does not use an action." if self.allow_check else "")
        text = (f"TASK: {self.goal()}\n\nActions: {self.actions_help}. {tool}\n"
                f"You have {self.max_steps} actions. Reply with exactly one action per turn.")
        return text + (f"\n\n{self.opening()}" if self.opening() else "")

    def step(self, text: str) -> str:
        self.rec.steps += 1
        line = next((ln.strip() for ln in text.strip().splitlines() if ln.strip()), "")
        self.line = line.strip("`*> ").strip()
        parts = self.line.split()
        verb = parts[0].upper().rstrip(":.") if parts else ""
        args = [a.strip("[](),") for a in parts[1:]]
        if verb == "CHECK" and self.allow_check:
            self.rec.steps -= 1  # checking is free, like running a test locally; capped below
            self.checks += 1
            if self.checks > 12:
                return "No checks left. Act."
            return "Check: " + self.check(args)
        note, refuse = self._coach(verb, args) if self.coach or self.teacher == "coach" else (None, False)
        if self.teacher == "coach":
            self.last_note, note, refuse = note, None, False
        if verb == "ASK":
            self.rec.asks += 1
            feedback = "The user is away and cannot answer. Continue on your own."
        elif refuse:
            feedback = f"Coach: {note} Answer not accepted; try again."
            note = None
        else:
            feedback = self.apply(verb, args)
            if feedback is None:
                self.rec.invalid += 1
                feedback = f"Invalid action: {self.line[:60]!r}."
        if note:
            feedback += f"\nCoach: {note}"
        if self.finished:
            self.rec.success = self.solved()
            return feedback + "\nEpisode over."
        left = self.max_steps - self.rec.steps
        if left <= 0:
            self.finished = True
            self.rec.success = self.solved()
            return feedback + "\nOut of actions. Episode over."
        first = feedback.split("\n")[0]
        if not first.startswith(("Invalid", "Coach")):
            self.evidence.append(f"{self.line} -> {first}")
        if self.ledger and self.evidence:
            feedback += "\nEvidence so far:\n" + "\n".join(f"  {e}" for e in self.evidence)
        if self.teacher:
            feedback += "\nTeacher: " + teacher_line(self)
        return f"{feedback}\n({left} actions left)"


def _int(s) -> int | None:
    """First integer in the text, so 'x = 3', 'stage 4' and '4.' all read as intended; '4.5' is rejected."""
    if isinstance(s, list):
        s = " ".join(s)
    m = re.search(r"-?\d+(\.\d+)?", s or "")
    if not m or m.group(1):
        return None
    return int(m.group(0))


def _expr(text: str) -> str:
    """A formula from free text: a quoted part if there is one, else the leading run of formula characters."""
    q = re.search(r"['\"`](.+?)['\"`]", text)
    text = q.group(1) if q else text
    text = text.split("=", 1)[1] if "=" in text else text
    text = text.strip().removeprefix("return ").replace("^", "**").replace("X", "x")
    text = re.sub(r"(\d)\s*x", r"\1*x", text)
    m = re.match(r"[x0-9+\-*() ]+", text)
    return m.group(0).strip() if m else ""


# ---------------------------------------------------------------- teachers
# Content-free styles (encourage, yell) carry no information about the problem; remind restates only the
# goal; hint asks a generic question; coach states a fact derived from the transcript (see _coach).

ENCOURAGE = ["You're doing great, keep going!", "Nice work, you can do this!", "Good effort, I believe in you!",
             "Great thinking, stay with it!"]
YELL = ["Wrong! Think harder.", "That's not good enough. Concentrate!", "Sloppy. Do better on the next move.",
        "Come on, you're wasting moves!"]
PRAISE_PROCESS = ["Good: you used the evidence carefully.", "Nice process: checking your idea before acting pays off.",
                  "You are working step by step; keep checking.", "Careful reasoning; keep testing your idea."]
PRAISE_ABILITY = ["You're so smart!", "You're a genius at this!", "You're naturally gifted at puzzles!",
                  "Brilliant mind, you'll get it!"]
HINTS = ["What does the last result rule out?", "Which possibilities are still consistent with everything you saw?",
         "Which action would tell you the most right now?", "Does your current idea agree with every observation so far?"]
TEACHERS = ("encourage", "yell", "remind", "hint", "coach", "praise_process", "praise_ability")


def teacher_line(env: "Env") -> str:
    i = env.rec.steps
    if env.teacher == "encourage":
        return ENCOURAGE[i % len(ENCOURAGE)]
    if env.teacher == "yell":
        return YELL[i % len(YELL)]
    if env.teacher == "praise_process":
        return PRAISE_PROCESS[i % len(PRAISE_PROCESS)]
    if env.teacher == "praise_ability":
        return PRAISE_ABILITY[i % len(PRAISE_ABILITY)]
    if env.teacher == "remind":
        return f"Remember your task: {env.goal()}"
    if env.teacher == "hint":
        return HINTS[i % len(HINTS)]
    if env.teacher == "coach":
        return env.last_note or "No contradiction with your evidence so far."
    raise ValueError(env.teacher)


# ---------------------------------------------------------------- training games

class GuessNumber(Env):
    name = "guessnumber"
    actions_help = "GUESS n, e.g. GUESS 50"

    def reveal(self):
        return f"The hidden number was {self.secret}."

    def setup(self):
        self.hi = self.rng.choice([100, 200])
        self.secret = self.rng.randint(1, self.hi)
        self.got = False
        self.lo_seen, self.hi_seen = 1, self.hi

    check_help = "n"

    def check(self, args):
        g = _int(args)
        if g is None:
            return "give a number."
        return "consistent." if self.lo_seen <= g <= self.hi_seen else f"inconsistent: the number is between {self.lo_seen} and {self.hi_seen}."

    def evidence_check(self, verb, args):
        g = _int(args) if verb == "GUESS" else None
        if g is not None and not self.lo_seen <= g <= self.hi_seen:
            return f"the feedback so far says the number is between {self.lo_seen} and {self.hi_seen}.", False
        return None, False

    def budget(self):
        return 8 if self.hi == 100 else 9

    def goal(self):
        return f"Find the hidden integer between 1 and {self.hi}. After each guess you are told higher or lower."

    def solved(self):
        return self.got

    def apply(self, verb, args):
        g = _int(args) if verb == "GUESS" else None
        if g is None:
            return None
        if g == self.secret:
            self.got, self.finished = True, True
            return "Correct!"
        if self.secret > g:
            self.lo_seen = max(self.lo_seen, g + 1)
            return "Higher."
        self.hi_seen = min(self.hi_seen, g - 1)
        return "Lower."


class CodeBreak(Env):
    name = "codebreak"
    actions_help = "GUESS d d d, e.g. GUESS 1 2 3"

    def reveal(self):
        return "The hidden code was " + " ".join(map(str, self.secret)) + "."

    def setup(self):
        self.k, self.digits = 3, 5
        self.secret = [self.rng.randint(1, self.digits) for _ in range(self.k)]
        self.got = False
        self.past: list[tuple[list[int], tuple[int, int]]] = []

    def parse(self, args):
        digits = [c for c in "".join(args) if c.isdigit()]
        if len(digits) != self.k or any(not 1 <= int(d) <= self.digits for d in digits):
            return None
        return [int(d) for d in digits]

    check_help = "d d d"

    def check(self, args):
        g = self.parse(args)
        if g is None:
            return f"give {self.k} digits."
        for pg, fb in self.past:
            if self.score(g, pg) != fb:
                return (f"inconsistent: against guess {' '.join(map(str, pg))} it would score "
                        f"EXACT={self.score(g, pg)[0]} MISPLACED={self.score(g, pg)[1]}, observed EXACT={fb[0]} MISPLACED={fb[1]}.")
        return "consistent with all feedback so far."

    def evidence_check(self, verb, args):
        g = self.parse(args) if verb == "GUESS" else None
        for pg, fb in self.past if g else []:
            if self.score(g, pg) != fb:
                return (f"this code cannot be the answer: if it were, guess {' '.join(map(str, pg))} would have "
                        f"scored EXACT={self.score(g, pg)[0]} MISPLACED={self.score(g, pg)[1]}, not "
                        f"EXACT={fb[0]} MISPLACED={fb[1]}."), False
        return None, False

    def budget(self):
        return 10

    def goal(self):
        return (f"Deduce a hidden code of {self.k} digits, each 1-{self.digits} (repeats allowed). After each guess "
                "you get EXACT = digits right and in the right place, and MISPLACED = right digit, wrong place.")

    def solved(self):
        return self.got

    @staticmethod
    def score(secret, guess):
        exact = sum(a == b for a, b in zip(secret, guess))
        common = sum(min(secret.count(d), guess.count(d)) for d in set(guess))
        return exact, common - exact

    def apply(self, verb, args):
        if verb != "GUESS":
            return None
        guess = self.parse(args)
        if guess is None:
            return None
        exact, mis = self.score(self.secret, guess)
        self.past.append((guess, (exact, mis)))
        if exact == self.k:
            self.got, self.finished = True, True
            return "EXACT=3. Code cracked!"
        return f"{' '.join(map(str, guess))} -> EXACT={exact} MISPLACED={mis}"


RULES = [("{a} * x + {b}", 2), ("x * x + {b}", 1), ("{a} * x - {b}", 2), ("x * x - {a} * x", 1),
         ("{b} - {a} * x", 2), ("x * x + {a} * x + {b}", 3)]


class HiddenRule(Env):
    name = "hiddenrule"
    actions_help = "QUERY x (see f(x), x an integer), ANSWER expr in x, e.g. ANSWER 2 * x + 1"

    def reveal(self):
        return f"The hidden function was f(x) = {self.rule}."

    def setup(self):
        tmpl, _ = self.rng.choice(RULES)
        self.rule = tmpl.format(a=self.rng.randint(2, 9), b=self.rng.randint(1, 20))
        self.answer = None
        self.points: dict[int, int] = {}

    check_help = "expr"

    def check(self, args):
        expr = _expr(" ".join(args))
        if not expr:
            return "give a formula in x."
        bad = []
        for x, v in self.points.items():
            try:
                got = self.f(expr, x)
            except Exception:
                return "could not evaluate that formula."
            if got != v:
                bad.append(f"f({x}): formula gives {got}, observed {v}")
        return "consistent with all queried points." if not bad else "inconsistent: " + "; ".join(bad)

    def evidence_check(self, verb, args):
        if verb != "ANSWER":
            return None, False
        rest = self.line.split(None, 1)
        expr = _expr(rest[1]) if len(rest) == 2 else ""
        for x, v in self.points.items():
            try:
                got = self.f(expr, x)
            except Exception:
                return None, False
            if got != v:
                return f"your formula gives f({x}) = {got}, but you observed f({x}) = {v}.", True
        return None, False

    def budget(self):
        return 6

    def goal(self):
        return ("A hidden function f(x) is a polynomial in x with integer coefficients, degree at most 2. "
                "Query it, then ANSWER its formula. The answer is checked on many hidden inputs; one ANSWER only.")

    def f(self, expr, x):
        return eval(expr, {"__builtins__": {}}, {"x": x})

    def solved(self):
        if not self.answer:
            return False
        try:
            return all(self.f(self.answer, x) == self.f(self.rule, x) for x in range(-7, 12))
        except Exception:
            return False

    def apply(self, verb, args):
        if verb == "QUERY":
            x = _int(args)
            if x is None or abs(x) > 1000:
                return None
            self.points[x] = self.f(self.rule, x)
            return f"f({x}) = {self.points[x]}"
        if verb == "ANSWER":
            rest = self.line.split(None, 1)
            expr = _expr(rest[1]) if len(rest) == 2 else ""
            if not expr or "x" not in expr and not re.fullmatch(r"-?\d+", expr):
                return None
            self.answer, self.finished = expr, True
            return "Answer recorded."
        return None


class OddCoin(Env):
    name = "oddcoin"
    actions_help = ("WEIGH left | right with coin numbers, e.g. WEIGH 1 2 3 | 4 5 6; "
                    "ANSWER n names the heavy coin (one ANSWER only)")

    def reveal(self):
        return f"The heavy coin was {self.heavy}."

    def setup(self):
        self.n = self.rng.choice([9, 12])
        self.heavy = self.rng.randint(1, self.n)
        self.answer = None
        self.cands = set(range(1, self.n + 1))

    check_help = "k"

    def check(self, args):
        k = _int(args)
        if k is None:
            return "give a coin number."
        return "consistent: coin %d could still be heavy." % k if k in self.cands else f"inconsistent: coin {k} is ruled out."

    def evidence_check(self, verb, args):
        if verb == "ANSWER":
            k = _int(args)
            if k is not None and k not in self.cands:
                return f"coin {k} was already ruled out by your weighings.", True
        if verb == "WEIGH":
            coins = {int(c) for c in re.findall(r"\d+", self.line)}
            if coins and not coins & self.cands:
                return "every coin on the balance is already ruled out; this weighing gives no new information.", False
        return None, False

    def budget(self):
        return 4 if self.n == 9 else 5

    def goal(self):
        return (f"There are {self.n} coins numbered 1-{self.n}. Exactly one is heavier; the rest weigh the same. "
                "Use the balance to find the heavy coin, then ANSWER it.")

    def solved(self):
        return self.answer == self.heavy

    def apply(self, verb, args):
        if verb == "ANSWER":
            k = _int(args)
            if k is None:
                return None
            self.answer, self.finished = k, True
            return "Answer recorded."
        if verb != "WEIGH":
            return None
        sides = re.split(r"\||\bvs\.?\b|\bversus\b|\bagainst\b|\bright\b", self.line.split(None, 1)[1] if " " in self.line else "", 1, flags=re.I)
        if len(sides) != 2:
            return None
        left, right = sides
        left, right = [_int(x) for x in re.findall(r"\d+", left)], [_int(x) for x in re.findall(r"\d+", right)]
        if not left or not right or set(left) & set(right) or any(not 1 <= c <= self.n for c in left + right):
            return None
        wl, wr = len(left) + (self.heavy in left), len(right) + (self.heavy in right)
        side = "LEFT is heavier" if wl > wr else "RIGHT is heavier" if wr > wl else "BALANCED"
        on = set(left) if wl > wr else set(right) if wr > wl else set()
        self.cands = self.cands & on if on else self.cands - set(left) - set(right)
        return f"{' '.join(map(str, left))} vs {' '.join(map(str, right))}: {side}"


# ---------------------------------------------------------------- held-out problems

STAGE_OPS = [("add", lambda v, c: v + c), ("multiply by", lambda v, c: v * c), ("subtract", lambda v, c: v - c)]


class Circuit(Env):
    """A chain of stages transforms an input. One stage is faulty (uses a wrong constant). Localize it."""
    name = "circuit"
    actions_help = "PROBE i shows the value after stage i, e.g. PROBE 4; ANSWER i names the faulty stage (one ANSWER only)"

    def setup(self):
        self.n = 8
        self.x0 = self.rng.randint(1, 9)
        self.stages = []
        for _ in range(self.n):
            name, fn = self.rng.choice(STAGE_OPS)
            c = self.rng.randint(2, 3) if name == "multiply by" else self.rng.randint(1, 9)
            self.stages.append((name, fn, c))
        self.bad = self.rng.randint(1, self.n)
        self.answer = None
        # A fault is only findable if it changes a value (multiplying 0 hides it); resample otherwise.
        if self.value(self.bad, True) == self.value(self.bad, False):
            self.setup()

    def budget(self):
        return 5

    check_help = "i"

    def check(self, args):
        k = _int(args)
        if k is None or not 1 <= k <= self.n:
            return "give a stage number."
        return f"by the specification, the value after stage {k} should be {self.value(k, False)}."

    def value(self, upto, faulty):
        v = self.x0
        for i, (_, fn, c) in enumerate(self.stages[:upto], 1):
            v = fn(v, c + 1 if (faulty and i == self.bad) else c)
        return v

    def goal(self):
        return (f"A pipeline of {self.n} stages processes an input. One stage is faulty. Probe intermediate values to "
                "find it, then ANSWER its number.")

    def opening(self):
        spec = "\n".join(f"stage {i}: {name} {c}" for i, (name, _, c) in enumerate(self.stages, 1))
        return f"Input = {self.x0}. Specification:\n{spec}\nFinal output observed: {self.value(self.n, True)}."

    def solved(self):
        return self.answer == self.bad

    def apply(self, verb, args):
        k = _int(args)
        if k is None or not 1 <= k <= self.n:
            return None
        if verb == "PROBE":
            return f"value after stage {k} = {self.value(k, True)}"
        if verb == "ANSWER":
            self.answer, self.finished = k, True
            return "Answer recorded."
        return None


BUGS = [  # (correct body, buggy body) — the bug is an operator or a constant, as in a real one-line fix
    ("x * {a} + {b}", "x * {a} - {b}"), ("x * {a} + {b}", "x * {b} + {a}"), ("(x + {a}) * {b}", "x + {a} * {b}"),
    ("x * x - {a}", "x * x + {a}"), ("x * {a} - {b}", "x * {a} - {c}"), ("{a} * (x - {b})", "{a} * x - {b}"),
]


class BugFix(Env):
    """Functions with failing tests. The fix is graded on hidden inputs, so hard-coding a test value fails."""
    name = "bugfix"
    actions_help = ("RUN runs the visible tests; SHOW n shows function n, e.g. SHOW 2; "
                    "PATCH n expr replaces the body of function n, e.g. PATCH 2 x * 3 + 1; DONE when finished")

    def setup(self):
        k = 3
        self.good, self.funcs, self.tests = {}, {}, {}
        broken = set(self.rng.sample(range(1, k + 1), 2))
        for n in range(1, k + 1):
            good_t, bad_t = self.rng.choice(BUGS)
            a, b = self.rng.sample(range(2, 9), 2)
            c = b + 1
            self.good[n] = good_t.format(a=a, b=b)
            self.funcs[n] = bad_t.format(a=a, b=b, c=c) if n in broken else self.good[n]
            xs = self.rng.sample(range(1, 8), 2)
            self.tests[n] = [(x, self.f(self.good[n], x)) for x in xs]

    @staticmethod
    def f(expr, x):
        return eval(expr, {"__builtins__": {}}, {"x": x})

    check_help = "n expr"

    def check(self, args):
        n = _int(args[0]) if args else None
        if n not in self.funcs:
            return "give a function number and a formula."
        expr = _expr(" ".join(args[1:]))
        try:
            res = [(x, self.f(expr, x), want) for x, want in self.tests[n]]
        except Exception:
            return "could not evaluate that formula."
        bad = [f"f{n}({x}) would be {got}, test expects {want}" for x, got, want in res if got != want]
        return f"passes f{n}'s visible tests." if not bad else "fails: " + "; ".join(bad)

    def budget(self):
        return 14

    def goal(self):
        return ("A small repo has functions f1, f2, f3, each with visible tests. Some functions are buggy. Fix the "
                "bugs so the functions are correct (fixes are checked on hidden inputs too), then say DONE.")

    def failing(self, hidden=False):
        out = []
        for n in self.funcs:
            cases = [(x, self.f(self.good[n], x)) for x in range(-5, 12)] if hidden else self.tests[n]
            for x, want in cases:
                try:
                    got = self.f(self.funcs[n], x)
                except Exception:
                    got = "error"
                if got != want:
                    out.append((n, x, want, got))
        return out

    def solved(self):
        return not self.failing(hidden=True)

    def apply(self, verb, args):
        if verb == "DONE":
            self.finished = True
            return "Submitted."
        if verb == "RUN":
            fails = self.failing()
            total = sum(len(t) for t in self.tests.values())
            if not fails:
                return f"{total}/{total} visible tests pass."
            lines = [f"FAIL f{n}({x}): expected {want}, got {got}" for n, x, want, got in fails]
            return f"{total - len(fails)}/{total} visible tests pass.\n" + "\n".join(lines)
        if verb in ("SHOW", "PATCH") and args:
            n = _int(args[0]) if args else None
            if n not in self.funcs:
                return None
            if verb == "SHOW":
                return f"def f{n}(x):\n    return {self.funcs[n]}"
            rest = self.line.split(None, 2)
            expr = _expr(rest[2]) if len(rest) == 3 else ""
            try:
                self.f(expr, 1)
            except Exception:
                return None
            self.funcs[n] = expr
            return f"Patched f{n}."
        return None


# ---------------------------------------------------------------- near-transfer problems
# Same skill as a training game, new surface (names, verbs, feedback words). Never practised. They separate
# "learned the game's wording" from "learned the skill": transfer should show here first if it shows anywhere.

LETTERS = "ABCDE"


class Lock(CodeBreak):
    """CodeBreak's deduction skill with letters, a different verb and different feedback words."""
    name = "lock"
    actions_help = "TRY L L L with letters A-E, e.g. TRY A B C"
    check_help = "L L L"

    def _s(self, code):
        return " ".join(LETTERS[d - 1] for d in code)

    def reveal(self):
        return f"The lock combination was {self._s(self.secret)}."

    def parse(self, args):
        letters = [c for c in "".join(args).upper() if c.isalpha()]
        if len(letters) != self.k or any(c not in LETTERS for c in letters):
            return None
        return [LETTERS.index(c) + 1 for c in letters]

    def check(self, args):
        g = self.parse(args)
        if g is None:
            return f"give {self.k} letters A-E."
        for pg, fb in self.past:
            got = self.score(g, pg)
            if got != fb:
                return (f"inconsistent: against try {self._s(pg)} it would show IN PLACE={got[0]} "
                        f"ELSEWHERE={got[1]}, observed IN PLACE={fb[0]} ELSEWHERE={fb[1]}.")
        return "consistent with all feedback so far."

    def evidence_check(self, verb, args):
        g = self.parse(args) if verb == "TRY" else None
        for pg, fb in self.past if g else []:
            got = self.score(g, pg)
            if got != fb:
                return (f"this combination cannot be right: if it were, try {self._s(pg)} would have shown "
                        f"IN PLACE={got[0]} ELSEWHERE={got[1]}, not IN PLACE={fb[0]} ELSEWHERE={fb[1]}."), False
        return None, False

    def goal(self):
        return (f"Open a lock whose combination is {self.k} letters, each A-E (repeats allowed). After each try you "
                "see IN PLACE = letters right and in the right position, and ELSEWHERE = right letter, wrong position.")

    def apply(self, verb, args):
        if verb != "TRY":
            return None
        guess = self.parse(args)
        if guess is None:
            return None
        exact, mis = self.score(self.secret, guess)
        self.past.append((guess, (exact, mis)))
        if exact == self.k:
            self.got, self.finished = True, True
            return "All 3 in place. The lock opens!"
        return f"{self._s(guess)} -> IN PLACE={exact} ELSEWHERE={mis}"


def _n_to_x(text):
    return re.sub(r"(?<![A-Za-z])n(?![A-Za-z])", "x", text)


class Machine(HiddenRule):
    """HiddenRule's induction skill with a machine, variable n, and different verbs."""
    name = "machine"
    actions_help = "INPUT n (the machine prints g(n), n an integer), SUBMIT expr in n, e.g. SUBMIT 3 * n - 4"

    def reveal(self):
        return f"The machine computed g(n) = {_n_to_x(self.rule).replace('x', 'n')}."

    def goal(self):
        return ("A machine computes g(n), a polynomial in n with integer coefficients, degree at most 2. Feed it "
                "inputs, then SUBMIT its formula. The formula is tested on many hidden inputs; one SUBMIT only.")

    def _as_parent(self, verb):
        self.line = _n_to_x(self.line)
        return {"INPUT": "QUERY", "SUBMIT": "ANSWER"}.get(verb, verb)

    def check(self, args):
        out = super().check([_n_to_x(a) for a in args])
        return out.replace("f(", "g(").replace("formula in x", "formula in n")

    def evidence_check(self, verb, args):
        fact, refuse = super().evidence_check(self._as_parent(verb), args)
        return (fact.replace("f(", "g(") if fact else fact), refuse

    def apply(self, verb, args):
        out = super().apply(self._as_parent(verb), args)
        if out is None:
            return None
        return "Formula recorded." if self.finished else out.replace("f(", "g(")


TRAIN_GAMES = [GuessNumber, CodeBreak, HiddenRule, OddCoin]
HELD_OUT = [Circuit, BugFix]
NEAR = [Lock, Machine]
ALL = {cls.name: cls for cls in TRAIN_GAMES + HELD_OUT + NEAR}


def all_codes(k, digits):
    return [list(c) for c in itertools.product(range(1, digits + 1), repeat=k)]
