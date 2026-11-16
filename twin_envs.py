"""A held-out twin of OddCoin, built the way Machine twins HiddenRule: same seed, same puzzle, new words.

Parcels subclasses OddCoin, so seed s gives the same number of items, the same heavy one and the same budget; only
the surface changes (coins -> parcels, WEIGH -> SCALE with '/' between the pans, ANSWER -> CLAIM). It lives outside
frozen/ so the frozen evaluator and every earlier condition stay untouched; importing this module registers it.
"""
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "frozen"))
import envs  # noqa: E402


class Parcels(envs.OddCoin):
    name = "parcels"
    actions_help = ("SCALE left / right with parcel labels, e.g. SCALE 1 2 3 / 4 5 6; "
                    "CLAIM n names the heavy parcel (one CLAIM only)")

    def reveal(self):
        return f"The heavy parcel was {self.heavy}."

    def goal(self):
        return (f"A courier hands you {self.n} parcels labelled 1-{self.n}. One holds extra weight; the others are "
                "identical. Put parcels on the two pans of the scale to find it, then CLAIM it.")

    def _as_parent(self, verb):
        self.line = self.line.replace("/", "|")
        return {"SCALE": "WEIGH", "CLAIM": "ANSWER"}.get(verb, verb)

    def evidence_check(self, verb, args):
        fact, refuse = super().evidence_check(self._as_parent(verb), args)
        return (fact.replace("coin", "parcel").replace("weighing", "scale reading") if fact else fact), refuse

    def apply(self, verb, args):
        out = super().apply(self._as_parent(verb), args)
        if out is None:
            return None
        if self.finished:
            return "Claim recorded."
        return (out.replace(" vs ", " / ").replace("LEFT is heavier", "LEFT pan sinks")
                .replace("RIGHT is heavier", "RIGHT pan sinks").replace("BALANCED", "LEVEL"))


# Practice versions whose wording is redrawn per seed (domain randomisation for RL). Pools exclude both held-out
# twins' wording (Machine: machine, g, n, INPUT, SUBMIT; Parcels: parcel, SCALE, '/', CLAIM, pan sinks, LEVEL) and
# the reserved ASK verb. The puzzle still comes from self.rng, so seed s is the same puzzle as the original game.
FINALS = ["REPORT", "DECLARE", "GIVE", "STATE", "COMMIT"]


def _wording(seed: int, pools: dict) -> dict:
    rng = random.Random(seed * 7919 + 17)
    return {k: rng.choice(v) for k, v in pools.items()}


class RandHiddenRule(envs.HiddenRule):
    name = "randhiddenrule"
    POOLS = {"noun": ["secret rule", "black box", "oracle", "device", "sealed formula"], "f": list("hprqF"),
             "v": list("tkmzw"), "probe": ["PROBE", "TEST", "SAMPLE", "MEASURE", "EVAL"], "final": FINALS}

    def __init__(self, seed: int):
        self.w = _wording(seed, self.POOLS)
        w = self.w
        self.actions_help = (f"{w['probe']} {w['v']} (see {w['f']}({w['v']}), {w['v']} an integer), "
                             f"{w['final']} expr in {w['v']}, e.g. {w['final']} 2 * {w['v']} + 1")
        super().__init__(seed)

    def goal(self):
        w = self.w
        article = "An" if w["noun"][0] in "aeiou" else "A"
        return (f"{article} {w['noun']} {w['f']}({w['v']}) is a polynomial in {w['v']} with integer coefficients, degree at "
                f"most 2. Query it, then {w['final']} its formula. The answer is checked on many hidden inputs; "
                f"one {w['final']} only.")

    def _as_parent(self, verb):
        self.line = re.sub(rf"(?<![A-Za-z]){self.w['v']}(?![A-Za-z])", "x", self.line)
        return {self.w["probe"]: "QUERY", self.w["final"]: "ANSWER"}.get(verb, verb)

    def evidence_check(self, verb, args):
        fact, refuse = super().evidence_check(self._as_parent(verb), args)
        return (fact.replace("f(", f"{self.w['f']}(") if fact else fact), refuse

    def apply(self, verb, args):
        out = super().apply(self._as_parent(verb), args)
        return out.replace("f(", f"{self.w['f']}(") if out else out


class RandOddCoin(envs.OddCoin):
    name = "randoddcoin"
    TILTS = [("LEFT is heavier", "RIGHT is heavier", "BALANCED"), ("LEFT side drops", "RIGHT side drops", "EVEN"),
             ("LEFT goes down", "RIGHT goes down", "STEADY"), ("LEFT outweighs", "RIGHT outweighs", "EQUAL")]
    POOLS = {"item": ["token", "weight", "ball", "block", "marble"], "weigh": ["BALANCE", "PLACE", "HOLD", "MATCH", "SET"],
             "sep": ["|", "vs", "against"], "tilt": list(range(4)), "final": FINALS}

    def __init__(self, seed: int):
        self.w = _wording(seed, self.POOLS)
        w = self.w
        self.actions_help = (f"{w['weigh']} left {w['sep']} right with {w['item']} numbers, e.g. {w['weigh']} 1 2 3 "
                             f"{w['sep']} 4 5 6; {w['final']} n names the heavy {w['item']} (one {w['final']} only)")
        super().__init__(seed)

    def goal(self):
        w = self.w
        return (f"There are {self.n} {w['item']}s numbered 1-{self.n}. Exactly one is heavier; the rest weigh the "
                f"same. Use the balance to find the heavy {w['item']}, then {w['final']} it.")

    def _as_parent(self, verb):
        if self.w["sep"] != "|":
            self.line = re.sub(rf"\b{self.w['sep']}\b", "|", self.line)
        return {self.w["weigh"]: "WEIGH", self.w["final"]: "ANSWER"}.get(verb, verb)

    def evidence_check(self, verb, args):
        fact, refuse = super().evidence_check(self._as_parent(verb), args)
        return (fact.replace("coin", self.w["item"]) if fact else fact), refuse

    def apply(self, verb, args):
        out = super().apply(self._as_parent(verb), args)
        if out is None or self.finished:
            return out
        for old, new in zip(self.TILTS[0], self.TILTS[self.w["tilt"]]):
            out = out.replace(old, new)
        return out.replace(" vs ", f" {self.w['sep']} ")


for _cls in (Parcels, RandHiddenRule, RandOddCoin):
    envs.ALL[_cls.name] = _cls
