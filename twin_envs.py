"""A held-out twin of OddCoin, built the way Machine twins HiddenRule: same seed, same puzzle, new words.

Parcels subclasses OddCoin, so seed s gives the same number of items, the same heavy one and the same budget; only
the surface changes (coins -> parcels, WEIGH -> SCALE with '/' between the pans, ANSWER -> CLAIM). It lives outside
frozen/ so the frozen evaluator and every earlier condition stay untouched; importing this module registers it.
"""
import os
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


envs.ALL[Parcels.name] = Parcels
