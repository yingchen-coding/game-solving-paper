#!/bin/bash
# Root-cause probes on the four failing tasks (base 7B, in the moment, eval seeds 10000-10009).
# selfcheck = told to check (behaviour); ledger = evidence listed every turn (memory); check_tool = a free
# verifier it can call (computation). Whichever fixes it names the missing piece.
cd "$(dirname "$0")"
P=.venv/bin/python; Q=mlx-community/Qwen2.5-7B-Instruct-4bit; E=codebreak,hiddenrule,bugfix,circuit
$P evaluate.py --policy mlx:$Q --envs $E --n 10 --system selfcheck --out runs/rc_selfcheck > /dev/null 2>&1
$P evaluate.py --policy mlx:$Q --envs $E --n 10 --ledger --out runs/rc_ledger > /dev/null 2>&1
$P evaluate.py --policy mlx:$Q --envs $E --n 10 --check-tool --out runs/rc_checktool > /dev/null 2>&1
for d in rc_selfcheck rc_ledger rc_checktool; do $P diagnose.py runs/$d > /dev/null; done
echo rootcause done
