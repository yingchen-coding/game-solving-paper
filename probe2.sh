#!/bin/bash
cd "$(dirname "$0")"
P=.venv/bin/python; Q=mlx-community/Qwen2.5-7B-Instruct-4bit; E=codebreak,hiddenrule,bugfix,circuit
PIDS=$(pgrep -f "evaluate.py --policy mlx.*q7_" ; pgrep -f "reflect.py|collect.py|mlx_lm lora")
[ -n "$PIDS" ] && kill -STOP $PIDS
$P evaluate.py --policy mlx:$Q --envs $E --n 10 --system paraphrase --out runs/rc_paraphrase > /dev/null 2>&1
$P evaluate.py --policy mlx:$Q --envs $E --n 10 --teacher coach --out runs/rc_coach > /dev/null 2>&1
for d in rc_paraphrase rc_coach; do $P diagnose.py runs/$d > /dev/null; done
[ -n "$PIDS" ] && kill -CONT $PIDS
echo probe2 done
