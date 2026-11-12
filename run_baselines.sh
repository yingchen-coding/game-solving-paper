#!/bin/bash
cd "$(dirname "$0")"
P=.venv/bin/python
Q3=mlx-community/Qwen2.5-3B-Instruct-4bit
$P evaluate.py --policy mlx:$Q3 --out runs/q3_plain --n 30 > runs/q3_plain.log 2>&1
$P evaluate.py --policy mlx:$Q3 --system strategy --out runs/q3_strategy --n 30 > runs/q3_strategy.log 2>&1
echo baselines done
