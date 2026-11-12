"""Keep the last LoRA checkpoint saved before training loss went NaN.

mlx_lm LoRA on the 4-bit 7B diverged to NaN at iter ~105 (q7_ei_r1, lr 5e-5) and the final adapter emitted only
'!' tokens. mlx_lm has no gradient clipping, so we save often and roll back to the last finite checkpoint.
Usage: pick_adapter.py ADAPTER_DIR LOGFILE  (exit 1 if no finite checkpoint exists)
"""
import re
import shutil
import sys
from pathlib import Path

adir, log = Path(sys.argv[1]), Path(sys.argv[2])
text = re.sub(r"\x1b\[[0-9;]*m", "", log.read_text())
first_nan = None
for m in re.finditer(r"^\s*(\d+)\s+(?:val\s+)?(nan|[\d.]+)", text, re.M):
    if m.group(2) == "nan":
        first_nan = int(m.group(1))
        break
saves = sorted(int(p.name[:7]) for p in adir.glob("*_adapters.safetensors"))
ok = [s for s in saves if first_nan is None or s < first_nan]
if first_nan is None:
    print(f"{adir}: no NaN, keeping final adapter")
    sys.exit(0)
if not ok:
    sys.exit(f"{adir}: NaN at iter {first_nan} before any checkpoint")
best = adir / f"{ok[-1]:07d}_adapters.safetensors"
final = adir / "adapters.safetensors"
final.rename(adir / "adapters_nan.safetensors")
shutil.copy(best, final)
print(f"{adir}: NaN at iter {first_nan}; using checkpoint {ok[-1]}")
