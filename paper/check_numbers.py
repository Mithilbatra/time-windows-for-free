"""Every decimal (0.474), percentage (91%) and k/m count (19/20) in the paper's text (main.tex's body,
shared.tex and body.tex), comments left out, must appear in results/*.txt.  Integers and the preamble's layout lengths are not checked.
Run:  python3 paper/check_numbers.py"""
import pathlib
import re
import sys

root = pathlib.Path(__file__).resolve().parent.parent
NUM = re.compile(r"\d+(?:\.\d+|%|/\d+)")
known = {m for f in (root / "results").glob("*.txt") for m in NUM.findall(f.read_text())}
bad = []
for name in ("main.tex", "shared.tex", "body.tex"):
    lines = (root / "paper" / name).read_text().splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith(r"\begin{document}")), 0)
    bad += [(name, i, m) for i, line in enumerate(lines[start:], start + 1)
            for m in NUM.findall(re.sub(r"(?<!\\)%.*", "", line).replace("\\%", "%")) if m not in known]
for name, i, m in bad:
    print(f"{name}:{i}: {m} is not in results/*.txt")
if bad:
    sys.exit(1)
print("ok: every decimal, percentage and k/m count in the paper is in results/*.txt")
