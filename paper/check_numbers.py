"""Every decimal (0.474), percentage (91%) and k/m count (19/20) in main.tex, comments left out,
must appear in results/*.txt.  Integers are not checked: they cite their source in a comment.
Run:  python3 paper/check_numbers.py"""
import pathlib
import re
import sys

root = pathlib.Path(__file__).resolve().parent.parent
NUM = re.compile(r"\d+(?:\.\d+|%|/\d+)")
known = {m for f in (root / "results").glob("*.txt") for m in NUM.findall(f.read_text())}
lines = (root / "paper" / "main.tex").read_text().splitlines()
bad = [(i, m) for i, line in enumerate(lines, 1)
       for m in NUM.findall(re.sub(r"(?<!\\)%.*", "", line).replace("\\%", "%")) if m not in known]
for i, m in bad:
    print(f"main.tex:{i}: {m} is not in results/*.txt")
if bad:
    sys.exit(1)
print("ok: every decimal, percentage and k/m count in main.tex is in results/*.txt")
