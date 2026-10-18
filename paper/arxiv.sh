#!/bin/sh
# Build paper/main.pdf and the arXiv upload paper/arxiv.zip (main.tex, main.bbl, llncs.cls and the
# two figures the paper uses), then check that the zip compiles on its own.
# Refuses while main.tex still says TODO or a number fails check_numbers.py.
# Run:  sh paper/arxiv.sh        (needs Tectonic: brew install tectonic)
set -e
cd "$(dirname "$0")"
if grep -n TODO main.tex; then echo "main.tex still has the TODOs above" >&2; exit 1; fi
python3 check_numbers.py

build=$(mktemp -d)
tectonic --keep-intermediates --keep-logs --outdir "$build" main.tex
cp "$build/main.pdf" main.pdf

bundle=$(mktemp -d)
cp main.tex "$build/main.bbl" ../figures/qaoa_depth.pdf ../figures/sa_tightness.pdf "$bundle"
tectonic -X bundle cat llncs.cls > "$bundle/llncs.cls"   # in case arXiv's TeX Live lacks it; the
                                                         # .bbl means arXiv needs no .bst
rm -f arxiv.zip
(cd "$bundle" && zip -q -r - .) > arxiv.zip

check=$(mktemp -d)                               # the zip alone must compile
(cd "$check" && unzip -q "$OLDPWD/arxiv.zip" && tectonic main.tex >/dev/null 2>&1) \
  || { echo "arxiv.zip does not compile on its own" >&2; exit 1; }
echo "ok: main.pdf and arxiv.zip ($(unzip -Z1 arxiv.zip | tr '\n' ' '))"
