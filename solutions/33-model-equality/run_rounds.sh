#!/bin/bash
# Collect rounds <first>..<last> one after the other (never two collectors at once).
# Stops at the first round that does not end with "done:" (budget mismatch, cap, error).
# Usage: run_rounds.sh <first> <last> <log dir>
set -uo pipefail
FIRST="$1"; LAST="$2"; LOGS="$3"
WT="$(cd "$(dirname "$0")/../.." && pwd)"   # repository root
ROUNDS_DIR="${ROUNDS_DIR:-solutions/33-model-equality/rounds}"   # written by make_roundC.py
PYTHON="${PYTHON:-.venv/bin/python}"
cd "$WT"
for (( k=FIRST; k<=LAST; k++ )); do
  f="$ROUNDS_DIR/round${k}_prompts.json"
  [ -f "$f" ] || { echo "round $k: no file, stop"; exit 0; }
  [ -f /tmp/xmc/STOP_LUCAS_PLAN ] && { echo "STOP file present, stop before round $k"; exit 0; }
  echo "round $k start $(date +%H:%M:%S)"
  PYTHONPATH=src OMP_NUM_THREADS=2 nice -n 10 "$PYTHON" -u \
    -m hackathon.model_equality.research.x_lucas_run "$f" --answers artifacts/model_equality/answers.jsonl \
    --yes --parallel 4 > "$LOGS/round${k}.log" 2>&1
  grep -q "^done:" "$LOGS/round${k}.log" || { echo "round $k did not finish cleanly"; tail -3 "$LOGS/round${k}.log"; exit 3; }
  echo "round $k done $(date +%H:%M:%S)"
done
