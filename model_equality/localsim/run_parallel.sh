#!/bin/bash
# Run the simulator on several base models at once, one process per model, spread over the 2 GPUs of the login
# node, then merge everything into runs/<RUN>/ (answers.jsonl, firsttok.jsonl, endpoints.csv).
#
#   bash model_equality/localsim/run_parallel.sh PROMPTS.jsonl RUN_NAME qwen05,llama1b,smol360 [extra localsim args]
#
# Endpoint names: <letter>-01..09 per model (a-.., b-.., c-..), anonymous inside each family.
# float32 for every model: small models only, and SmolLM2 / Gemma-like models overflow in float16.

#
# Settings (environment variables, all optional):
#   LOCALSIM_DIR  folder holding src/, model_equality/ and runs/   (default: biechy1's copy in the team folder)
#   LOCALSIM_PY   python with torch + CUDA                         (default: the unlearning .venv of the team)
#   NGPU          number of GPUs to spread the models over         (default 2 = login node; 4 on a compute node)

PROMPTS=$1; RUN=$2; BASES=$3; shift 3
T=/p/scratch/training2666/pierron1/tempete
D=${LOCALSIM_DIR:-$T/biechy1_localsim}
PY=${LOCALSIM_PY:-$T/unlearning/.venv/bin/python}
NGPU=${NGPU:-2}
cd "$D" || exit 1
source ~/.bashrc >/dev/null 2>&1
set -u
mkdir -p "runs/$RUN"
LETTERS=(a b c d e f g h i j k l m n o p)
i=0
pids=()
for b in ${BASES//,/ }; do
  gpu=$((i % NGPU))
  tag="${LETTERS[$i]}"
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$D/pylib "$PY" model_equality/localsim/localsim.py \
      --prompts "$PROMPTS" --bases "$b" --tag "$tag" --dtype float32 --out "runs/$RUN/$b" "$@" \
      > "runs/$RUN/$b.log" 2>&1 &
  pids+=($!)
  echo "started $b on GPU $gpu (tag $tag, pid $!)"
  i=$((i + 1))
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
# merge
: > "runs/$RUN/answers.jsonl"; : > "runs/$RUN/firsttok.jsonl"
echo "endpoint,family,model,kind" > "runs/$RUN/endpoints.csv"
for b in ${BASES//,/ }; do
  cat "runs/$RUN/$b/answers.jsonl" >> "runs/$RUN/answers.jsonl" 2>/dev/null
  cat "runs/$RUN/$b/firsttok.jsonl" >> "runs/$RUN/firsttok.jsonl" 2>/dev/null
  tail -n +2 "runs/$RUN/$b/endpoints.csv" >> "runs/$RUN/endpoints.csv" 2>/dev/null
done
# a resumed run recomputes first-token probabilities of finished variants: keep one line per (endpoint, id)
"$PY" -c "
import json, sys
p = sys.argv[1]; seen = {}
for line in open(p):
    r = json.loads(line); seen[(r['endpoint'], r['id'])] = line
open(p, 'w').writelines(seen.values())
" "runs/$RUN/firsttok.jsonl"
wc -l "runs/$RUN/answers.jsonl" "runs/$RUN/firsttok.jsonl"
[ $fail -eq 0 ] && echo "ALL DONE runs/$RUN" || echo "SOME RUN FAILED: see runs/$RUN/*.log"
