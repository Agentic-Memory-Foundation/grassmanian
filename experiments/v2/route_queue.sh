#!/bin/bash
# Encode-aware routing queue. Skips pairs whose per-query arrays already exist.
set -u
cd /workspace/experiments/v2
OUT=/workspace/experiments/v2/results_repro
mkdir -p "$OUT"
DATASETS="scifact nfcorpus arguana scidocs fiqa trec-covid"
EMBEDS="minilm bge-small"

# Do not start a second copy of a run that is already writing this directory.
while pgrep -f "run_routing.py" >/dev/null; do
  echo "waiting for the in-flight routing process"
  sleep 20
done

for emb in $EMBEDS; do
  for ds in $DATASETS; do
    while [[ ! -f emb/${ds}__${emb}.npz ]]; do
      echo "waiting for emb/${ds}__${emb}.npz"
      sleep 20
    done
    if [[ -f $OUT/routing__${ds}__${emb}.json && -f $OUT/routing__${ds}__${emb}.npz ]]; then
      echo "have $ds $emb"
      continue
    fi
    echo "START $ds $emb $(date -u +%H:%M:%S)"
    OMP_NUM_THREADS=4 ROUTING_OUT="$OUT" python3 -u run_routing.py "$ds" "$emb" || exit 1
    echo "DONE $ds $emb $(date -u +%H:%M:%S)"
  done
done

python3 -u query_level.py "$OUT" || exit 1
echo QUERY_LEVEL_OK
