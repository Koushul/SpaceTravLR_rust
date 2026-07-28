#!/bin/bash
# Poll until 36 gene betadata feathers exist, then re-collect/rank BR terms.
#SBATCH --partition=preempt
#SBATCH --job-name=stlr-br-eval36
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=14:00:00
#SBATCH --requeue
#SBATCH --output=/ix1/ylee/kor11/tools/spacetravlr_microbiome/runs/logs/eval36_%j.out
#SBATCH --error=/ix1/ylee/kor11/tools/spacetravlr_microbiome/runs/logs/eval36_%j.err

set -euo pipefail
OUT=/ix1/ylee/kor11/tools/spacetravlr_microbiome/runs/tumor_br_full
EXPECT_GENES=36

echo "[eval-wait-36] start=$(date -Is)"
for i in $(seq 1 420); do
  n=0
  if [[ -d "$OUT" ]]; then
    n=$(find "$OUT" -name '*_betadata.feather' 2>/dev/null | wc -l)
  fi
  missing=""
  for g in Cxcl1 Muc3 Nos2 S100a8 S100a9 Stat1; do
    if [[ ! -f "$OUT/${g}_betadata.feather" ]]; then
      missing="$missing $g"
    fi
  done
  echo "[eval-wait-36] iter=$i betadata=$n / $EXPECT_GENES missing:$missing $(date -Is)"
  if [[ "$n" -ge "$EXPECT_GENES" ]]; then
    sleep 120
    n2=$(find "$OUT" -name '*_betadata.feather' 2>/dev/null | wc -l)
    echo "[eval-wait-36] settled betadata=$n2"
    if [[ "$n2" -ge "$EXPECT_GENES" ]]; then
      break
    fi
  fi
  sleep 90
done

n=$(find "$OUT" -name '*_betadata.feather' 2>/dev/null | wc -l || echo 0)
echo "[eval-wait-36] final betadata=$n"
[[ "$n" -ge "$EXPECT_GENES" ]] || { echo "expected $EXPECT_GENES feathers, got $n"; exit 1; }

bash /ix1/ylee/kor11/tools/spacetravlr_microbiome/scripts/07_collect_br_eval.sh
echo "[eval-wait-36] done=$(date -Is)"
