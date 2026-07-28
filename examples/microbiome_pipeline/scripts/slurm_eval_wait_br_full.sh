#!/bin/bash
# Poll until >=30 gene betadata feathers exist, then collect/rank BR terms.
#SBATCH --partition=preempt
#SBATCH --job-name=stlr-br-eval
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=14:00:00
#SBATCH --requeue
#SBATCH --output=/ix1/ylee/kor11/tools/spacetravlr_microbiome/runs/logs/eval_%j.out
#SBATCH --error=/ix1/ylee/kor11/tools/spacetravlr_microbiome/runs/logs/eval_%j.err

set -euo pipefail
OUT=/ix1/ylee/kor11/tools/spacetravlr_microbiome/runs/tumor_br_full
EXPECT_GENES=30

echo "[eval-wait] start=$(date -Is)"
for i in $(seq 1 420); do
  n=0
  if [[ -d "$OUT" ]]; then
    n=$(find "$OUT" -name '*_betadata.feather' 2>/dev/null | wc -l)
  fi
  echo "[eval-wait] iter=$i betadata=$n / $EXPECT_GENES $(date -Is)"
  if [[ "$n" -ge "$EXPECT_GENES" ]]; then
    sleep 180
    n2=$(find "$OUT" -name '*_betadata.feather' 2>/dev/null | wc -l)
    echo "[eval-wait] settled betadata=$n2"
    if [[ "$n2" -ge "$EXPECT_GENES" ]]; then
      break
    fi
  fi
  sleep 120
done

n=$(find "$OUT" -name '*_betadata.feather' 2>/dev/null | wc -l || echo 0)
echo "[eval-wait] final betadata=$n"
[[ "$n" -ge 1 ]] || { echo "no betadata produced"; exit 1; }

bash /ix1/ylee/kor11/tools/spacetravlr_microbiome/scripts/07_collect_br_eval.sh
echo "[eval] done=$(date -Is)"
