#!/bin/bash
# Collect/rank BR after orphan-gene resume (33 feathers; 3 true zero-beta orphans remain).
#SBATCH --partition=preempt
#SBATCH --job-name=stlr-br-eval33
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=/ix1/ylee/kor11/tools/spacetravlr_microbiome/runs/logs/eval33_%j.out
#SBATCH --error=/ix1/ylee/kor11/tools/spacetravlr_microbiome/runs/logs/eval33_%j.err

set -euo pipefail
OUT=/ix1/ylee/kor11/tools/spacetravlr_microbiome/runs/tumor_br_full
n=$(find "$OUT" -name '*_betadata.feather' 2>/dev/null | wc -l)
echo "[eval33] betadata=$n start=$(date -Is)"
for g in Cxcl1 Muc3 Nos2 S100a8 S100a9 Stat1; do
  if [[ -f "$OUT/${g}_betadata.feather" ]]; then
    echo "  recovered $g"
  elif [[ -f "$OUT/${g}.orphan" ]]; then
    echo "  orphan $g"
  else
    echo "  missing $g"
  fi
done
[[ "$n" -ge 33 ]] || { echo "expected >=33 feathers"; exit 1; }
bash /ix1/ylee/kor11/tools/spacetravlr_microbiome/scripts/07_collect_br_eval.sh
echo "[eval33] done=$(date -Is)"
