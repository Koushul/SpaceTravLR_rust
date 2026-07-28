#!/bin/bash
#SBATCH --partition=preempt
#SBATCH --job-name=stlr-r2x-R
#SBATCH --mem=200G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --array=1-2
#SBATCH --requeue
#SBATCH --output=/ix1/ylee/kor11/tools/spacetravlr_microbiome/runs/logs/r2x_resume_%A_%a.out
#SBATCH --error=/ix1/ylee/kor11/tools/spacetravlr_microbiome/runs/logs/r2x_resume_%A_%a.err

set -euo pipefail
export HDF5_USE_FILE_LOCKING=FALSE
BIN=/ix1/ylee/kor11/tools/SpaceTravLR_rust/target/release/spacetravlr
OUT=/ix1/ylee/kor11/tools/spacetravlr_microbiome/runs/tumor_br_r2x
echo "[resume-r2x $SLURM_ARRAY_TASK_ID] start=$(date -Is) host=$(hostname)"
[[ -f "$OUT/celloracle_tf_priors.feather" ]] || { echo "missing priors"; exit 1; }
"$BIN" --join-output-dir "$OUT" --plain --parallel 2 --skip-auto-adata-prep
echo "[resume-r2x $SLURM_ARRAY_TASK_ID] done=$(date -Is) feathers=$(find "$OUT" -name '*_betadata.feather' | wc -l)"
