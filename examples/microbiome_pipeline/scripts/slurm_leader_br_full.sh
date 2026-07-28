#!/bin/bash
# Leader: full spatial TF+LR+BR train (writes spacetravlr_run_repro.toml)
#SBATCH --partition=preempt
#SBATCH --job-name=stlr-br-lead
#SBATCH --mem=200G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/ix1/ylee/kor11/tools/spacetravlr_microbiome/runs/logs/leader_%j.out
#SBATCH --error=/ix1/ylee/kor11/tools/spacetravlr_microbiome/runs/logs/leader_%j.err

set -euo pipefail
export HDF5_USE_FILE_LOCKING=FALSE
# Keep panel genes through HVG subset during auto-prep
export SPACETRAVLR_FORCE_KEEP_GENES="Lyz1,Lyz2,Muc2,Muc3,Defa17,Defa24,Reg3g,Reg3b,Ang4,Nos2,Duox2,Cd74,Guca2a,Guca2b,Tff3,Pigr,S100a8,S100a9,S100g,Krt18,Krt19,Ace2,Lct,Apoa1,Apoa4,Spink1,Clca4b,Itln1,Pla2g2a,Cxcl1,Ccl25,Nfkb1,Rela,Stat1,Stat3,Hif1a,Tlr2,Tlr4,Tlr5,Tlr9,Nod1,Nod2,Ffar2,Ffar3,Cd14,Fpr1"

BIN=/ix1/ylee/kor11/tools/SpaceTravLR_rust/target/release/spacetravlr
CFG=/ix1/ylee/kor11/tools/spacetravlr_microbiome/configs/tumor_br_full.toml
OUT=/ix1/ylee/kor11/tools/spacetravlr_microbiome/runs/tumor_br_full
mkdir -p "$OUT" /ix1/ylee/kor11/tools/spacetravlr_microbiome/runs/logs

echo "[leader] host=$(hostname) start=$(date -Is)"
# Ready imputed h5ad already has cell_type + imputed_count; skip QC/HVG wipe of panel genes
"$BIN" --config "$CFG" --plain --parallel 2 --output-dir "$OUT" --clean-output-dir --skip-auto-adata-prep
echo "[leader] done=$(date -Is)"
