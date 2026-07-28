#!/bin/bash
# Leader: expanded genes + 2× radii BR full train
#SBATCH --partition=preempt
#SBATCH --job-name=stlr-r2x-L
#SBATCH --mem=200G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/ix1/ylee/kor11/tools/spacetravlr_microbiome/runs/logs/r2x_leader_%j.out
#SBATCH --error=/ix1/ylee/kor11/tools/spacetravlr_microbiome/runs/logs/r2x_leader_%j.err

set -euo pipefail
export HDF5_USE_FILE_LOCKING=FALSE
export SPACETRAVLR_FORCE_KEEP_GENES="Lyz1,Lyz2,Muc2,Muc3,Muc4,Defa17,Defa21,Defa22,Defa24,Defa26,Reg3g,Reg3b,Reg3a,Ang4,Nos2,Duox2,Duoxa2,Cd74,Guca2a,Guca2b,Tff3,Pigr,S100a8,S100a9,S100g,S100a14,Krt18,Krt19,Krt20,Ace2,Lct,Apoa1,Apoa4,Fabp1,Fabp2,Spink1,Clca4b,Itln1,Pla2g2a,Pla2g2e,Cxcl1,Cxcl2,Cxcl5,Ccl25,Ccl20,Nfkb1,Rela,Relb,Stat1,Stat3,Stat2,Hif1a,Il18,Il1b,Tnf,Cxcl9,Cxcl10,Cd14,Tlr2,Tlr4,Tlr5,Tlr9,Nod1,Nod2,Ffar2,Ffar3,Fpr1,Nlrp3,Casp1,Jchain,Igha,Ighm,Alpi,Sis,Chga,Chgb,Lgr5,Ascl2,Olfm4,Axin2,Myc,Jun,Fos,Mmp7,Mmp9,Retnlb,Retnla,Defb1,Camp,Chil3,Chil4,Saa1,Saa3,Socs3,Ido1"

BIN=/ix1/ylee/kor11/tools/SpaceTravLR_rust/target/release/spacetravlr
CFG=/ix1/ylee/kor11/tools/spacetravlr_microbiome/configs/tumor_br_r2x.toml
OUT=/ix1/ylee/kor11/tools/spacetravlr_microbiome/runs/tumor_br_r2x
mkdir -p "$OUT" /ix1/ylee/kor11/tools/spacetravlr_microbiome/runs/logs

echo "[leader-r2x] host=$(hostname) start=$(date -Is)"
echo "[leader-r2x] radii: spatial/cnn=400 contact=60 BR csv=2× dmax_factor=3"
"$BIN" --config "$CFG" --plain --parallel 2 --output-dir "$OUT" --clean-output-dir --skip-auto-adata-prep
echo "[leader-r2x] done=$(date -Is) feathers=$(find "$OUT" -name '*_betadata.feather' 2>/dev/null | wc -l)"
