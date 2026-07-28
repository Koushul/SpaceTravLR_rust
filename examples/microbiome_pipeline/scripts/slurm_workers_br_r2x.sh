#!/bin/bash
# Worker array: join tumor_br_r2x (95 genes, 2× radii)
#SBATCH --partition=preempt
#SBATCH --job-name=stlr-r2x-W
#SBATCH --mem=200G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=1-12
#SBATCH --requeue
#SBATCH --output=/ix1/ylee/kor11/tools/spacetravlr_microbiome/runs/logs/r2x_worker_%A_%a.out
#SBATCH --error=/ix1/ylee/kor11/tools/spacetravlr_microbiome/runs/logs/r2x_worker_%A_%a.err

set -euo pipefail
export HDF5_USE_FILE_LOCKING=FALSE
export SPACETRAVLR_FORCE_KEEP_GENES="Lyz1,Lyz2,Muc2,Muc3,Muc4,Defa17,Defa21,Defa22,Defa24,Defa26,Reg3g,Reg3b,Reg3a,Ang4,Nos2,Duox2,Duoxa2,Cd74,Guca2a,Guca2b,Tff3,Pigr,S100a8,S100a9,S100g,S100a14,Krt18,Krt19,Krt20,Ace2,Lct,Apoa1,Apoa4,Fabp1,Fabp2,Spink1,Clca4b,Itln1,Pla2g2a,Pla2g2e,Cxcl1,Cxcl2,Cxcl5,Ccl25,Ccl20,Nfkb1,Rela,Relb,Stat1,Stat3,Stat2,Hif1a,Il18,Il1b,Tnf,Cxcl9,Cxcl10,Cd14,Tlr2,Tlr4,Tlr5,Tlr9,Nod1,Nod2,Ffar2,Ffar3,Fpr1,Nlrp3,Casp1,Jchain,Igha,Ighm,Alpi,Sis,Chga,Chgb,Lgr5,Ascl2,Olfm4,Axin2,Myc,Jun,Fos,Mmp7,Mmp9,Retnlb,Retnla,Defb1,Camp,Chil3,Chil4,Saa1,Saa3,Socs3,Ido1"

BIN=/ix1/ylee/kor11/tools/SpaceTravLR_rust/target/release/spacetravlr
OUT=/ix1/ylee/kor11/tools/spacetravlr_microbiome/runs/tumor_br_r2x

echo "[worker-r2x $SLURM_ARRAY_TASK_ID] host=$(hostname) start=$(date -Is)"
for i in $(seq 1 180); do
  if [[ -f "$OUT/spacetravlr_run_repro.toml" ]]; then
    if grep -q '\[microbial\]' "$OUT/spacetravlr_run_repro.toml" && grep -A8 '\[microbial\]' "$OUT/spacetravlr_run_repro.toml" | grep -q 'enabled = true'; then
      if grep -A20 '\[spatial\]' "$OUT/spacetravlr_run_repro.toml" | grep -q 'radius = 400'; then
        break
      fi
    fi
  fi
  sleep 20
done
[[ -f "$OUT/spacetravlr_run_repro.toml" ]] || { echo "no repro toml after wait"; exit 1; }
grep -A6 '\[spatial\]' "$OUT/spacetravlr_run_repro.toml" || true
grep -A8 '\[microbial\]' "$OUT/spacetravlr_run_repro.toml" || true
grep -A8 '\[cnn\]' "$OUT/spacetravlr_run_repro.toml" || true
if ! grep -A5 '\[spatial\]' "$OUT/spacetravlr_run_repro.toml" | grep -q 'radius = 400'; then
  echo "FATAL: expected spatial.radius=400 in repro"; exit 1
fi
if ! grep -A8 '\[microbial\]' "$OUT/spacetravlr_run_repro.toml" | grep -q 'enabled = true'; then
  echo "FATAL: microbial.enabled is not true"; exit 1
fi

"$BIN" --join-output-dir "$OUT" --plain --parallel 2 --skip-auto-adata-prep
echo "[worker-r2x $SLURM_ARRAY_TASK_ID] done=$(date -Is) feathers=$(find "$OUT" -name '*_betadata.feather' 2>/dev/null | wc -l)"
