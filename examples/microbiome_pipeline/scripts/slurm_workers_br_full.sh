#!/bin/bash
# Worker array: join same output dir on preempt GPUs
#SBATCH --partition=preempt
#SBATCH --job-name=stlr-br-w
#SBATCH --mem=200G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=1-8
#SBATCH --requeue
#SBATCH --output=/ix1/ylee/kor11/tools/spacetravlr_microbiome/runs/logs/worker_%A_%a.out
#SBATCH --error=/ix1/ylee/kor11/tools/spacetravlr_microbiome/runs/logs/worker_%A_%a.err

set -euo pipefail
export HDF5_USE_FILE_LOCKING=FALSE
export SPACETRAVLR_FORCE_KEEP_GENES="Lyz1,Lyz2,Muc2,Muc3,Defa17,Defa24,Reg3g,Reg3b,Ang4,Nos2,Duox2,Cd74,Guca2a,Guca2b,Tff3,Pigr,S100a8,S100a9,S100g,Krt18,Krt19,Ace2,Lct,Apoa1,Apoa4,Spink1,Clca4b,Itln1,Pla2g2a,Cxcl1,Ccl25,Nfkb1,Rela,Stat1,Stat3,Hif1a,Tlr2,Tlr4,Tlr5,Tlr9,Nod1,Nod2,Ffar2,Ffar3,Cd14,Fpr1"

BIN=/ix1/ylee/kor11/tools/SpaceTravLR_rust/target/release/spacetravlr
OUT=/ix1/ylee/kor11/tools/spacetravlr_microbiome/runs/tumor_br_full

echo "[worker $SLURM_ARRAY_TASK_ID] host=$(hostname) start=$(date -Is)"
for i in $(seq 1 180); do
  if [[ -f "$OUT/spacetravlr_run_repro.toml" ]]; then
    # wait until microbial.enabled is true (guards against stale/broken repro)
    if grep -q 'enabled = true' "$OUT/spacetravlr_run_repro.toml" && grep -q '\[microbial\]' "$OUT/spacetravlr_run_repro.toml"; then
      break
    fi
  fi
  sleep 20
done
[[ -f "$OUT/spacetravlr_run_repro.toml" ]] || { echo "no repro toml after wait"; exit 1; }
grep -A2 '\[microbial\]' "$OUT/spacetravlr_run_repro.toml" || true
if ! grep -A5 '\[microbial\]' "$OUT/spacetravlr_run_repro.toml" | grep -q 'enabled = true'; then
  echo "FATAL: microbial.enabled is not true in repro"; exit 1
fi

# --skip-auto-adata-prep: belt-and-suspenders; join mode also skips prep in the binary
"$BIN" --join-output-dir "$OUT" --plain --parallel 2 --skip-auto-adata-prep
echo "[worker $SLURM_ARRAY_TASK_ID] done=$(date -Is)"
