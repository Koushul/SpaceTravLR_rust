#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO="$(pwd)"
BIN_TRAIN="$REPO/target/release/spacetravlr"
BIN_PERTURB="$REPO/target/release/spacetravlr-perturb"
EXP="$REPO/experiments/il21_bcl6_ablation"
RUNS="$EXP/runs"
OVER="$EXP/overlays"
H5AD_SRC="/ix/djishnu/shared/djishnu_kor11/training_data_revision/snrna_human_tonsil.h5ad"
export SPACETRAVLR_FORCE_CPU="${SPACETRAVLR_FORCE_CPU:-1}"

if [[ ! -x "$BIN_TRAIN" ]]; then
  echo "Build release first: cargo build --release --bin spacetravlr --bin spacetravlr-perturb --bin spacetravlr-alignment" >&2
  exit 1
fi

python3 "$EXP/generate_overlays.py"

echo "=== Train tf_only__seed (CellOracle priors) ==="
"$BIN_TRAIN" --plain --config "$OVER/tf_only__seed.toml"

python3 "$EXP/patch_tf_priors.py"

for f in "$OVER"/*.toml; do
  base="$(basename "$f" .toml)"
  if [[ "$base" == "tf_only__seed" ]]; then
    continue
  fi
  echo "=== Train $base ==="
  "$BIN_TRAIN" --plain --config "$f"
done

echo "=== Perturb each run ==="
for d in "$RUNS"/*; do
  [[ -d "$d" ]] || continue
  name="$(basename "$d")"
  repro="$d/spacetravlr_run_repro.toml"
  if [[ ! -f "$repro" ]]; then
    echo "skip $name (no repro)" >&2
    continue
  fi
  out="$d/perturb_feathers"
  mkdir -p "$out"
  for bt in "$REPO/experiments/il21_bcl6_ablation/perturb_jobs"/*.toml; do
    j=$(basename "$bt")
    cp "$bt" "$d/_perturb_$j"
    ( cd "$d" && "$BIN_PERTURB" --run-toml "$repro" --batch-toml "$d/_perturb_$j" && rm -f "$d/_perturb_$j" )
  done
done

echo "=== Analysis h5ad ==="
python3 "$EXP/prepare_analysis_h5ad.py"

MAN="$EXP/analysis/manifest.csv"
: >"$MAN"
echo "label,feather_path" >>"$MAN"
for d in "$RUNS"/*; do
  [[ -d "$d" ]] || continue
  name="$(basename "$d")"
  pf="$d/perturb_feathers"
  [[ -d "$pf" ]] || continue
  for feather in "$pf"/*.feather; do
    [[ -f "$feather" ]] || continue
    job="$(basename "$feather" .feather)"
    echo "${name}__${job},${feather}" >>"$MAN"
  done
done

echo "=== Alignment (Rust; no Python/velocyto) ==="
if [[ -x "$REPO/target/release/spacetravlr-alignment" ]] || [[ -x "$REPO/target/debug/spacetravlr-alignment" ]]; then
  ALIGN_BIN="$REPO/target/release/spacetravlr-alignment"
  [[ -x "$ALIGN_BIN" ]] || ALIGN_BIN="$REPO/target/debug/spacetravlr-alignment"
  "$ALIGN_BIN" \
    --h5ad "$H5AD_SRC" \
    --manifest "$MAN" \
    --out-csv "$EXP/analysis/alignment_rust_per_celltype.csv" \
    --annot cell_type
  echo "Wrote $EXP/analysis/alignment_rust_per_celltype.csv"
else
  echo "Build alignment binary: cargo build --bin spacetravlr-alignment" >&2
fi

echo "=== Summarize L2 delta norms ==="
python3 "$EXP/summarize_deltas.py" --runs-dir "$RUNS" --out-csv "$EXP/analysis/l2_delta_summary.csv"

echo "Done. Key outputs:"
echo "  $MAN"
echo "  $EXP/analysis/l2_delta_summary.csv"
