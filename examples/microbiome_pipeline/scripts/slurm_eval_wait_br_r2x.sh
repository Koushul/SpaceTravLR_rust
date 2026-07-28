#!/bin/bash
# Wait for ~80+ betadata feathers then collect microbiome BR ranking (r2x run).
#SBATCH --partition=preempt
#SBATCH --job-name=stlr-r2x-E
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=14:00:00
#SBATCH --requeue
#SBATCH --output=/ix1/ylee/kor11/tools/spacetravlr_microbiome/runs/logs/r2x_eval_%j.out
#SBATCH --error=/ix1/ylee/kor11/tools/spacetravlr_microbiome/runs/logs/r2x_eval_%j.err

set -euo pipefail
export HDF5_USE_FILE_LOCKING=FALSE
OUT=/ix1/ylee/kor11/tools/spacetravlr_microbiome/runs/tumor_br_r2x
BIN=/ix1/ylee/kor11/tools/SpaceTravLR_rust/target/release/spacetravlr
TOML="$OUT/spacetravlr_run_repro.toml"
PY=/ix1/ylee/kor11/tools/susceptibility_pilot/.venv/bin/python
EXPECT=60

echo "[eval-r2x] start=$(date -Is) expect>=$EXPECT"
for i in $(seq 1 480); do
  n=0
  [[ -d "$OUT" ]] && n=$(find "$OUT" -name '*_betadata.feather' 2>/dev/null | wc -l)
  o=$(find "$OUT" -name '*.orphan' 2>/dev/null | wc -l || echo 0)
  echo "[eval-r2x] iter=$i feathers=$n orphans=$o $(date -Is)"
  if [[ "$n" -ge "$EXPECT" ]]; then
    sleep 180
    n2=$(find "$OUT" -name '*_betadata.feather' 2>/dev/null | wc -l)
    echo "[eval-r2x] settled feathers=$n2"
    [[ "$n2" -ge "$EXPECT" ]] && break
  fi
  sleep 120
done

n=$(find "$OUT" -name '*_betadata.feather' 2>/dev/null | wc -l || echo 0)
echo "[eval-r2x] final feathers=$n"
[[ "$n" -ge 1 ]] || { echo "no betadata"; exit 1; }
[[ -f "$TOML" ]] || { echo "missing $TOML"; exit 1; }

"$BIN" collect-interactions --run-toml "$TOML" --annot cell_type --aggregate mean --mode microbiome \
  --out "$OUT/plucked_feathers_microbiome.feather"
"$BIN" collect-interactions --run-toml "$TOML" --annot cell_type --aggregate mean --mode all \
  --out "$OUT/plucked_feathers_all.feather"

"$PY" - <<'PY'
import json
from pathlib import Path
import pandas as pd
import pyarrow.feather as feather

out = Path("/ix1/ylee/kor11/tools/spacetravlr_microbiome/runs/tumor_br_r2x")
inter = pd.read_csv(
    "/ix1/ylee/kor11/tools/spacetravlr_microbiome/configs/bact_host_interactions.train_scfa_merged_r2x.csv"
)
meta = inter.rename(columns={"signal_id": "signal"}).drop_duplicates(["signal", "receptor"])
df = feather.read_feather(out / "plucked_feathers_microbiome.feather")
rename = {}
for c in df.columns:
    cl = c.lower()
    if cl in ("interaction", "modulator", "pair", "feature", "name") and "interaction" not in rename.values():
        rename[c] = "interaction"
    elif cl in ("target", "target_gene", "gene") and "target" not in rename.values():
        rename[c] = "target"
    elif cl in ("value", "mean", "beta", "coef") and "value" not in rename.values():
        rename[c] = "value"
    elif "interaction_type" in cl or cl == "type":
        rename[c] = "interaction_type"
    elif cl in ("cell_type", "annot", "cluster"):
        rename[c] = "cell_type"
df = df.rename(columns=rename)
if "interaction" not in df.columns:
    raise SystemExit(f"no interaction col: {df.columns.tolist()}")
if "value" not in df.columns:
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    df["value"] = df[num[0]]
if "target" not in df.columns:
    df["target"] = ""
df["abs"] = df["value"].abs()
clean = df["interaction"].astype(str).str.replace(r"^beta_", "", regex=True)
parts = clean.str.split("$", n=1, expand=True)
df["signal"] = parts[0]
df["receptor"] = parts[1] if parts.shape[1] > 1 else ""
df = df.merge(
    meta[["signal", "receptor", "pathway", "signaling_class", "signal_name", "default_radius_um"]],
    on=["signal", "receptor"], how="left",
)
rank = (
    df.groupby(
        ["interaction", "signal", "receptor", "pathway", "signaling_class", "signal_name", "default_radius_um"],
        dropna=False,
    )
    .agg(
        n_rows=("value", "count"),
        mean_beta=("value", "mean"),
        mean_abs=("abs", "mean"),
        max_abs=("abs", "max"),
        sum_abs=("abs", "sum"),
        targets=("target", lambda s: ",".join(sorted(set(map(str, s)))[:12])),
    )
    .reset_index()
    .sort_values(["sum_abs", "mean_abs"], ascending=False)
)
rank.to_csv(out / "br_term_ranking_spatial.csv", index=False)
top = rank.head(40)
top.to_csv(out / "top_br_terms.csv", index=False)
print(top.to_string(index=False))
if df["target"].astype(str).str.len().gt(0).any():
    per = df.sort_values("abs", ascending=False).groupby("target", as_index=False).head(5)
    per.to_csv(out / "top_br_per_target.csv", index=False)
perf_summary = {}
perf_path = out / "spacetravlr_gene_performance.feather"
if perf_path.exists():
    perf = feather.read_feather(perf_path)
    perf.to_csv(out / "gene_performance.csv", index=False)
    perf_summary = {"n_genes": int(len(perf)), "cols": perf.columns.tolist()}
    for c in perf.columns:
        if pd.api.types.is_numeric_dtype(perf[c]) and ("r2" in c.lower() or "cnn" in c.lower() or "lasso" in c.lower()):
            perf_summary[f"mean_{c}"] = float(perf[c].mean())
payload = {
    "run": "tumor_br_r2x",
    "radii_multiplier": 2,
    "n_feathers": int(len(list(out.glob("*_betadata.feather")))),
    "n_rows_microbiome": int(len(df)),
    "n_br_terms": int(len(rank)),
    "top_br": top.to_dict(orient="records"),
    "pathway_sum_abs": {
        str(k): float(v)
        for k, v in rank.groupby("pathway", dropna=False)["sum_abs"].sum().sort_values(ascending=False).items()
    },
    "perf": perf_summary,
}
(out / "br_spatial_eval.json").write_text(json.dumps(payload, indent=2, default=float))
print("[wrote]", out / "top_br_terms.csv")
PY
echo "[eval-r2x] done=$(date -Is)"
