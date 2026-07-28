#!/bin/bash
# Collect microbiome-mode interactions and rank top BR terms (run after training).
set -euo pipefail
export HDF5_USE_FILE_LOCKING=FALSE
BIN=/ix1/ylee/kor11/tools/SpaceTravLR_rust/target/release/spacetravlr
OUT=/ix1/ylee/kor11/tools/spacetravlr_microbiome/runs/tumor_br_full
TOML="$OUT/spacetravlr_run_repro.toml"
PY=/ix1/ylee/kor11/tools/susceptibility_pilot/.venv/bin/python

[[ -f "$TOML" ]] || { echo "missing $TOML"; exit 1; }
n_beta=$(find "$OUT" -name '*_betadata.feather' 2>/dev/null | wc -l)
echo "[collect] betadata=$n_beta"

"$BIN" collect-interactions \
  --run-toml "$TOML" \
  --annot cell_type \
  --aggregate mean \
  --mode microbiome \
  --out "$OUT/plucked_feathers_microbiome.feather"

"$BIN" collect-interactions \
  --run-toml "$TOML" \
  --annot cell_type \
  --aggregate mean \
  --mode all \
  --out "$OUT/plucked_feathers_all.feather"

"$PY" - <<'PY'
import json
from pathlib import Path
import pandas as pd
import pyarrow.feather as feather

out = Path("/ix1/ylee/kor11/tools/spacetravlr_microbiome/runs/tumor_br_full")
inter = pd.read_csv(
    "/ix1/ylee/kor11/tools/spacetravlr_microbiome/configs/bact_host_interactions.train_scfa_merged.csv"
)
meta = inter.rename(columns={"signal_id": "signal"}).drop_duplicates(["signal", "receptor"])

df = feather.read_feather(out / "plucked_feathers_microbiome.feather")
print("columns", df.columns.tolist())
print("n", len(df))

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
    raise SystemExit(f"no interaction col in {df.columns.tolist()}")
if "value" not in df.columns:
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num:
        raise SystemExit("no numeric value column")
    df["value"] = df[num[0]]
if "target" not in df.columns:
    df["target"] = ""

df["abs"] = df["value"].abs()
clean = df["interaction"].astype(str).str.replace(r"^beta_", "", regex=True)
parts = clean.str.split("$", n=1, expand=True)
df["signal"] = parts[0]
df["receptor"] = parts[1] if parts.shape[1] > 1 else ""
df = df.merge(
    meta[["signal", "receptor", "pathway", "signaling_class", "signal_name"]],
    on=["signal", "receptor"],
    how="left",
)

rank = (
    df.groupby(
        ["interaction", "signal", "receptor", "pathway", "signaling_class", "signal_name"],
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

def tag_row(r):
    path = str(r.get("pathway") or "")
    targets = str(r.get("targets") or "")
    mb = float(r["mean_beta"])
    if path in ("TLR", "NLR") and mb > 0 and any(
        t in targets for t in ("Defa", "Reg3", "Lyz", "Rela", "Nfkb", "Nos2", "Duox", "Muc", "Ang")
    ):
        return "plausible_pamp"
    if path == "GPCR_SCFA":
        return "scfa"
    if path == "FPR":
        return "fpr"
    return "review"

rank["bio_tag"] = [tag_row(r) for _, r in rank.iterrows()]
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
        if pd.api.types.is_numeric_dtype(perf[c]) and (
            "r2" in c.lower() or "cnn" in c.lower() or "lasso" in c.lower()
        ):
            perf_summary[f"mean_{c}"] = float(perf[c].mean())

payload = {
    "n_rows_microbiome": int(len(df)),
    "n_br_terms": int(len(rank)),
    "top_br": top.to_dict(orient="records"),
    "pathway_sum_abs": {
        str(k): float(v)
        for k, v in rank.groupby("pathway", dropna=False)["sum_abs"]
        .sum()
        .sort_values(ascending=False)
        .items()
    },
    "bio_tag_counts": rank["bio_tag"].value_counts().to_dict(),
    "perf": perf_summary,
}
(out / "br_spatial_eval.json").write_text(json.dumps(payload, indent=2, default=float))
print("[wrote]", out / "top_br_terms.csv")
print("[wrote]", out / "br_spatial_eval.json")
PY

echo "[collect] done"
