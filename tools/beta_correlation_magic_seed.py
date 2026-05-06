#!/usr/bin/env python3
"""Run matched MAGIC-imputed SpaceTravLR seed runs and compare exported betas."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ID_COLUMN_PRIORITY = ("CellID", "obs_names", "cell_id", "Cluster", "index", "__index_level_0__")
INTERCEPT_COLUMNS = {"beta0", "beta_0"}


@dataclass(frozen=True)
class BetaValue:
    gene: str
    row_id: str
    coefficient: str
    left: float
    right: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "MAGIC-impute both H&E-derived and real spatial AnnData inputs, train matched "
            "10-gene SpaceTravLR seed models, and test whether betadata coefficients correlate."
        )
    )
    parser.add_argument("--he-h5ad", required=True, type=Path, help="H&E-imputed gene-expression AnnData")
    parser.add_argument("--st-h5ad", required=True, type=Path, help="Real spatial transcriptomics AnnData")
    parser.add_argument("--genes", help="Comma-separated target genes. The first 10 are used unless --allow-non10 is set.")
    parser.add_argument("--genes-file", type=Path, help="One target gene per line; comments beginning with # are ignored.")
    parser.add_argument("--allow-non10", action="store_true", help="Allow a target count other than 10.")
    parser.add_argument("--config", type=Path, default=Path("spaceship_config.toml"), help="SpaceTravLR config overlay.")
    parser.add_argument("--work-dir", type=Path, default=Path("beta_correlation_magic_runs"))
    parser.add_argument(
        "--spacetravlr-cmd",
        default="cargo run --bin spacetravlr --",
        help="Command prefix used to run spacetravlr; pass an installed binary path to avoid cargo.",
    )
    parser.add_argument("--skip-magic", action="store_true", help="Use the inputs as already-MAGIC-imputed H5ADs.")
    parser.add_argument("--magic-batch-obs", help="Batch obs column for MAGIC; passed to both inputs.")
    parser.add_argument("--condition", help="Training condition split and default MAGIC batch obs when set by SpaceTravLR.")
    parser.add_argument("--random-seed", default=42, type=int)
    parser.add_argument("--parallel", default=1, type=int)
    parser.add_argument("--n-iter", default=500, type=int)
    parser.add_argument("--tol", default=1e-4, type=float)
    parser.add_argument("--l1-reg", type=float)
    parser.add_argument("--group-reg", type=float)
    parser.add_argument("--max-ligands", type=int)
    parser.add_argument("--force-cpu", action="store_true", help="Set SPACETRAVLR_FORCE_CPU=1 for both training runs.")
    parser.add_argument("--train-extra-arg", action="append", default=[], help="Extra argument appended to both training commands.")
    parser.add_argument("--magic-extra-arg", action="append", default=[], help="Extra argument appended to both MAGIC commands.")
    parser.add_argument("--min-global-r", default=0.5, type=float)
    parser.add_argument("--min-median-gene-r", default=0.5, type=float)
    parser.add_argument("--min-common-coefs", default=3, type=int)
    parser.add_argument("--min-common-rows", default=1, type=int)
    parser.add_argument("--keep-going", action="store_true", help="Continue comparison when individual gene feathers are missing.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser.parse_args()


def read_genes(args: argparse.Namespace) -> list[str]:
    genes: list[str] = []
    if args.genes:
        genes.extend(g.strip() for g in args.genes.split(",") if g.strip())
    if args.genes_file:
        for line in args.genes_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                genes.append(line)

    deduped: list[str] = []
    seen: set[str] = set()
    for gene in genes:
        if gene not in seen:
            deduped.append(gene)
            seen.add(gene)

    if not deduped:
        raise SystemExit("Provide exactly 10 target genes with --genes and/or --genes-file.")
    if len(deduped) != 10 and not args.allow_non10:
        if len(deduped) > 10:
            print(f"Using first 10 of {len(deduped)} provided genes.", file=sys.stderr)
            return deduped[:10]
        raise SystemExit(f"Expected 10 genes, got {len(deduped)}. Pass --allow-non10 to override.")
    return deduped


def command_prefix(raw: str) -> list[str]:
    cmd = shlex.split(raw)
    if not cmd:
        raise SystemExit("--spacetravlr-cmd cannot be empty")
    return cmd


def run_command(cmd: list[str], env: dict[str, str], dry_run: bool) -> None:
    printable = " ".join(shlex.quote(part) for part in cmd)
    print(f"+ {printable}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, env=env, check=True)


def magic_output_path(work_dir: Path, label: str) -> Path:
    return work_dir / "magic" / f"{label}_rust_magic.h5ad"


def run_magic(args: argparse.Namespace, label: str, input_h5ad: Path, env: dict[str, str]) -> Path:
    if args.skip_magic:
        return input_h5ad

    out = magic_output_path(args.work_dir, label)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = command_prefix(args.spacetravlr_cmd) + [
        "--h5ad",
        str(input_h5ad),
        "--umap",
        "--leiden",
        "--rust-magic",
        "--output",
        str(out),
        "--plain",
    ]
    if args.condition:
        cmd += ["--condition", args.condition]
    if args.magic_batch_obs:
        cmd += ["--magic-batch-obs", args.magic_batch_obs]
    cmd += args.magic_extra_arg
    run_command(cmd, env, args.dry_run)
    return out


def train_seed(args: argparse.Namespace, label: str, h5ad: Path, genes: list[str], env: dict[str, str]) -> Path:
    out_dir = args.work_dir / "training" / label
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = command_prefix(args.spacetravlr_cmd) + [
        "--plain",
        "--clean-output-dir",
        "--config",
        str(args.config),
        "--h5ad",
        str(h5ad),
        "--output-dir",
        str(out_dir),
        "--training-mode",
        "seed",
        "--genes",
        ",".join(genes),
        "--max-genes",
        str(len(genes)),
        "--random-seed",
        str(args.random_seed),
        "--parallel",
        str(args.parallel),
        "--n-iter",
        str(args.n_iter),
        "--tol",
        str(args.tol),
    ]
    if args.condition:
        cmd += ["--condition", args.condition]
    if args.l1_reg is not None:
        cmd += ["--l1-reg", str(args.l1_reg)]
    if args.group_reg is not None:
        cmd += ["--group-reg", str(args.group_reg)]
    if args.max_ligands is not None:
        cmd += ["--max-ligands", str(args.max_ligands)]
    cmd += args.train_extra_arg
    run_command(cmd, env, args.dry_run)
    return out_dir


def import_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("Comparison requires pandas with pyarrow/feather support.") from exc
    return pd


def id_column(columns: Iterable[str]) -> str | None:
    names = list(columns)
    for candidate in ID_COLUMN_PRIORITY:
        if candidate in names:
            return candidate
    for name in names:
        if name.startswith("__index"):
            return name
    return None


def coefficient_columns(df, row_id_column: str | None) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        if col == row_id_column or col in INTERCEPT_COLUMNS:
            continue
        try:
            df[col].astype(float)
        except (TypeError, ValueError):
            continue
        cols.append(col)
    return cols


def normalize_coefficient(name: str) -> str:
    return name.removeprefix("beta_")


def feather_for_gene(directory: Path, gene: str) -> Path:
    direct = directory / f"{gene}_betadata.feather"
    if direct.exists():
        return direct
    matches = sorted(directory.glob(f"**/{gene}_betadata.feather"))
    if matches:
        return matches[0]
    return direct


def aligned_gene_values(left_dir: Path, right_dir: Path, gene: str, keep_going: bool) -> list[BetaValue]:
    pd = import_pandas()
    left_path = feather_for_gene(left_dir, gene)
    right_path = feather_for_gene(right_dir, gene)
    if not left_path.exists() or not right_path.exists():
        message = f"Missing betadata for {gene}: {left_path} or {right_path}"
        if keep_going:
            print(f"Warning: {message}", file=sys.stderr)
            return []
        raise SystemExit(message)

    left = pd.read_feather(left_path)
    right = pd.read_feather(right_path)
    left_id = id_column(left.columns)
    right_id = id_column(right.columns)
    if left_id is None:
        left = left.copy()
        left_id = "__row"
        left[left_id] = [str(i) for i in range(len(left))]
    if right_id is None:
        right = right.copy()
        right_id = "__row"
        right[right_id] = [str(i) for i in range(len(right))]

    left = left.copy()
    right = right.copy()
    left[left_id] = left[left_id].astype(str)
    right[right_id] = right[right_id].astype(str)
    left_cols = {normalize_coefficient(c): c for c in coefficient_columns(left, left_id)}
    right_cols = {normalize_coefficient(c): c for c in coefficient_columns(right, right_id)}
    common_coefs = sorted(set(left_cols) & set(right_cols))
    common_rows = sorted(set(left[left_id]) & set(right[right_id]))

    if len(common_coefs) == 0 or len(common_rows) == 0:
        message = f"No aligned beta values for {gene}: {len(common_rows)} rows, {len(common_coefs)} coefficients"
        if keep_going:
            print(f"Warning: {message}", file=sys.stderr)
            return []
        raise SystemExit(message)

    left = left.set_index(left_id, drop=False)
    right = right.set_index(right_id, drop=False)
    values: list[BetaValue] = []
    for row_id in common_rows:
        for coef in common_coefs:
            lv = float(left.at[row_id, left_cols[coef]])
            rv = float(right.at[row_id, right_cols[coef]])
            if math.isfinite(lv) and math.isfinite(rv):
                values.append(BetaValue(gene, row_id, coef, lv, rv))
    return values


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    sx = math.sqrt(sum(x * x for x in dx))
    sy = math.sqrt(sum(y * y for y in dy))
    if sx == 0.0 or sy == 0.0:
        return None
    return sum(x * y for x, y in zip(dx, dy)) / (sx * sy)


def grouped_correlations(values: list[BetaValue], key_attr: str) -> list[dict[str, object]]:
    groups: dict[str, list[BetaValue]] = {}
    for value in values:
        groups.setdefault(getattr(value, key_attr), []).append(value)
    rows: list[dict[str, object]] = []
    for key, group in sorted(groups.items()):
        r = pearson([v.left for v in group], [v.right for v in group])
        rows.append({"key": key, "n": len(group), "pearson_r": r})
    return rows


def median(nums: list[float]) -> float | None:
    if not nums:
        return None
    nums = sorted(nums)
    mid = len(nums) // 2
    if len(nums) % 2:
        return nums[mid]
    return (nums[mid - 1] + nums[mid]) / 2.0


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def compare_runs(args: argparse.Namespace, left_dir: Path, right_dir: Path, genes: list[str]) -> dict[str, object]:
    all_values: list[BetaValue] = []
    for gene in genes:
        gene_values = aligned_gene_values(left_dir, right_dir, gene, args.keep_going)
        all_values.extend(gene_values)

    if not all_values:
        raise SystemExit("No aligned beta values were available for comparison.")

    global_r = pearson([v.left for v in all_values], [v.right for v in all_values])
    per_gene = grouped_correlations(all_values, "gene")
    per_coef = grouped_correlations(all_values, "coefficient")
    valid_gene_rs = [row["pearson_r"] for row in per_gene if row["pearson_r"] is not None]
    median_gene_r = median(valid_gene_rs)

    pair_rows = [
        {
            "gene": v.gene,
            "row_id": v.row_id,
            "coefficient": v.coefficient,
            "he_magic_beta": v.left,
            "st_magic_beta": v.right,
        }
        for v in all_values
    ]
    summary = {
        "n_genes_requested": len(genes),
        "n_genes_compared": len({v.gene for v in all_values}),
        "n_beta_pairs": len(all_values),
        "n_coefficients": len({v.coefficient for v in all_values}),
        "n_rows": len({v.row_id for v in all_values}),
        "global_pearson_r": global_r,
        "median_gene_pearson_r": median_gene_r,
        "min_global_r": args.min_global_r,
        "min_median_gene_r": args.min_median_gene_r,
    }

    report_dir = args.work_dir / "comparison"
    write_csv(report_dir / "aligned_beta_pairs.csv", pair_rows)
    write_csv(report_dir / "per_gene_correlations.csv", per_gene)
    write_csv(report_dir / "per_coefficient_correlations.csv", per_coef)
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    failures: list[str] = []
    if global_r is None or global_r < args.min_global_r:
        failures.append(f"global Pearson r {global_r} < {args.min_global_r}")
    if median_gene_r is None or median_gene_r < args.min_median_gene_r:
        failures.append(f"median per-gene Pearson r {median_gene_r} < {args.min_median_gene_r}")
    if summary["n_coefficients"] < args.min_common_coefs:
        failures.append(f"common coefficients {summary['n_coefficients']} < {args.min_common_coefs}")
    if summary["n_rows"] < args.min_common_rows:
        failures.append(f"common rows {summary['n_rows']} < {args.min_common_rows}")

    print(json.dumps(summary, indent=2, sort_keys=True))
    if failures:
        raise SystemExit("Beta correlation test failed: " + "; ".join(failures))
    return summary


def main() -> None:
    args = parse_args()
    genes = read_genes(args)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if args.force_cpu:
        env["SPACETRAVLR_FORCE_CPU"] = "1"

    he_magic = run_magic(args, "he", args.he_h5ad, env)
    st_magic = run_magic(args, "st", args.st_h5ad, env)
    he_run = train_seed(args, "he_magic_seed", he_magic, genes, env)
    st_run = train_seed(args, "st_magic_seed", st_magic, genes, env)

    if args.dry_run:
        return
    compare_runs(args, he_run, st_run, genes)


if __name__ == "__main__":
    main()
