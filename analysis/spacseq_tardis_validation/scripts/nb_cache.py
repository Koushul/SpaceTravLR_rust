"""Load cached validation artifacts for interactive notebooks."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SECTIONS = [
    "multislice",
    "spatial",
    "scorecard",
    "beta_leiden",
    "niche_deg",
    "niche_spp1",
    "paper",
    "extended_paper",
    "dashboard",
    "cnn",
]


def _artifact_map(tag: str, spatial_tag: str, cnn_tag: str) -> dict[str, dict[str, str]]:
    t, st, ct = tag, spatial_tag, cnn_tag
    return {
        "multislice": {
            "overall": f"results/multislice/overall_summary_{t}.json",
            "combined": f"results/multislice/per_celltype_corr_all_slices_{t}.csv",
            "meta": f"results/multislice/meta_analysis_{t}.csv",
        },
        "spatial": {
            "overall": f"results/spatial/overall_spatial_{t}.json",
            "summary": f"results/spatial/spatial_summary_{t}.csv",
            "niche_corr": f"results/spatial/niche_corr_{t}.csv",
        },
        "scorecard": {
            "summary": "results/scorecard/scorecard_summary.json",
            "table": "results/scorecard/prediction_scorecard.csv",
        },
        "beta_leiden": {
            "overall": f"results/beta_leiden/overall_{t}.json",
            "summary": f"results/beta_leiden/summary_{t}.csv",
            "niche_corr": f"results/beta_leiden/niche_corr_{t}.csv",
            "silhouette": f"results/beta_leiden/silhouette_{t}.csv",
        },
        "niche_deg": {
            "overall": f"results/niche_deg/overall_{st}.json",
            "spatial_neighbor": f"results/niche_deg/spatial_neighbor_stats_{st}.csv",
            "ccc": f"results/niche_deg/ccc_state_scores_{st}.csv",
        },
        "niche_spp1": {
            "overall": f"results/niche_spp1/overall_{t}.json",
            "direct_deg": f"results/niche_spp1/direct_cell_deg_stats_{t}.csv",
            "spp1_module": f"results/niche_spp1/spp1_module_{t}.csv",
            "spp1_tracking": f"results/niche_spp1/spp1_tracking_{t}.csv",
        },
        "paper": {
            "overall": f"results/paper_findings/overall_{t}.json",
            "modules": f"results/paper_findings/hypothesis_scores_{t}.csv",
            "gene_level": f"results/paper_findings/gene_level_{t}.csv",
        },
        "extended_paper": {
            "overall": f"results/extended_paper/overall_{t}.json",
            "lung_icam1": f"results/extended_paper/lung_icam1_modules_{t}.csv",
            "lung_bcam": f"results/extended_paper/lung_bcam_modules_{t}.csv",
            "subq_icam1": f"results/extended_paper/subq_icam1_modules_{t}.csv",
            "subq_lung_icam1": f"results/extended_paper/subq_vs_lung_icam1_{t}.csv",
            "in_silico": f"results/extended_paper/in_silico_spp1_cd44_{t}.csv",
        },
        "dashboard": {
            "overall": f"results/validation_dashboard/overall_{st}.json",
            "metrics": f"results/validation_dashboard/metrics_{st}.csv",
        },
        "cnn": {
            "overall": f"results/cnn_enrichment/overall_{ct}.json",
            "enrichment": f"results/cnn_enrichment/niche_enrichment_{ct}.csv",
            "corr": f"results/cnn_enrichment/enrichment_corr_{ct}.csv",
        },
    }


def build_manifest(
    tag: str,
    spatial_tag: str,
    cnn_tag: str,
    cfg: dict,
    sections: list[str],
) -> dict:
    from datetime import datetime, timezone

    arts = _artifact_map(tag, spatial_tag, cnn_tag)
    manifest_sections: dict[str, dict] = {}
    missing: list[str] = []
    for sec in sections:
        if sec not in arts:
            continue
        files = arts[sec]
        present = {k: v for k, v in files.items() if (ROOT / v).exists()}
        for k, v in files.items():
            if k not in present:
                missing.append(v)
        manifest_sections[sec] = {"artifacts": present}
    return {
        "version": 1,
        "tag": tag,
        "spatial_tag": spatial_tag,
        "cnn_tag": cnn_tag,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": cfg,
        "sections": manifest_sections,
        "missing": missing,
    }


@dataclass
class ValidationBundle:
    """In-memory view of one cached validation run."""

    root: Path
    tag: str
    spatial_tag: str
    cnn_tag: str
    manifest: dict[str, Any] = field(repr=False)
    config: dict[str, Any] = field(default_factory=dict)

    def artifact(self, section: str, name: str) -> Path | None:
        sec = self.manifest.get("sections", {}).get(section, {})
        rel = sec.get("artifacts", {}).get(name)
        if not rel:
            return None
        path = self.root / rel
        return path if path.exists() else None

    def table(self, section: str, name: str) -> pd.DataFrame:
        path = self.artifact(section, name)
        return pd.read_csv(path) if path else pd.DataFrame()

    def json(self, section: str, name: str) -> dict:
        path = self.artifact(section, name)
        return json.loads(path.read_text()) if path else {}

    def missing(self) -> list[str]:
        return list(self.manifest.get("missing", []))

    def sections(self) -> list[str]:
        return list(self.manifest.get("sections", {}).keys())

    def has(self, section: str) -> bool:
        arts = self.manifest.get("sections", {}).get(section, {}).get("artifacts", {})
        return bool(arts)


def default_config_from_manifest(manifest: dict[str, Any]) -> dict[str, str]:
    cfg = manifest.get("config", {})
    tag = manifest.get("tag", "tuned")
    return {
        "tag": tag,
        "spatial_tag": manifest.get("spatial_tag", "spatial_v3"),
        "cnn_tag": manifest.get("cnn_tag", "cnn"),
        "pred_dir": cfg.get("pred_dir", f"results/predictions_{tag}"),
        "pred_dir_cnn": "results/predictions_cnn",
        "betadata_dir": cfg.get("betadata_dir", "runs/baseline_pooled_seed"),
        "betadata_dir_cnn": "runs/baseline_pooled_cnn",
        "baseline_h5ad": cfg.get("baseline_h5ad", "data/pooled/baseline_ntc.h5ad"),
    }


def load_cache(tag: str = "tuned", root: Path | None = None) -> ValidationBundle:
    """Load cache/{tag}/manifest.json and expose tables/JSON helpers."""
    root = root or ROOT
    manifest_path = root / "cache" / tag / "manifest.json"
    if not manifest_path.exists():
        cfg_all = json.loads((root / "config/validation_runs.json").read_text())
        manifest = build_manifest(
            tag=tag,
            spatial_tag="spatial_v3",
            cnn_tag="cnn",
            cfg=cfg_all["models"]["pooled_tuned"],
            sections=DEFAULT_SECTIONS,
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2))
    else:
        manifest = json.loads(manifest_path.read_text())

    return ValidationBundle(
        root=root,
        tag=manifest.get("tag", tag),
        spatial_tag=manifest.get("spatial_tag", "spatial_v3"),
        cnn_tag=manifest.get("cnn_tag", "cnn"),
        manifest=manifest,
        config=manifest.get("config", {}),
    )
