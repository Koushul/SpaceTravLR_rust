"""Refined cell-type labels for Visium HD bins (GSE280315 P1 CRC)."""

from __future__ import annotations

import numpy as np
import pandas as pd

TUMOR_PREFIX = "Tumor"
T_CELL_LABELS = {"CD4 T cell", "CD8 T cell", "NK", "Proliferating Immune II", "Treg"}
B_CELL_LABELS = {"Mature B", "Memory B", "Plasma", "Plasma cell"}
MYELOID_LABELS = {
    "Macrophage", "Proliferating Macrophages", "Neutrophil", "Mast", "Monocyte",
    "mRegDC", "cDC I", "pDC", "Dendritic cell",
}
STROMA_LABELS = {
    "CAF", "Fibroblast", "Proliferating Fibroblast", "Myofibroblast", "Pericytes",
    "vSM", "Smooth Muscle", "SM Stress Response", "Vascular Fibroblast",
}
ENDOTHELIAL_LABELS = {"Endothelial", "Lymphatic Endothelial"}
EPITHELIAL_LABELS = {
    "Goblet", "Enterocyte", "Epithelial", "Tuft", "Neuroendocrine", "Enteric Glial",
    "Adipocyte",
}
OTHER_LABELS = {"Unknown III (SM)"}

UL1_LINEAGE = {
    "Tumor": "tumor",
    "T cells": "immune",
    "B cells": "immune",
    "Myeloid": "myeloid",
    "Fibroblast": "stroma",
    "Endothelial": "endothelial",
    "Intestinal Epithelial": "epithelial",
    "Smooth Muscle": "stroma",
    "Neuronal": "neuronal",
    "Unknown": "unknown",
}

LINEAGE_ORDER = ["tumor", "immune", "myeloid", "stroma", "endothelial", "epithelial", "neuronal", "unknown"]


def _lineage_from_l1(label1: str) -> str:
    if label1.startswith(TUMOR_PREFIX):
        return "tumor"
    if label1 in T_CELL_LABELS or "T cell" in label1:
        return "immune"
    if label1 in B_CELL_LABELS:
        return "immune"
    if label1 in MYELOID_LABELS:
        return "myeloid"
    if label1 in STROMA_LABELS:
        return "stroma"
    if label1 in ENDOTHELIAL_LABELS:
        return "endothelial"
    if label1 in EPITHELIAL_LABELS:
        return "epithelial"
    if label1 in OTHER_LABELS:
        return "unknown"
    return "unknown"


def _fine_from_ul2(unsupervised_l2: str, lineage: str) -> str | None:
    ul2 = str(unsupervised_l2)
    if lineage == "immune" and ul2.startswith("Tcells"):
        return "T cell (unsupervised)"
    if lineage == "myeloid" and ul2.startswith("Myeloid"):
        return "Myeloid (unsupervised)"
    if lineage == "stroma" and ul2.startswith("Fibroblast"):
        return "Fibroblast (unsupervised)"
    if lineage == "epithelial" and ul2.startswith("IntestinalEpithelial"):
        return "Intestinal epithelial (unsupervised)"
    return None


def _resolve_fine_label(label1: str, label2: str, unsupervised_l2: str, lineage: str) -> str:
    l1 = str(label1)
    l2 = str(label2)

    if lineage == "tumor" and l1.startswith(TUMOR_PREFIX):
        return l1

    if lineage == "immune":
        if l1 in T_CELL_LABELS or "T cell" in l1:
            return l1
        if l1 in B_CELL_LABELS:
            return l1
        if l2 in T_CELL_LABELS | B_CELL_LABELS:
            return l2
        alt = _fine_from_ul2(unsupervised_l2, lineage)
        return alt or "T cell (unsupervised)"

    if lineage == "myeloid":
        if l1 in MYELOID_LABELS:
            return l1
        if l2 in MYELOID_LABELS:
            return l2
        alt = _fine_from_ul2(unsupervised_l2, lineage)
        return alt or "Myeloid (unsupervised)"

    if lineage == "stroma":
        if l1 in STROMA_LABELS:
            return l1
        if l2 in STROMA_LABELS:
            return l2
        alt = _fine_from_ul2(unsupervised_l2, lineage)
        return alt or "Stromal (unsupervised)"

    if lineage == "endothelial":
        if l1 in ENDOTHELIAL_LABELS:
            return l1
        if l2 in ENDOTHELIAL_LABELS:
            return l2
        return "Endothelial"

    if lineage == "epithelial":
        if l1 in EPITHELIAL_LABELS:
            return l1
        if l2 in EPITHELIAL_LABELS:
            return l2
        alt = _fine_from_ul2(unsupervised_l2, lineage)
        return alt or "Intestinal epithelial (unsupervised)"

    if lineage == "neuronal":
        if l1 == "Enteric Glial" or l2 == "Enteric Glial":
            return "Enteric Glial"
        return "Neuronal (unsupervised)"

    return l1 if l1 else "Unknown"


def _lineage_from_scores(row: pd.Series) -> str:
    score_cols = {
        "tumor": row.get("score_tumor", 0.0),
        "immune": row.get("score_immune", 0.0),
        "myeloid": row.get("score_myeloid", 0.0),
        "stroma": row.get("score_stroma", 0.0),
        "epithelial": row.get("score_epithelial", 0.0),
    }
    return max(score_cols, key=score_cols.get)


def refine_cell_types(obs: pd.DataFrame) -> pd.DataFrame:
    """Return obs with lineage_refined, cell_type_refined, annotation_source columns."""
    out = obs.copy()
    lineages = []
    fine = []
    sources = []

    for idx, row in out.iterrows():
        l1 = str(row.get("DeconvolutionLabel1", ""))
        l2 = str(row.get("DeconvolutionLabel2", ""))
        ul1 = str(row.get("UnsupervisedL1", ""))
        ul2 = str(row.get("UnsupervisedL2", ""))

        l1_lin = _lineage_from_l1(l1)
        ul1_lin = UL1_LINEAGE.get(ul1, "unknown")

        if l1_lin == ul1_lin:
            lineage = l1_lin
            source = "label1+ul1_agree"
        elif l1_lin == "unknown" or ul1_lin == "unknown":
            lineage = ul1_lin if l1_lin == "unknown" else l1_lin
            source = "single_source"
        else:
            score_lin = _lineage_from_scores(row)
            if score_lin in {l1_lin, ul1_lin}:
                lineage = score_lin
                source = "marker_tiebreak"
            else:
                lineage = ul1_lin
                source = "ul1_preferred"

        fine_label = _resolve_fine_label(l1, l2, ul2, lineage)
        lineages.append(lineage)
        fine.append(fine_label)
        sources.append(source)

    out["lineage_refined"] = lineages
    out["cell_type_refined"] = fine
    out["annotation_source"] = sources
    return out
