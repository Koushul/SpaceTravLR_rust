"""Python port of key MERCI R functions (MERCI_LOO_MT_est, MERCI_ReceiverPre, CellNumber_test)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.svm import SVR


def merci_loo_mt_est(
    cell_exp: pd.DataFrame,
    receiver_cells: list[str],
    donor_cells: list[str],
    organism: str = "mouse",
    s_markers: list[str] | None = None,
    epsilon: float = 0.1,
    max_receivers: int | None = 5000,
    seed: int = 0,
) -> pd.DataFrame:
    if organism.lower() == "human":
        mt_genes = [g for g in cell_exp.index if g.startswith("MT-")]
    else:
        mt_genes = [g for g in cell_exp.index if g.lower().startswith("mt-")]

    if s_markers is None:
        s_markers = mt_genes

    receiver_cells = [c for c in receiver_cells if c in cell_exp.columns]
    donor_cells = [c for c in donor_cells if c in cell_exp.columns]

    if max_receivers is not None and len(receiver_cells) > max_receivers:
        rng = np.random.default_rng(seed)
        receiver_cells = list(rng.choice(receiver_cells, size=max_receivers, replace=False))

    marker_d = cell_exp.loc[s_markers, donor_cells]
    ref_d = marker_d.mean(axis=1)
    marker_r = cell_exp.loc[s_markers, receiver_cells]

    zero_cells = marker_r.columns[marker_r.sum(axis=0) == 0]
    if len(zero_cells):
        marker_r = marker_r.drop(columns=zero_cells)

    weights = []
    for col in marker_r.columns:
        t_exp = marker_r[col].to_numpy()
        t_marker_r = marker_r.drop(columns=col)
        ref_r = t_marker_r.mean(axis=1).to_numpy()
        x = np.column_stack([ref_d.to_numpy(), ref_r])
        y = t_exp
        model = SVR(kernel="linear", epsilon=epsilon)
        model.fit(x, y)
        w = model.coef_.ravel()
        weights.append(w)

    cell_w = np.asarray(weights)
    cell_w = np.clip(cell_w, 0, None)
    row_sums = cell_w.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    t_cell_w = cell_w / row_sums

    w_donor_rank = pd.Series(
        stats.rankdata(cell_w[:, 0], method="average"),
        index=marker_r.columns,
        name="W_Donor_rank",
    )
    w_receiver_rank = pd.Series(
        stats.rankdata(-cell_w[:, 1], method="average"),
        index=marker_r.columns,
        name="W_Receiver_rank",
    )

    out = pd.DataFrame(
        {
            "Donor_MT_ind": cell_w[:, 0],
            "Receiver_MT_ind": cell_w[:, 1],
            "Donor_MT_frac": t_cell_w[:, 0],
            "Receiver_MT_frac": t_cell_w[:, 1],
            "W_Donor_rank": w_donor_rank,
            "W_Receiver_rank": w_receiver_rank,
        },
        index=marker_r.columns,
    )
    return out


def merci_receiver_pre(
    dna_rank: pd.DataFrame,
    rna_rank: pd.DataFrame,
    top_rank: float = 50.0,
) -> pd.DataFrame:
    cutoff = top_rank * 0.01
    rank1 = dna_rank["MTvar_rank"]
    rank2 = rna_rank["W_Donor_rank"]
    common = rank1.index.intersection(rank2.index)
    rank1 = rank1.loc[common]
    rank2 = rank2.loc[common]

    break_point = int(np.ceil(np.quantile(np.arange(1, len(common) + 1), cutoff)))
    pa1 = rank1.sort_values(ascending=False).index[:break_point]
    pa2 = rank2.sort_values(ascending=False).index[:break_point]
    positives = set(pa1).intersection(pa2)

    labels = ["Receiver" if c in positives else "non-Receiver" for c in common]
    return pd.DataFrame({"cell": common, "prediction": labels}, index=common)


def cell_number_test(
    dna_rank: pd.DataFrame,
    rna_rank: pd.DataFrame,
    number_r: int = 500,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    common = dna_rank.index.intersection(rna_rank.index)
    d = dna_rank.loc[common, "MTvar_rank"].to_numpy()
    r = rna_rank.loc[common, "W_Donor_rank"].to_numpy()
    n = len(common)
    cutoffs = np.linspace(0.1, 0.8, 8)
    rcm_vals = []
    for c in cutoffs:
        bp = int(np.ceil(c * n))
        observed = 0
        for _ in range(number_r):
            perm = rng.permutation(n)
            d_perm = d[perm]
            r_perm = r[perm]
            d_top = set(np.argsort(-d_perm)[:bp])
            r_top = set(np.argsort(-r_perm)[:bp])
            observed = max(observed, len(d_top.intersection(r_top)))
        expected = bp * bp / n
        rcm_vals.append(observed / expected if expected > 0 else np.nan)
    return pd.DataFrame({"cutoff": cutoffs, "Rcm": rcm_vals})


def donor_mt_signature_score(
    cell_exp: pd.DataFrame,
    donor_cells: list[str],
    candidate_cells: list[str],
    organism: str = "mouse",
) -> pd.Series:
    if organism.lower() == "human":
        mt_genes = [g for g in cell_exp.index if g.startswith("MT-")]
    else:
        mt_genes = [g for g in cell_exp.index if g.lower().startswith("mt-")]

    donor_profile = cell_exp.loc[mt_genes, donor_cells].mean(axis=1)
    donor_profile = donor_profile / (donor_profile.sum() + 1e-9)

    scores = {}
    for cell in candidate_cells:
        if cell not in cell_exp.columns:
            continue
        prof = cell_exp.loc[mt_genes, cell].to_numpy()
        if prof.sum() == 0:
            scores[cell] = 0.0
            continue
        prof = prof / prof.sum()
        scores[cell] = float(np.dot(prof, donor_profile.to_numpy()))
    s = pd.Series(scores)
    rank = pd.Series(stats.rankdata(s.to_numpy(), method="average"), index=s.index, name="MTvar_rank")
    return rank.to_frame()
