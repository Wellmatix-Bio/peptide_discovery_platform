"""Pareto-based multiobjective ranking, similarity clustering, diverse selection and per-candidate rationale."""
from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .. import settings
from ..screening.screening import jaccard, kmers


def dominator_counts(X: np.ndarray, chunk: int = 1000) -> np.ndarray:
    """How many points dominate each point (all objectives maximised). Used to pre-shortlist a large pool cheaply."""
    n = len(X)
    out = np.zeros(n, dtype=np.int32)
    for s in range(0, n, chunk):
        blk = X[s:s + chunk]
        ge = (X[None, :, :] >= blk[:, None, :]).all(-1)
        gt = (X[None, :, :] > blk[:, None, :]).any(-1)
        out[s:s + chunk] = (ge & gt).sum(1)
    return out


def pareto_fronts(X: np.ndarray) -> np.ndarray:
    """Front number (1 = non-dominated) for every row, by repeated peeling."""
    n = len(X)
    front = np.zeros(n, dtype=np.int32)
    remaining = np.arange(n)
    k = 0
    while len(remaining):
        k += 1
        sub = X[remaining]
        ge = (sub[None, :, :] >= sub[:, None, :]).all(-1)
        gt = (sub[None, :, :] > sub[:, None, :]).any(-1)
        dominated = (ge & gt).any(1)
        front[remaining[~dominated]] = k
        remaining = remaining[dominated]
    return front


def diverse_shortlist(seqs: list[str], order: np.ndarray, n: int) -> list[int]:
    """Walk `order` (best first) keeping a peptide only if it is not a near-duplicate of one already kept."""
    kept, kept_k = [], []
    for i in order:
        k = kmers(seqs[i])
        if all(jaccard(k, o) < settings.SHORTLIST_SIMILARITY for o in kept_k):
            kept.append(int(i))
            kept_k.append(k)
            if len(kept) == n:
                break
    return kept


def cluster(seqs: list[str]) -> np.ndarray:
    """Average-linkage clusters on 1 - 3-mer Jaccard; cluster ids start at 1."""
    if len(seqs) < 2:
        return np.ones(len(seqs), dtype=int)
    ks = [kmers(s) for s in seqs]
    d = np.array([[1 - jaccard(a, b) for b in ks] for a in ks])
    np.fill_diagonal(d, 0)
    return fcluster(linkage(squareform(d, checks=False), "average"), settings.CLUSTER_DISTANCE, "distance")


def crowding_distance(Y: np.ndarray, front: np.ndarray) -> np.ndarray:
    """NSGA-II crowding distance within each front (boundary points get a large value). Larger = more isolated, so
    ordering a front by it spreads the choice over the objective space without averaging the objectives."""
    d = np.zeros(len(Y))
    for f in np.unique(front):
        idx = np.where(front == f)[0]
        if len(idx) <= 2:
            d[idx] = 1e9
            continue
        for j in range(Y.shape[1]):
            o = idx[np.argsort(Y[idx, j], kind="stable")]
            span = Y[o[-1], j] - Y[o[0], j]
            d[o[0]] = d[o[-1]] = 1e9
            if span > 0:
                d[o[1:-1]] += (Y[o[2:], j] - Y[o[:-2], j]) / span
    return d


def select_final(front: np.ndarray, crowd: np.ndarray, clusters: np.ndarray, parents: list[str], direct: np.ndarray,
                 n: int, min_direct: int, weak: np.ndarray | None = None, strong: np.ndarray | None = None) -> tuple[list[int], dict]:
    """Evidence-gated, diverse selection. Order inside a front: fewest weak objectives first (a count of poor scores, shown in the output),
    then most strong objectives (top third of the pool), then crowding distance for spread. Counts, never a weighted total.

    Pass 1 takes candidates that pass the evidence gate, front by front, until `min_direct` are chosen (or none are left).
    Pass 2 fills the remaining slots from everything else. Caps: MAX_PER_CLUSTER per sequence cluster and
    MAX_PER_PARENT_CLUSTER per parent protein family. Returns (chosen indices, gate report)."""
    chosen: list[int] = []
    per_c: dict = {}
    per_p: dict = {}

    def take(i: int) -> None:
        if per_c.get(clusters[i], 0) < settings.MAX_PER_CLUSTER and per_p.get(parents[i], 0) < settings.MAX_PER_PARENT_CLUSTER:
            chosen.append(i)
            per_c[clusters[i]] = per_c.get(clusters[i], 0) + 1
            per_p[parents[i]] = per_p.get(parents[i], 0) + 1

    w = np.zeros(len(front), dtype=int) if weak is None else weak
    st = np.zeros(len(front), dtype=int) if strong is None else strong
    order = sorted(range(len(front)), key=lambda i: (front[i], w[i], -st[i], -crowd[i], i))
    want = min(min_direct, n)
    for i in order:
        if len(chosen) >= want:
            break
        if direct[i]:
            take(i)
    n_direct = len(chosen)
    for i in order:
        if len(chosen) >= n:
            break
        if i not in chosen:
            take(i)
    chosen.sort(key=lambda i: (front[i], w[i], -st[i], -crowd[i], i))
    return chosen, {"min_exact_evidence_requested": want, "exact_evidence_selected_in_pass_1": n_direct,
                    "exact_evidence_in_final": int(sum(direct[i] for i in chosen)), "shortfall": max(0, want - n_direct)}


def rationale(row: dict, objs: list[str], pool_best: dict) -> str:
    bits = [f"Pareto front {row['pareto_front']} of {row['n_fronts']}, cluster {row['cluster']}"]
    strong = [o for o in objs if row[o] >= pool_best[o] - 0.05]
    weak = [o for o in objs if row[o] <= 0.5 * pool_best[o]]
    if strong:
        bits.append("near pool-best on " + ", ".join(strong))
    if weak:
        bits.append("weak on " + ", ".join(weak))
    bits.append(f"applicability: {row.get('applicability_overall', 'n/a')} (worst of {row.get('applicability_assessed', '?')} dimensions assessed)")
    kind = row.get("evidence_type", "NO_EVIDENCE")
    if kind == "NO_EVIDENCE":
        bits.append("no evidence peptide matches the sequence (no data, not inactive)")
    else:
        bits.append(f"evidence {kind.lower().replace('_', ' ')}: {row['evidence_peptide']} (reported effect {row['evidence_direction']}, similarity {row['evidence_similarity']}, "
                    f"{row['evidence_study']}; {row['evidence_assay'] or 'assay not recorded'}; "
                    f"food origin: {row['evidence_food_derivation'].replace('_', ' ')}; a sequence relationship, not proof of GLP-1 activity)")
    if row.get("template_evidence_type") not in (None, "", "NO_EVIDENCE"):
        bits.append(f"its template has {row['template_evidence_type'].lower().replace('_', ' ')} with {row['template_evidence_peptide']} (shown only, not transferred to this peptide)")
    if row.get("prior_art_alert"):
        bits.append("prior-art alert: " + row["prior_art_alert"])
    if row.get("liabilities"):
        bits.append("liabilities: " + row["liabilities"])
    return "; ".join(bits)


def _dominance(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(strength, dominated_by): how many rows each row dominates, and how many rows dominate it. All objectives maximised."""
    ge = (Y[:, None, :] >= Y[None, :, :]).all(-1)
    gt = (Y[:, None, :] > Y[None, :, :]).any(-1)
    dom = ge & gt                                  # dom[i, j]: i dominates j
    return dom.sum(1), dom.sum(0)


def pareto_pick_selection(pool: list[dict], objs: list[str], clusters, parents: list[str], n: int, min_direct: int,
                          tiebreak: tuple | None = None, max_per_cluster: int | None = None, max_per_parent: int | None = None):
    """Iterative Pareto selection. No thresholds, no pool percentiles, no randomness.

    pick: among the remaining candidates take the best of Pareto front 1 (the peptide that dominates the most remaining candidates; ties by
          the fixed `tiebreak` columns, then sequence). Put it in the list, remove it from the candidates, repeat until n are chosen.
    scan: (a) diversity: the first peptide, in pick order, that takes a sequence cluster or parent family over its cap is dropped for good
          (never returned to the candidates); (b) evidence gate: if fewer than `min_direct` pass the evidence gate (EXACT_SEQUENCE), the last-added peptide without it
          peptide is dropped for good. After a drop the slot is refilled with the same pick rule (from direct-evidence candidates only while
          the gate is unmet) and the scan runs again, until nothing is dropped. Rank = order of addition.
    Returns (chosen pool indices in rank order, report, per-peptide fields for the whole pool)."""
    tiebreak = settings.TIEBREAK if tiebreak is None else tiebreak
    max_per_cluster = settings.MAX_PER_CLUSTER if max_per_cluster is None else max_per_cluster
    max_per_parent = settings.MAX_PER_PARENT_CLUSTER if max_per_parent is None else max_per_parent
    Yall = np.array([[c[o] for o in objs] for c in pool])
    direct = np.array([bool(c["evidence_gate_passed"]) for c in pool])  # an EXACT_SEQUENCE match with a positive or mixed effect
    chosen: list[int] = []
    dominated_n: dict[int, int] = {}
    removed: set[int] = set()
    dropped = {"diversity": [], "evidence_gate": []}

    def pick_one(direct_only: bool) -> bool:
        rem = [i for i in range(len(pool)) if i not in removed and (direct[i] or not direct_only)]
        if not rem:
            return False
        strength, dominated_by = _dominance(Yall[rem])
        front1 = [j for j in range(len(rem)) if dominated_by[j] == 0]
        j = min(front1, key=lambda j: (-strength[j], *[-pool[rem[j]].get(t, 0.0) for t in tiebreak], pool[rem[j]]["sequence"]))
        chosen.append(rem[j])
        removed.add(rem[j])
        dominated_n[rem[j]] = int(strength[j])
        return True

    def gate_unmet() -> bool:
        return len(chosen) >= min(min_direct, n) and sum(direct[i] for i in chosen) < min(min_direct, n)

    while len(chosen) < n and pick_one(False):
        pass
    for _ in range(10 * len(pool)):                       # each pass drops one peptide for good, so this terminates
        per_c: dict = {}
        per_p: dict = {}
        bad = None
        for i in chosen:                                  # pick order: the first one over a cap is the last added of its group
            per_c[clusters[i]] = per_c.get(clusters[i], 0) + 1
            per_p[parents[i]] = per_p.get(parents[i], 0) + 1
            if per_c[clusters[i]] > max_per_cluster or per_p[parents[i]] > max_per_parent:
                bad = i
                break
        reason = "diversity"
        if bad is None and gate_unmet():
            expl = [i for i in chosen if not direct[i]]
            if expl:
                bad, reason = expl[-1], "evidence_gate"
        if bad is None:
            break
        chosen.remove(bad)                                # stays in `removed`: never goes back to the candidates
        dropped[reason].append(pool[bad]["sequence"])
        if not pick_one(reason == "evidence_gate"):     # gate refill: direct-evidence candidates only; diversity refill: any candidate
            if reason == "evidence_gate":
                break                                     # no direct-evidence candidate left: report the shortfall, do not loop
            pick_one(False)
    fr_all = pareto_fronts(Yall)
    per = {"pareto_front": [int(f) for f in fr_all], "n_fronts": int(fr_all.max()),
           "dominates_n_at_pick": [dominated_n.get(i, "") for i in range(len(pool))]}
    rep = {"selected": len(chosen), "exact_evidence_in_final": int(sum(direct[i] for i in chosen)), "min_exact_evidence_requested": min(min_direct, n),
           "shortfall": max(0, min(min_direct, n) - int(sum(direct[i] for i in chosen))), "dropped_for_diversity": dropped["diversity"],
           "dropped_for_evidence_gate": dropped["evidence_gate"]}
    return chosen, rep, per
