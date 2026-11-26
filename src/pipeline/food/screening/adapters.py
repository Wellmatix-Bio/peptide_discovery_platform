"""Adapters that give the platform's model_store predictors the screening contract: predict(seqs) -> {seq: {score, label, applicability, detail}}.

Score is in [0, 1], higher = safer / more developable. A sequence a model cannot score is left out of its result and listed in `skipped`; the stage reports
it as not run. Every output is a raw probability of a model trained on a different population from food peptides.
Ported from Phase 1 `7. AI Screening and Ranking/adapters/platform_screens.py`; the platform's own FeatureExtractor replaces the one Phase 1 built over D:.
"""
from __future__ import annotations

import numpy as np

from .model_registry import ScreeningModel

SOLVENT = "Ultrapure water"


class _Base:
    def __init__(self, feature_extractor):
        self.fe = feature_extractor
        self.skipped: dict[str, str] = {}  # sequence -> why this peptide could not be scored

    @staticmethod
    def _out(score, label, app, detail):
        return {"score": float(np.clip(score, 0.0, 1.0)), "label": label, "applicability": app, "detail": detail}

    @classmethod
    def _out_p(cls, output, score, label, app, detail):
        """`output` is the model's own value for the outcome it predicts (IgE binding, cytotoxicity, solubility, ...) and is what the reports show.
        `score` is the same value oriented so that higher is the favourable outcome (1 - output for IgE binding, cytotoxicity, aggregation and bitterness); only the ranking uses it."""
        return {**cls._out(score, label, app, detail), "output": float(output)}

    def _try(self, s, fn):
        try:
            return fn()
        except Exception as e:  # e.g. propy divides by zero on a poly-residue peptide
            self.skipped[s] = f"{type(e).__name__}: {e}"
            return None


class Allergenicity(_Base):
    """Platform allergenicity_predictor_v1 (IEDB IgE-binding MLP + CNN; positive class = IgE binding). `ige_binding_score` = p_allergen is the model's own output and is what the reports show;
    `score` = 1 - p_allergen is only the value the ranking maximises (Phase 1 used 1 - the hold-out percentile rank instead)."""

    def __init__(self, feature_extractor):
        super().__init__(feature_extractor)
        from model_store.allergenicity_predictor_v1 import AllergenicityPredictor
        from model_store.allergenicity_predictor_v1 import predictor as mod

        self.mod, self.pred = mod, AllergenicityPredictor()

    def predict(self, seqs):
        ok = [s for s in seqs if self.mod.TRAINED_MIN_LENGTH <= len(s) <= self.mod.MAX_LENGTH]
        out = {}
        if ok:
            for s, r in zip(ok, self.pred.predict_batch(ok, self.fe)):
                p = r["p_allergen"]
                app = "in_domain" if 7 <= len(s) <= 20 else "limited"  # Phase 1: 7-20 aa is the 5th-95th percentile of training lengths
                out[s] = self._out_p(p, 1 - p, "predicted_IgE_binding" if p >= 0.5 else "predicted_non_binding", app,
                                   f"p_allergen={p:.2f} (IgE binding only, not a food-safety clearance)")
        return out


class Cytotoxicity(_Base):
    """4-fold BiLSTM/attention ensemble over ESM2-150M residue embeddings, trained on 322 balanced peptides; cell type DRAMP_aggregate."""

    def __init__(self, feature_extractor):
        super().__init__(feature_extractor)
        from model_store.cytotoxicity_predictor_v1 import predictor as mod

        self.mod, self.clf = mod, mod.CytotoxicityClassifier()

    def predict(self, seqs):
        out = {}
        ok = [s for s in seqs if len(s) <= self.mod.MAX_LEN]
        self.fe.get_esm2_embedding_batch(ok)  # batched, memoized
        for s in ok:
            p = self._try(s, lambda: self.clf.predict_cytotoxicity(s, self.fe, cell_type=self.mod.DEFAULT_CELL_TYPE))
            if p is None:
                continue
            app = "in_domain" if 5 <= len(s) <= self.mod.MAX_LEN else "limited"
            out[s] = self._out_p(p, 1 - p, "predicted_cytotoxic" if p >= 0.5 else "predicted_not_cytotoxic", app,
                               f"P(cytotoxic)={p:.2f} (mammalian cells, cell type {self.mod.DEFAULT_CELL_TYPE}; 4-fold ensemble, trained on 322 balanced peptides, fold AUROC 0.73-0.83)")
        return out


class Solubility(_Base):
    """XGBoost over the ESM2-150M mean embedding plus solvent descriptors, scored for ultrapure water."""

    def __init__(self, feature_extractor):
        super().__init__(feature_extractor)
        from model_store.solubility_predictor_v1 import predictor as mod

        self.pred = mod.SolubilityPredictor()

    def predict(self, seqs):
        out = {}
        self.fe.get_esm2_embedding_batch(list(seqs))
        for s in seqs:
            p = self._try(s, lambda: self.pred.predict_proba(s, SOLVENT, self.fe))
            if p is None:
                continue
            out[s] = self._out_p(p, p, "predicted_soluble" if p >= 0.5 else "predicted_insoluble", "limited",
                               f"P(soluble in {SOLVENT.lower()})={p:.2f} (XGBoost on ESM2-150M; pair-stratified split, so it may reward memorised peptides; "
                               "leave-one-solvent-out macro AUC 0.77; training length range not documented)")
        return out


class Aggregation(_Base):
    """XGBoost over biopython / propy3 / AAindex descriptors. Needs >= 5 aa and the optional propy3 package."""

    def __init__(self, feature_extractor):
        super().__init__(feature_extractor)
        import pipeline.feature_extractor as fe_mod

        if not fe_mod.PROPY_AVAILABLE:
            raise RuntimeError(fe_mod.PROPY_UNAVAILABLE_REASON)
        from model_store.aggregation_predictor_v1 import predictor as mod

        self.pred = mod.AggregationPredictor()

    def predict(self, seqs):
        out = {}
        for s in seqs:
            if len(s) < 5:
                continue  # model requirement; reported as not run for this peptide
            p = self._try(s, lambda: self.pred.predict_aggregation(s, self.fe))
            if p is None:
                continue
            out[s] = self._out_p(p, 1 - p, "predicted_aggregation_prone" if p >= 0.5 else "predicted_not_aggregation_prone", "limited",
                               f"P(aggregation-prone)={p:.2f} (descriptor XGBoost, seed_bh label; test AUC 0.70, MCC 0.28)")
        return out


class SynthesisFeasibility(_Base):
    """Soft-voting LogReg + SVM + RandomForest over ESM2-150M mean embeddings."""

    def __init__(self, feature_extractor):
        super().__init__(feature_extractor)
        from model_store.synthesis_feasibility_predictor_v1 import predictor as mod

        self.pred = mod.SynthesisFeasibilityEnsemble()
        self.pred._load()
        for m in self.pred.models:  # pickled with an older scikit-learn that stored `multi_class`; the installed one no longer does
            est = m.steps[-1][1] if hasattr(m, "steps") else m
            if type(est).__name__ == "LogisticRegression" and not hasattr(est, "multi_class"):
                est.multi_class = "auto"  # in memory only; the weights file is not touched

    def predict(self, seqs):
        out = {}
        self.fe.get_esm2_embedding_batch(list(seqs))
        for s in seqs:
            p = self._try(s, lambda: self.pred.predict_proba(s, self.fe))
            if p is None:
                continue
            out[s] = self._out_p(p, p, "predicted_feasible" if p >= 0.5 else "predicted_difficult", "limited",
                               f"P(synthesis feasible)={p:.2f} (LogReg+SVM+RF soft vote; 5-fold CV AUC 0.74-0.78, MCC 0.32-0.42)")
        return out


class Bitterness(_Base):
    """Platform bitterness_predictor_v1. Score = 1 - P(bitter); the label is predicted_bitter at 0.5."""

    def __init__(self, feature_extractor):
        super().__init__(feature_extractor)
        from model_store.bitterness_predictor_v1 import predictor as mod

        self.mod, self.pred = mod, mod.BitternessPredictor()

    def predict(self, seqs):
        out = {}
        for s, r in zip(seqs, self.pred.predict_batch(list(seqs), self.fe)):
            p = r["p_bitter"]
            out[s] = self._out_p(p, 1 - p, "predicted_bitter" if p >= 0.5 else "predicted_not_bitter", self.mod.applicability(len(s)),
                               f"P(bitter)={p:.2f}; if bitter, detection threshold ~{r['threshold_mM']:.2f} mM (lower = more bitter)")
        return out


#: (name, objective, description, adapter class), in Phase 1 slot order.
SLOTS = [
    ("allergenicity", "safety", "allergen likelihood of the peptide", Allergenicity),
    ("cytotoxicity", "safety", "cytotoxicity / hemolysis risk", Cytotoxicity),
    ("solubility", "developability", "aqueous solubility", Solubility),
    ("aggregation", "developability", "aggregation propensity", Aggregation),
    ("synthesis_feasibility", "developability", "synthesis difficulty", SynthesisFeasibility),
    ("bitterness", "developability", "bitterness risk", Bitterness),
]


def build_model_slots(feature_extractor) -> list[ScreeningModel]:
    """Every slot, loaded; one whose weights or dependency are missing stays "error" and is reported as not run."""
    slots = []
    for name, objective, description, cls in SLOTS:
        m = ScreeningModel(name, {"objective": objective, "description": description})
        m.load_with(lambda cls=cls: cls(feature_extractor))
        slots.append(m)
    return slots
