"""The receptor column order, now CONFIRMED, and the unit that goes with it.

`all_sequences_aligned.csv` carries three unnamed EC50 columns, EC50_T1..EC50_T3, and both
metabolic models were trained taking them to be GCGR, GLP1R, GIPR in that order. For a long time
the model cards recorded that as an assumption nobody had checked, because the file was not in
this repository.

**It was checked on 2026-11-03, against the file itself, and it is right.** The CSV contains the
natural ligands, whose receptor selectivity is not in question, and each one is most potent in
exactly the expected column by 7-8 orders of magnitude:

| ligand | T1 (pM) | T2 (pM) | T3 (pM) | most potent |
|---|---|---|---|---|
| glucagon | **0.5** | 43.9 | 1e8 | T1 -> GCGR |
| GLP-1 | 1e8 | **0.9** | 1e8 | T2 -> GLP1R |
| exendin-4 | 1e8 | **0.4** | 2e5 | T2 -> GLP1R |
| liraglutide | 1e8 | **6.4** | 1e8 | T2 -> GLP1R |
| semaglutide | 1e8 | **6.2** | 1e8 | T2 -> GLP1R |
| GIP | 1e8 | 2e5 | **0.4** | T3 -> GIPR |

The file is NOT in the repository and must not be (it is the models' training labels, and
`.gitignore` names it), so this file cannot re-derive the table. What it can do is keep both
cards from drifting apart or quietly losing the confirmation -- which matters because a wrong
order has no symptom: the numbers stay in range, the gates fire, the ranking is stable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

#: Both models trained from the same three columns of the same file.
CARDS = (
    "model_store/functional_potency_predictor_v1/model_card.json",
    "model_store/agonism_classifier_v1/model_card.json",
)

CONFIRMED_ORDER = ("GCGR", "GLP1R", "GIPR")

SOURCE_FILE = "all_sequences_aligned.csv"


def training_data(card: str) -> dict:
    return json.loads((ROOT / card).read_text(encoding="utf-8"))["training_data"]


@pytest.mark.parametrize("card", CARDS)
def test_the_card_still_names_the_source_file(card: str) -> None:
    """If the source changes, the mapping has to be re-derived from the new one."""
    assert SOURCE_FILE in training_data(card)["sources"]


@pytest.mark.parametrize("card", CARDS)
def test_the_card_records_the_confirmed_column_order(card: str) -> None:
    assumption = training_data(card)["receptor_column_assumption"]
    for position, receptor in enumerate(CONFIRMED_ORDER, start=1):
        assert f"EC50_T{position} = {receptor}" in assumption, (
            f"{card} no longer records EC50_T{position} as {receptor}. This was confirmed against"
            f" {SOURCE_FILE}; changing it means the file changed, and then BOTH cards and the"
            " ligand table in this module's docstring have to change together."
        )


@pytest.mark.parametrize("card", CARDS)
def test_the_card_says_the_order_was_confirmed_and_against_what(card: str) -> None:
    """Not just the order -- the fact that someone checked, and what they checked against.

    The previous version of this test asserted the opposite: that the card still said "not
    confirmed". That was right while it was unverified, and keeping it would now be a test
    demanding that the project forget what it learned.
    """
    assumption = training_data(card)["receptor_column_assumption"]
    assert "CONFIRMED" in assumption
    assert SOURCE_FILE in assumption
    # The evidence is the ligands, so the card has to name at least one of them.
    assert any(ligand in assumption for ligand in ("glucagon", "GLP-1", "GIP"))


@pytest.mark.parametrize("card", CARDS)
def test_the_card_records_that_the_source_column_is_picomolar(card: str) -> None:
    """The second thing the file settled, which neither card used to state.

    EC50_LOG_T* = log10(EC50_T* x 1e-12) to a mean absolute error of 0.0013 over all 496 paired
    values; assuming nanomolar is out by exactly 3.0000. Anyone retraining from the raw column
    rather than the log column needs this, and a 1000x error in potency would pass every check
    the pipeline makes.
    """
    units = training_data(card)["ec50_units_in_source"]
    assert "PICOMOLAR" in units
    assert "1e-12" in units


def test_gipr_is_the_receptor_the_order_matters_least_for() -> None:
    """Context worth having beside the mapping: GIPR's own metrics say its column could be
    anything and the model would score the same. GCGR and GLP1R are the two with signal, so a
    swap between THOSE two is the one that would have quietly corrupted a shortlist."""
    metrics = json.loads((ROOT / CARDS[0]).read_text(encoding="utf-8"))[
        "loco_cv_metrics_ensemble"
    ]
    assert metrics["GIPR"]["R2"] < 0
    assert metrics["GCGR"]["R2"] > 0.4 and metrics["GLP1R"]["R2"] > 0.4
