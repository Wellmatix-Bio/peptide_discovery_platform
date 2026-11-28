"""PDF of the review dossiers, written for the biology lead in plain words: an overview page of the shortlisted peptides, a page with what is the same for every peptide, a page on how the list was made (when the run recorded it), then one A4 landscape page per peptide.

Two parts. First the words: plain labels and explanations of the dossier's own values, as blocks per column (nothing is added or estimated; every explanation restates what the pipeline code does). Then the drawing of those blocks with reportlab: the cells the biology lead fills in stay empty, and a page that holds more than fits is set in smaller type, down to `MIN_SCALE`. Technical detail (random seed, generator version, database checksum) stays in the JSON dossier.

Blocks: ("title", text) ("text", text) ("small", text) ("pairs", [(label, value), ...]) ("bullets", [text, ...]) ("table", header, rows, width_fractions) ("mono", [line, ...]) ("blank", [label, ...])
"""
from __future__ import annotations

import io
import re
from xml.sax.saxutils import escape

from food.a_proteins.fasta import TERMINUS_MEANING
from food.shared.dossier import MODEL_OUTPUT_MEANING, FoodDossier, model_outputs


def terminus_label(code: str | None) -> str:
    """One end of a parent protein chain in plain words; 'not recorded' when the input did not say."""
    return sentence(TERMINUS_MEANING.get(code, words(code))) if code else "Not recorded"

MAX_PARENT_ROWS = 6
MAX_ROUTE_ROWS = 4

#: The aggregation model skips a peptide shorter than this ("model requirement" in `food/c_discovery/screening/adapters.py`).
AGGREGATION_MIN_LENGTH = 5

ORIGIN = {
    "DIGESTION_DERIVED": "Released from a food protein by simulated digestion",
    "SYNTHETIC_DESIGN": "Designed variant: an edited copy of a digestion fragment, not found in a food protein as written",
}
EVIDENCE_CLASS = {"THEORETICAL": "Predicted only; nothing here was measured"}
METHOD = {"in_silico_digestion": "Simulated digestion", "reference_guided_edit": "Edited copy of a digestion fragment, steered toward a reference peptide"}
ENZYME = {"pepsin_ph1.3": "pepsin (pH 1.3 rule)", "pepsin_ph2.0": "pepsin (pH 2.0 rule)", "trypsin": "trypsin",
          "chymotrypsin_high": "chymotrypsin (high-specificity rule)", "chymotrypsin_low": "chymotrypsin (low-specificity rule)"}
ENZYME_SHORT = {"pepsin_ph1.3": "pepsin", "pepsin_ph2.0": "pepsin", "trypsin": "trypsin", "chymotrypsin_high": "chymotrypsin", "chymotrypsin_low": "chymotrypsin"}
STAGE = {"gastric": "stomach", "intestinal": "intestine"}
APPLICABILITY = {"in_domain": "Within range", "limited": "Limited", "limited_evidence": "Limited", "out_of_domain": "Outside range", "not_run": "Not run"}
MODEL = {"allergenicity": "IgE binding", "cytotoxicity": "Cytotoxicity", "solubility": "Solubility in water", "aggregation": "Aggregation",
         "synthesis_feasibility": "Synthesis feasibility", "bitterness": "Bitterness"}
CHECK = {**MODEL, "evidence": "Evidence", "stability": "Cut prediction"}
ASSAY = {"glp1_endpoint": "GLP-1 endpoint (active or total)", "concentration_range": "Concentration range", "exposure_time": "Exposure time", "cell_line": "Cell line",
         "vehicle": "Vehicle", "viability_control": "Viability control", "interference_control": "Interference control",
         "biological_replicates": "Biological replicates", "technical_replicates": "Technical replicates"}
ROLE = {"positive_control": "Positive control", "negative_control": "Negative control"}
EVIDENCE_TYPE = {"EXACT_SEQUENCE": "Exact sequence", "CONTAINS_REFERENCE": "Contains the evidence peptide", "CONTAINED_IN_REFERENCE": "Contained in the evidence peptide",
                 "SEQUENCE_SIMILARITY": "Sequence similarity", "NO_EVIDENCE": "No evidence"}
EVIDENCE_TYPE_SHORT = {"EXACT_SEQUENCE": "Exact sequence", "CONTAINS_REFERENCE": "Contains reference", "CONTAINED_IN_REFERENCE": "Contained in reference",
                       "SEQUENCE_SIMILARITY": "Sequence similarity", "NO_EVIDENCE": "No evidence"}
REVIEW = {"approve": "Approved", "hold": "On hold", "reject": "Rejected"}
#: Terms in the generated texts that mean nothing to a reader of the PDF.
PHRASES = {"SYNTHETIC_DESIGN": "designed peptide", "*_flagged_models": "the models left out of the score", "digestion_plausibility": "digestion plausibility",
           "developability_rule": "rule-based developability"}


def words(value) -> str:
    return str(value).replace("_", " ")


def sentence(text: str) -> str:
    return text[:1].upper() + text[1:]


def yes_no(value) -> str:
    return "Not recorded" if value is None else ("Yes" if value else "No")


def number(value, digits: int = 3):
    return f"{value:.{digits}f}".rstrip("0").rstrip(".") if isinstance(value, float) else value


def plain_text(text: str) -> str:
    """A generated sentence with the code-style terms replaced; 'model_<name>_score' reads '<name> score'."""
    for old, new in PHRASES.items():
        text = text.replace(old, new)
    out = []
    for token in text.split(" "):
        core = token.strip(";,.()")
        if core.startswith("model_") and core.endswith("_score"):
            token = token.replace(core, words(core[len("model_"):-len("_score")]) + " score")
        out.append(token)
    return " ".join(out)


def enzymes_text(names) -> str:
    return " + ".join(ENZYME_SHORT.get(n, words(n)) for n in names)


def pathway_text(pathway_id: str) -> str:
    """The digestion pathway in words, from the fixed digestion settings; an unknown id is shown with its underscores removed."""
    from food.b_digestion.settings import DEFAULT_SETTINGS

    for pathway in DEFAULT_SETTINGS.pathways:
        if pathway.id == pathway_id:
            steps = []
            for stage in pathway.stages:
                ph = f", pH {number(float(stage.ph))}" if stage.ph is not None else ""
                steps.append(f"{STAGE.get(stage.name, stage.name)} ({enzymes_text(stage.enzymes)}{ph})")
            return " then ".join(steps)
    return words(pathway_id)


def cluster_text(key: str) -> str:
    if key.startswith("family:"):
        return "protein family " + key[len("family:"):].strip()
    if key.startswith("protein:"):
        return "protein " + key[len("protein:"):]
    return words(key)


def edit_text(edits: str) -> str:
    """'A1>S;K2>del' reads 'A1 to S; K2 deleted': the residue, its position in the template, and what replaced it."""
    parts = []
    for token in edits.split(";"):
        old_pos, _, new = token.partition(">")
        parts.append(f"{old_pos} deleted" if new == "del" else f"{old_pos} to {new}")
    return "; ".join(parts)


def position_text(parent: dict) -> str:
    start = parent.get("start_in_precursor") if parent.get("start_in_precursor") is not None else parent.get("start_in_product")
    end = parent.get("end_in_precursor") if parent.get("end_in_precursor") is not None else parent.get("end_in_product")
    return f"{start}-{end}"


def cut_text(sequence: str, positions: list[int]) -> str:
    """'after Y4' means the bond after the tyrosine at position 4 is cut."""
    return ", ".join(f"after {sequence[p - 1]}{p}" for p in positions) if positions else "none"


def keep(pairs):
    return [(label, value) for label, value in pairs if value not in (None, "", [], {})]


def properties_block(d: FoodDossier) -> list:
    p = d.properties
    parts = [f"{p.get('length')} amino acids", f"net charge at pH 7.4: {number(p.get('net_charge_pH7_4'))}", f"isoelectric point: {number(p.get('isoelectric_point'))}",
             f"hydropathy (GRAVY): {number(p.get('gravy'))}"]
    return [("title", "Peptide properties"), ("text", "  |  ".join(x for x, v in zip(parts, [p.get("length"), p.get("net_charge_pH7_4"), p.get("isoelectric_point"), p.get("gravy")]) if v is not None))]


def origin_blocks(d: FoodDossier) -> list:
    prov, designed = d.provenance, d.origin_type == "SYNTHETIC_DESIGN"
    merged: dict[tuple, dict] = {}
    for p in prov.get("parents", []):
        key = (p.get("protein_id"), p.get("start_in_precursor"), p.get("end_in_precursor"))
        row = merged.setdefault(key, {"parent": p, "pathways": [], "final": False})
        if p.get("pathway") and pathway_text(p["pathway"]) not in row["pathways"]:
            row["pathways"].append(pathway_text(p["pathway"]))
        row["final"] = row["final"] or bool(p.get("is_limit_product"))
    rows = []
    for row in list(merged.values())[:MAX_PARENT_ROWS]:
        p = row["parent"]
        cells = [f"{p.get('accession') or p.get('protein_id')}" + (f" ({p['food_source']})" if p.get("food_source") else ""), yes_no(p.get("allergen_source")),
                 position_text(p), terminus_label(p.get("protein_n_terminus")), terminus_label(p.get("protein_c_terminus")), "; ".join(row["pathways"]) or "not recorded"]
        rows.append(cells if designed else [*cells, "Final" if row["final"] else "Intermediate"])
    header = ["Food protein", "Parent is an allergen source", "Residues in the UniProt entry", "N-terminus of the protein chain", "C-terminus of the protein chain",
              "Digestion routes that release it (template)" if designed else "Digestion routes that release it"]
    widths = [0.14, 0.08, 0.1, 0.2, 0.2, 0.28] if designed else [0.12, 0.07, 0.09, 0.18, 0.18, 0.24, 0.12]
    blocks = [("title", "Where it comes from"),
              ("small", "This peptide was designed: it does not occur in a food protein as written. The table shows the protein its template fragment comes from." if designed else
               f"{d.digestion.get('statement', '')} {d.digestion.get('product_type_meaning', '')}".strip())]
    blocks.append(("table", header if designed else [*header, "Digestion product"], rows, widths) if rows else ("small", "No parent protein recorded."))
    if len(merged) > MAX_PARENT_ROWS:
        blocks.append(("small", f"{len(merged) - MAX_PARENT_ROWS} more parent site(s) are not listed here; the JSON dossier lists all of them."))
    if designed and prov.get("design"):
        g = prov["design"]
        template = f"{g.get('template_sequence')} ({edit_text(g['edits'])})" if g.get("edits") else g.get("template_sequence")
        guide = f"{g.get('guide_peptide')} (study {g.get('guide_study')}; similarity {number(g.get('sim_to_guide'))})" if g.get("guide_peptide") else None
        blocks += [("title", "How it was designed"), ("pairs", keep([("Template fragment and its edits (residue, position, change)", template), ("Reference peptide it was steered toward", guide)]))]
    return blocks


def comparison_blocks(d: FoodDossier) -> list:
    """The parent proteins, the hydrolysates they define and the reference peptides to compare with; the run's controls are on the shared page."""
    cm = d.comparison_materials
    if not cm:
        return []
    accession = {p.get("protein_id"): p.get("accession") or p.get("protein_id") for p in cm.get("parent_proteins", [])}
    parents = [f"{p.get('accession') or p.get('protein_id')} ({p.get('food_source')})" if p.get("food_source") else str(p.get("accession") or p.get("protein_id"))
               for p in cm.get("parent_proteins", [])]
    hydrolysates = [f"{accession.get(h.get('parent_protein'), h.get('parent_protein'))}: {pathway_text(h['pathway'])}" for h in cm.get("hydrolysates", [])]
    def role_text(role: str) -> str:
        if role.startswith("closest evidence peptide (") and role.endswith(")"):
            kind = role[len("closest evidence peptide ("):-1]
            return f"Closest evidence peptide ({EVIDENCE_TYPE.get(kind, words(kind)).lower()})"
        return sentence(role)

    refs = [[role_text(r.get("role", "")), r.get("sequence"), r.get("source") if r.get("source") is not None else (number(r.get("similarity")) if r.get("similarity") is not None else "")]
            for r in cm.get("reference_peptides", []) if not str(r.get("role", "")).endswith("control")]
    short = lambda items, n: "; ".join(items[:n]) + (f" and {len(items) - n} more" if len(items) > n else "")  # noqa: E731
    blocks = [("title", "Compare with"),
              ("pairs", keep([("Parent proteins", short(parents, 3)), ("Parent hydrolysates to prepare (parent protein and digestion route; not existing records)", short(hydrolysates, 2))]))]
    if refs:
        blocks.append(("table", ["Reference peptide", "Sequence", "Source or similarity"], refs, [0.5, 0.25, 0.25]))
    return blocks


def evidence_blocks(d: FoodDossier) -> list:
    ev = d.evidence
    gate = ev.get("gate", {})
    sentence_gate = ("Evidence gate: passed (an exact sequence match to an evidence peptide with a positive or mixed reported effect)." if gate.get("passed") else
                     "Evidence gate: not passed (it needs an exact sequence match to an evidence peptide with a positive or mixed reported effect).")

    def describe(block: dict) -> str:
        if block["evidence_type"] == "NO_EVIDENCE":
            return "No evidence peptide qualifies (no data, not inactive)."
        return (f"{block.get('evidence_peptide')} (similarity {number(block.get('similarity'))}; reported effect "
                f"{words(block.get('reported_effect') or 'not recorded')})")

    def source(block: dict) -> str | None:
        src = block.get("source") or {}
        parts = [x for x in (src.get("study"), src.get("assay"), f"food origin: {words(src['food_derivation'])}" if src.get("food_derivation") else None) if x]
        return "; ".join(parts) or None

    out = [("title", "Evidence"), ("text", sentence_gate)]
    rows = []
    if ev.get("candidate"):
        c = ev["candidate"]
        rows += [(f"This peptide: {EVIDENCE_TYPE.get(c['evidence_type'], words(c['evidence_type'])).lower()}", describe(c)), ("Source of that evidence", source(c))]
    if ev.get("template"):
        t = ev["template"]
        rows += [(f"Its template (shown, not transferred): {EVIDENCE_TYPE.get(t['evidence_type'], words(t['evidence_type'])).lower()}",
                  describe(t) + (f"; {source(t)}" if source(t) else ""))]
    rows.append(("Prior-art note", ev.get("prior_art_alert")))
    out += [("pairs", keep(rows)), ("small", "An evidence type is a sequence relationship, not proof of GLP-1 activity.")]
    return out


def gut_blocks(d: FoodDossier) -> list:
    g = d.gut_stability
    positions = g.get("predicted_cleavage_positions") or {}
    counts = g.get("cut_sites_per_protease") or {}
    names = list(positions) or list(counts)
    cuts = " | ".join(f"{words(n)}: " + (cut_text(d.sequence, positions.get(n, [])) if positions else f"{counts.get(n)} site(s)") for n in names)
    pairs = keep([("Uncut-bond fraction (0-1; the share of its internal bonds that no enzyme rule cuts)", number(g.get("uncut_bond_fraction"))),
                  ("Predicted cuts by gut enzymes (the bond after the residue shown, with its position)", cuts)])
    return [("title", "Predicted cuts by gut enzymes (theoretical)"), ("pairs", pairs), ("small", f"{g.get('statement') or ''} {g.get('note') or ''}".strip().replace("uncut_bond_fraction is", "The uncut-bond fraction is"))]


def safety_blocks(d: FoodDossier) -> list:
    s = d.safety_screening
    if not s:
        return []
    ige = s.get("ige_binding", {})
    if ige.get("status") == "RUN":
        ige_text = (f"{sentence(words(ige.get('label', '')))}; IgE-binding score {number(ige.get('ige_binding_score'), 2)} (0 to 1; higher means more likely to bind IgE; "
                    f"0.5 is the model's decision boundary); applicability: {APPLICABILITY.get(ige.get('applicability'), words(ige.get('applicability', '')))}")
    else:
        ige_text = "Not run. Its absence is not a pass."
    cyt = (s.get("other_safety_models") or {}).get("cytotoxicity")
    cyt_text = (f"{sentence(words(cyt.get('label', '')))}; cytotoxicity output {number(cyt.get('output'), 2)} (higher means more likely to be cytotoxic); "
                f"applicability: {APPLICABILITY.get(cyt.get('applicability'), words(cyt.get('applicability', '')))}" if cyt else "Not run. Its absence is not a pass.")
    names = lambda items: ", ".join(MODEL.get(n.split(":")[0], words(n)) for n in items) or "none"  # noqa: E731
    pairs = keep([("Predicted IgE binding (its own risk)", ige_text),
                  ("Parent protein an allergen source", {"allergen_source": "Yes", "no_flag": "No", "not_available": "Not recorded"}.get(s.get("parent_allergen_flag"), None)),
                  ("Cytotoxicity", cyt_text),
                  ("Models not run | left out of the safety score (outside their range)", f"{names(s.get('predictions_not_run', []))} | {names(s.get('models_left_out_of_the_safety_score', []))}")])
    return [("title", "Safety screening"), ("pairs", pairs), ("small", s.get("rule", ""))]


def screening_blocks(d: FoodDossier) -> list:
    s = d.screening
    outputs = s.get("model_outputs") or model_outputs(s.get("models", {}))
    rows = [[MODEL.get(n, words(n)), sentence(words(o.get("label", ""))), number(o["output"], 2), APPLICABILITY.get(o.get("applicability"), words(o.get("applicability", ""))),
             sentence(o["higher_means"])] for n, o in outputs.items()]
    for name in s.get("predictions_not_run", []):
        if name in MODEL:
            reason = (f"The aggregation model needs at least {AGGREGATION_MIN_LENGTH} amino acids; this peptide has {d.length}." if name == "aggregation" and d.length < AGGREGATION_MIN_LENGTH
                      else "The model did not run for this peptide.")
            rows.append([MODEL[name], "Not run", "-", "Not run", reason])
    pairs = keep([("Overall applicability (the worst of all checks)", APPLICABILITY.get(s.get("applicability_overall"), words(s.get("applicability_overall", "")))),
                  ("GLP-1 activity classifier", (str(s.get("glp1_classifier_status", "")).lower() + "; it is not used for ranking") if s.get("glp1_classifier_status") else None)])
    return [("title", "Screening models"),
            ("small", "Each value is the model's own output for what it predicts, from 0 to 1; the last column says what a higher value means. The models predict different things. "
                      "Applicability is how well the peptide fits what each model was trained on: read it before the values."),
            ("table", ["Model", "Result", "Model output (0-1)", "Applicability", "A higher value means"], rows, [0.17, 0.24, 0.12, 0.14, 0.33]) if rows else ("small", "No model recorded."),
            ("pairs", pairs)]


def split_outside_brackets(text: str) -> list[str]:
    """Split on '; ' where it is not inside brackets."""
    parts, depth, start = [], 0, 0
    for i, ch in enumerate(text):
        depth += (ch == "(") - (ch == ")")
        if ch == ";" and depth == 0 and text[i + 1:i + 2] == " ":
            parts.append(text[start:i])
            start = i + 2
    parts.append(text[start:])
    return [p.strip() for p in parts if p.strip()]


#: The ranking compares each model as a value where higher is favourable (1 minus the output for the first four). In words about what the model predicts, a good value is the
#: first phrase and a poor one the second.
RANKING_TERMS = {"model_allergenicity_score": ("low predicted IgE binding", "high predicted IgE binding"), "model_cytotoxicity_score": ("low predicted cytotoxicity", "high predicted cytotoxicity"),
                 "model_aggregation_score": ("low predicted aggregation", "high predicted aggregation"), "model_bitterness_score": ("low predicted bitterness", "high predicted bitterness"),
                 "model_solubility_score": ("high predicted solubility", "low predicted solubility"),
                 "model_synthesis_feasibility_score": ("high predicted synthesis feasibility", "low predicted synthesis feasibility")}


def reasons(summary: str) -> str:
    """The strengths and weaknesses of the ranking rationale, in words about what each model predicts (never the ranking's turned-around value). Its Pareto and cluster numbers are
    technical, and what it says about evidence, applicability, prior art and liabilities is shown in its own section of the page."""
    out = []
    for part in split_outside_brackets(summary or ""):
        for prefix, label, index in (("near pool-best on ", "Among the best of the pool on", 0), ("weak on ", "Weak on", 1)):
            if part.startswith(prefix):
                items = [RANKING_TERMS[i][index] if i in RANKING_TERMS else plain_text(i) for i in part[len(prefix):].split(", ")]
                out.append(f"{label}: {', '.join(items)}.")
    return " ".join(out)


def limits_of(d: FoodDossier) -> list[str]:
    return [sentence(plain_text(s)) for s in split_outside_brackets(d.limitations or "") if s.lower() != "models not run: none"]


def shared_limits(dossiers: list[FoodDossier]) -> list[str]:
    """The limitations that every peptide of the PDF carries, in the first peptide's order; they go on the shared page once."""
    if not dossiers:
        return []
    common = set(limits_of(dossiers[0]))
    for d in dossiers[1:]:
        common &= set(limits_of(d))
    return [s for s in limits_of(dossiers[0]) if s in common]


def review_blocks(d: FoodDossier) -> list:
    r = d.review
    if r.get("decision"):
        return [("title", "Review"), ("pairs", keep([("Decision", REVIEW.get(r["decision"], r["decision"])), ("Reason", r.get("reason") or "none given"), ("Reviewer", r.get("reviewer")),
                                                     ("Decided at", r.get("decided_at"))]))]
    return [("title", "Review"), ("blank", ["Decision (approve / hold / reject)", "Reason", "Reviewer"])]


def plain_dossier(d: FoodDossier, shared: list[str] | None = None) -> dict:
    """The page of one peptide: only what is specific to it, plus its review. `shared` are the limitations already on the shared page."""
    limits = [s for s in limits_of(d) if s not in (shared or [])]
    position = f"Shortlist position {d.shortlist_position}" if d.shortlist_position is not None else "Not on the shortlist"
    top = [d.sequence, f"{d.peptide_id}  |  {d.length} amino acids", sentence(d.status) + ".",
           f"{position}  |  Rank {d.rank}  |  {ORIGIN.get(d.origin_type, words(d.origin_type))}  |  {EVIDENCE_CLASS.get(d.evidence_class, words(d.evidence_class))}"]
    ranking, compare = ranking_blocks(d), comparison_blocks(d)
    head = [("title", "Why it ranks here"), ("text", reasons(d.summary) or "No standout strength or weakness among the ranking objectives.")]
    left_body = [*properties_block(d), *origin_blocks(d), *digestion_blocks(d), *safety_blocks(d)]
    right_body = [*screening_blocks(d), *evidence_blocks(d), *alignment_blocks(d), *chemistry_blocks(d), *gut_blocks(d)]
    tail = ([("title", "Limitations specific to this peptide"), ("bullets", limits)] if limits else []) + review_blocks(d)
    # the ranking and the comparison blocks go to whichever column leaves the page the larger type (food_dossier_pdf.fit tries each)
    alternatives = [(head + left_body, [*ranking, *right_body, *compare, *tail]), (head + ranking + left_body, [*right_body, *compare, *tail]),
                    (head + left_body + compare, [*ranking, *right_body, *tail]), (head + ranking + left_body + compare, [*right_body, *tail])]
    left, right = alternatives[0]
    return {"top": top, "left": left, "right": right, "alternatives": alternatives}


def review_label(d: FoodDossier) -> str:
    return REVIEW.get(d.review.get("decision"), "Pending")


def cytotoxicity_output(d: FoodDossier) -> str:
    cyt = (d.safety_screening or {}).get("other_safety_models", {}).get("cytotoxicity")
    return str(number(cyt["output"], 2)) if cyt else "n/a"


def ige_score(d: FoodDossier) -> str:
    ige = (d.safety_screening or {}).get("ige_binding", {})
    return str(number(ige["ige_binding_score"], 2)) if ige.get("status") == "RUN" and ige.get("ige_binding_score") is not None else "n/a"


def ige_label(d: FoodDossier) -> str:
    ige = (d.safety_screening or {}).get("ige_binding", {})
    return sentence(words(ige.get("label", ""))) if ige.get("status") == "RUN" else "Not run"


RANK_VS_SHORTLIST = [
    "Rank is a peptide's place among the 25 ranked peptides (1 is first). It is set by the computed ranking and never changes, whatever the biology lead decides.",
    "Shortlist number is a peptide's place on the shortlist of at most 10, which is taken from the 25 in rank order. It is not the rank: shortlist number 1 can be rank 4. "
    "A peptide is left off when its evidence is only a similarity or none, when a model itself calls it cytotoxic, aggregation-prone, insoluble or hard to synthesise, when a higher-ranked peptide "
    "of the same parent protein family is already on the list, or when 4 designed peptides are already on it. The reason for each peptide is in the results.",
    "Why a peptide is among the 25: the 400 best peptides of the whole pool go to the models. The 25 are then picked front by front (a front is a group of peptides that no other peptide beats on "
    "every score), at most 2 from one sequence cluster and 3 from one parent protein family, and 15 with an exact sequence match to an evidence peptide are asked for first. If fewer exist, "
    "the shortfall is reported. A peptide outside the 25 is not listed, and no reason is stored for it.",
]


OBJECTIVE_NAME = {"digestion_plausibility": "digestion plausibility (the digestion rules and the uncut-bond fraction)",
                  "coverage": "coverage (how fully the peptide is assessed; a check that did not run counts as zero)",
                  "developability_rule": "developability by the sequence rules (fewer liabilities scores higher)", "evidence": "evidence (from the lookup against labelled peptides)"}


OBJECTIVE_SHORT = {"digestion_plausibility": "digestion plausibility", "coverage": "coverage", "developability_rule": "developability rules", "evidence": "evidence"}


def objective_name(name: str) -> str:
    """An objective of the ranking in words. A model objective is the ranking value, so it is named by the favourable outcome (for example 'low predicted IgE binding')."""
    if name in OBJECTIVE_NAME:
        return OBJECTIVE_NAME[name]
    return RANKING_TERMS[name][0] if name in RANKING_TERMS else plain_text(name)


def enzyme_labels(cleaved_by: list[str]) -> str:
    return enzymes_text(dict.fromkeys(x.split(":")[-1] for x in cleaved_by))


def cut_label(cut) -> str:
    if isinstance(cut, str):
        return "Protein end"
    return f"{cut['p1_residue']}{cut['position_in_precursor']} | {cut['p1_prime_residue']} ({enzyme_labels(cut['cleaved_by'])})"


def digestion_blocks(d: FoodDossier) -> list:
    routes = d.digestion.get("routes") or []
    designed = d.origin_type == "SYNTHETIC_DESIGN"
    title = "Digestion detail of the template fragment (predicted)" if designed else "Digestion detail (predicted)"
    if not routes:
        return [("title", title), ("small", "Not recorded for this run.")]
    rows = [[f"{r.get('accession') or r.get('protein_id', '')} {r.get('start_in_precursor', '')}-{r.get('end_in_precursor', '')}: {pathway_text(r['pathway'])}", cut_label(r["n_side_cut"]), cut_label(r["c_side_cut"]),
             "; ".join(f"after residue {x['position_in_peptide']} ({x['p1_residue']} | {x['p1_prime_residue']}, {enzyme_labels(x['cleaved_by'])})" for x in r["cuts_inside_after_the_pathway"]) or "none",
             "Final" if r["product_type"] == "FINAL_PRODUCT" else "Intermediate"] for r in routes[:MAX_ROUTE_ROWS]]
    note = "A designed peptide has no digestion of its own: this is its template fragment." if designed else ""
    out = [("title", title), ("table", ["Protein, residues and digestion route", "Cut before it", "Cut after it", "Cuts still inside it", "Product"], rows, [0.34, 0.15, 0.15, 0.22, 0.14]), ("small", note)]
    if len(routes) > MAX_ROUTE_ROWS:
        out.append(("small", f"{len(routes) - MAX_ROUTE_ROWS} more route(s) are in the JSON dossier."))
    return out


def alignment_blocks(d: FoodDossier) -> list:
    out = []
    for label, block in (("this peptide", d.evidence.get("candidate")), ("its template, shown not transferred", d.evidence.get("template"))):
        a = (block or {}).get("alignment")
        if not a:
            continue
        out += [("title", f"Alignment with the evidence peptide ({label})"), ("mono", [a["aligned_candidate"], a["midline"], a["aligned_reference"]]),
                ("small", f"Lengths {a['candidate_length']} and {a['reference_length']}; aligned residues "
                          f"{a['candidate_start']}-{a['candidate_end']} and {a['reference_start']}-{a['reference_end']}; identity {a['identical_positions']} of {a['aligned_columns']} "
                          f"({number(a['identity'] * 100, 1)}%); coverage {number(a['candidate_coverage'] * 100, 1)}% of the peptide and {number(a['reference_coverage'] * 100, 1)}% of the "
                          f"evidence peptide; {a['relationship']}.")]
    return out


def chemistry_blocks(d: FoodDossier) -> list:
    c = d.chemistry
    if not c:
        return []
    liabilities = "; ".join(x["text"] for x in c.get("predicted_liabilities", [])) or "none found"
    mass = (f"{number(c.get('monoisotopic_mass_da'), 4)} Da monoisotopic; {number(c.get('average_mass_da'), 2)} Da average" if c.get("monoisotopic_mass_da") is not None else "Not available")
    return [("title", "Chemistry"),
            ("pairs", keep([("Mass", mass + (f" ({c['mass_source']})" if c.get("mass_source") else "")), ("Terminal state", c.get("terminal_modifications")),
                            ("Predicted liabilities (from the sequence, not measured)", liabilities), ("Measured chemical stability", c.get("measured_chemical_stability")),
                            ("Protease susceptibility", c.get("protease_susceptibility"))]))]


def ranking_blocks(d: FoodDossier) -> list:
    r = d.ranking
    if not r:
        return []
    front = f"{r['pareto_front']} of {r['n_pareto_fronts']}" if r.get("pareto_front") is not None and r.get("n_pareto_fronts") else None
    rank = f"{r['computational_rank']}" + (f" of {r['ranked_peptides']}" if r.get("ranked_peptides") else "")
    position = str(r["shortlist_position"]) if r.get("shortlist_position") is not None else "Not on the shortlist"
    pairs = keep([("Computational rank", rank), ("Pareto front (1 = no pooled peptide beats it on every objective)", front), ("Shortlist position (a separate rule)", position),
                  ("Shortlist reason", sentence(r["shortlist_reason"]) if r.get("shortlist_reason") else None)])
    out = [("title", "Ranking"), ("pairs", pairs)]
    objectives = r.get("objectives")
    if objectives:
        out.append(("small", "Objective values (the ranking values, 0 to 1, higher is favourable; each model's own output is in the screening table): " + "; ".join(f"{OBJECTIVE_SHORT.get(o['name']) or objective_name(o['name'])} {number(o['value'], 2)}" for o in objectives) + "."))
    return out


def plain_ranking(dossiers: list[FoodDossier]) -> dict:
    """The page on how the run narrowed its candidates, from the run summary every dossier of the run carries."""
    s = dossiers[0].ranking_summary
    top = ["How the list was made", "From the whole pool of candidates to the ranked peptides and the shortlist", sentence(dossiers[0].status) + ".", ""]
    fmt = lambda n: "-" if n in (None, 0) else f"{n:,}"  # noqa: E731
    rows = [[st["label"], fmt(st["n_in"]), fmt(st["n_out"]), "; ".join(f"{n:,} {why}" for why, n in st["removed"].items()) or "-"] for st in s.get("stages", [])]
    if s.get("shortlisted") is not None:
        rows.append(["Shortlist for biological review (its own rule, taken from the ranked list)", fmt(s.get("ranked")), fmt(s["shortlisted"]), "-"])
    hard, gate, caps, pareto, objs = s.get("hard_rules") or {}, s.get("evidence_gate") or {}, s.get("caps") or {}, s.get("pareto") or {}, s.get("objectives") or {}
    length = (f"shorter than {hard['min_length']} or longer than {hard['max_length']} amino acids" if hard.get("min_length") else "outside the allowed length")
    removed = "; ".join(f"{n:,} {why}" for why, n in (hard.get("removed") or {}).items()) or "none"
    left = [("title", "From the pool to the ranked list"), ("table", ["Step", "In", "Out", "Removed, and why"], rows, [0.4, 0.1, 0.1, 0.4]) if rows else ("small", "Not recorded for this run."),
            ("title", "Hard exclusions"),
            ("text", f"Before any model runs, a peptide is removed if it has a residue other than the 20 standard amino acids or is {length}. Removed in this run: {removed}. "
                     f"After that, the best {fmt(s.get('model_stage_pool_size') or s.get('pool_sent_to_the_models'))} of the pool on three cheap objectives (near-duplicates dropped) go to the models."),
            ("title", "Evidence gate"),
            ("text", f"{gate.get('rule', '')} In this run: {fmt(gate.get('exact_evidence_requested'))} asked for, {fmt(gate.get('exact_evidence_in_final'))} with an exact sequence match in the ranked list, "
                     f"shortfall {fmt(gate.get('shortfall'))}; {len(gate.get('dropped_for_the_gate') or [])} peptide(s) were dropped and replaced to meet the gate.")]
    cap_list = ", ".join((caps.get("dropped_for_the_caps") or [])[:10])
    right = [("title", "Objectives"),
             ("bullets", [sentence(objective_name(o)) for o in objs.get("used") or []] or ["Not recorded for this run."]),
             ("small", objs.get("meaning", "") + (" Not used in this run because a pooled peptide lacks them: " + ", ".join(objective_name(o) for o in objs["not_used_because_a_pooled_peptide_lacks_them"]) + "."
                                                  if objs.get("not_used_because_a_pooled_peptide_lacks_them") else "")),
             ("title", "How peptides are picked"),
             ("text", pareto.get("selection_rule", "")),
             ("small", "Tie-break order (higher first), then sequence: " + (", ".join(objective_name(o) for o in pareto["tie_break"]) if pareto.get("tie_break") else "not recorded for this run")
                       + f". Weights: {pareto.get('weights', '')} Pareto fronts in the pool: {fmt(pareto.get('n_fronts'))}; sequence clusters: {fmt(pareto.get('n_sequence_clusters'))}."),
             ("title", "Cluster and parent-family caps"),
             ("text", f"At most {caps.get('max_per_sequence_cluster') or 'a set number of'} peptides from one sequence cluster and {caps.get('max_per_parent_family') or 'a set number of'} from one parent "
                      f"protein family. {caps.get('rule', '')} In this run {len(caps.get('dropped_for_the_caps') or [])} peptide(s) were dropped for a cap" + (f": {cap_list}." if cap_list else "."))]
    return {"top": top, "left": left, "right": right}


def plain_overview(dossiers: list[FoodDossier]) -> dict:
    """The first page: the shortlisted peptides in shortlist order, one line each, and how to read the peptide pages."""
    first = dossiers[0]
    theoretical = all(d.evidence_class == "THEORETICAL" for d in dossiers)
    n = len(dossiers)
    top = ["Shortlist for biological review", f"{n} peptide{'s' if n != 1 else ''}. After the shared information page, one page per peptide.", sentence(first.status) + ".",
           EVIDENCE_CLASS["THEORETICAL"] if theoretical else ""]
    rows = [[d.shortlist_position if d.shortlist_position is not None else "-", d.rank, d.sequence, d.length, "Designed" if d.origin_type == "SYNTHETIC_DESIGN" else "Digested",
             EVIDENCE_TYPE_SHORT.get((d.evidence.get("candidate") or {}).get("evidence_type"), "-"),
             APPLICABILITY.get(d.screening.get("applicability_overall"), words(d.screening.get("applicability_overall", ""))), yes_no(d.provenance.get("any_parent_allergen_source")),
             ige_label(d), ige_score(d), cytotoxicity_output(d), number(d.gut_stability.get("uncut_bond_fraction"), 2), review_label(d)]
            for d in sorted(dossiers, key=lambda d: (d.shortlist_position is None, d.shortlist_position or 0, d.rank))]
    wide = [("title", "Overview"),
            ("table", ["Shortlist no.", "Rank", "Peptide", "Length", "Origin", "Evidence type", "Applicability", "Parent is an allergen source", "Predicted IgE binding", "IgE-binding score",
                       "Cytotoxicity output", "Uncut bonds", "Review"], rows, [0.06, 0.04, 0.13, 0.05, 0.06, 0.10, 0.07, 0.08, 0.10, 0.07, 0.08, 0.06, 0.10])]
    left = [("title", "Rank, shortlist number and the cut-off at 25"), ("bullets", RANK_VS_SHORTLIST),
            ("title", "How to read the peptide pages"),
            ("bullets", ["Every model value runs from 0 to 1 and is the model's own output for what it predicts: a higher IgE-binding value means more likely to bind IgE, a higher cytotoxicity value "
                         "means more likely to be cytotoxic, and so on. Each page says what a higher value means for each model. Predicted IgE binding is its own risk.",
                         "Applicability is how well a peptide fits what a model was trained on: within range, limited or outside range. Read it before the scores.",
                         "Evidence type is a sequence relationship to a labelled evidence peptide: exact sequence, contains it, contained in it, or similar. It is never proof of GLP-1 activity, "
                         "and no evidence means no data, not inactive. Only an exact sequence match with a positive or mixed reported effect passes the evidence gate.",
                         "Uncut bonds: the rules for pepsin, trypsin and chymotrypsin say where each enzyme cuts a peptide chain. The score is the share of the peptide's internal bonds that no rule "
                         "cuts. It is a rule-based prediction and does not measure survival in the gut.",
                         "A designed variant is an edited copy of a digestion fragment. It is looked up as itself. Its template's evidence is shown separately and is not transferred to it.",
                         "The biology lead's decision does not change the computed rank."])]
    return {"top": top, "wide": wide, "left": left, "right": []}


READING_NOTES = [
    "Digestion detail: a cut is shown as the residue before it (with its position in the UniProt entry) | the residue after it, with the enzymes that cut there. Final means no enzyme of the "
    "route can cut inside the fragment; intermediate means one still could. These are rule predictions, not observed by LC-MS/MS.",
    "Alignment: the top line is the peptide and the bottom line the evidence peptide; | marks an identical position and + a similar one. Identity is identical aligned positions over aligned "
    "positions. Coverage is the aligned residues over the length of that sequence. A sequence similarity is not experimentally confirmed GLP-1 activity.",
]


def plain_shared(dossiers: list[FoodDossier]) -> dict:
    """The second page: what is the same for every peptide."""
    first = dossiers[0]
    controls = [[ROLE.get(c.get("role"), words(c.get("role", ""))), c.get("sequence"), c.get("study"), c.get("assay"), c.get("n_observations")] for c in first.controls]
    top = ["Shared information", "What applies to every peptide on the pages that follow", sentence(first.status) + ".", ""]
    left = [("title", "Reference controls (proposed by rule)"),
            ("table", ["Role", "Sequence", "Study", "Assay", "Observations"], controls, [0.19, 0.24, 0.23, 0.16, 0.18]) if controls else ("small", "None."),
            ("title", "Assay plan: set by the biology lead"),
            ("text", f"Primary endpoint: {first.assay_plan.get('primary_endpoint', '')}. Material: {first.assay_plan.get('material_type', '')}. "
                     "The same cells, one row per candidate, are in the experiment sheet (CSV)."),
            ("blank", [ASSAY.get(k.split(" (")[0], words(k)) for k in first.assay_plan.get("set_by_biology_lead", {})])]
    left += [("title", "How to read the digestion and alignment blocks"), ("bullets", READING_NOTES)]
    right = [("title", "Not established"), ("bullets", list(first.not_established)),
             ("title", "Limitations that apply to every peptide"), ("bullets", shared_limits(dossiers) or ["None recorded."])]
    return {"top": top, "left": left, "right": right}


# ----------------------------------------------------------------------------------------------------
# Drawing the pages
# ----------------------------------------------------------------------------------------------------

#: Smallest type scale tried (1 = 7.6 pt body text).
MIN_SCALE = 0.7

#: Characters the PDF's built-in font (Windows-1252) cannot draw, and what stands in for them; any other such character becomes "?".
_ASCII = {"≥": ">=", "≤": "<=", "→": "->", "←": "<-", "−": "-", "±": "+/-", "×": "x", "≈": "~", "Δ": "delta "}

#: Saved results of earlier runs carry a "; uncalibrated" note in each model's detail text; the PDF leaves it out.
_UNCALIBRATED = re.compile(r"\s*[;,]\s*uncalibrated|uncalibrated\s*", re.IGNORECASE)


def _text(value) -> str:
    """Escaped text for a Paragraph; a character the font cannot draw becomes its plain form or '?'."""
    out = []
    for ch in _UNCALIBRATED.sub("", str(value)):
        ch = _ASCII.get(ch, ch)
        try:
            ch.encode("cp1252")
        except UnicodeEncodeError:
            ch = "?"
        out.append(ch)
    return escape("".join(out))


def page_scale(dossier: FoodDossier) -> float:
    """The type scale (1 = 7.6 pt body text) the peptide page of this dossier is set at."""
    return _build([dossier])[1][2]


def dossier_pdf(dossiers: list[FoodDossier]) -> bytes:
    """The overview page of the dossiers' peptides, the shared-information page, then one page per dossier in the order given."""
    return _build(dossiers)[0]


def _build(dossiers: list[FoodDossier]) -> tuple[bytes, list[float]]:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import BaseDocTemplate, Frame, PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle

    page_w, page_h = landscape(A4)
    margin, gap = 24, 18
    inner_w, inner_h = page_w - 2 * margin, page_h - 2 * margin
    col_w = (inner_w - gap) / 2
    ink, grey, line, wash, warn = colors.HexColor("#182638"), colors.HexColor("#627389"), colors.HexColor("#c9d4e0"), colors.HexColor("#eef3f8"), colors.HexColor("#fff3df")

    def render(plain: dict, k: float):
        base = ParagraphStyle("base", fontName="Helvetica", fontSize=7.6 * k, leading=9.2 * k, textColor=ink)
        small = ParagraphStyle("small", parent=base, fontSize=6.6 * k, leading=7.9 * k, textColor=grey)
        head = ParagraphStyle("head", parent=base, fontName="Helvetica-Bold", fontSize=8.8 * k, leading=10.6 * k, textColor=colors.HexColor("#0d3b66"), spaceBefore=6 * k, spaceAfter=2 * k)
        title = ParagraphStyle("title", parent=base, fontName="Helvetica-Bold", fontSize=20 * k, leading=23 * k)
        bold = ParagraphStyle("bold", parent=base, fontName="Helvetica-Bold")
        banner = ParagraphStyle("banner", parent=base, fontSize=7.8 * k, leading=9.4 * k, backColor=warn, borderPadding=(3, 4, 3, 4), borderColor=colors.HexColor("#a35a00"))

        def para(value, style=base):
            return Paragraph(_text(value), style)

        def grid(rows, widths, header=None, label_col=False):
            data = ([[para(h, bold) for h in header]] if header else []) + [[para(c) for c in row] for row in rows]
            table = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
            style = [("GRID", (0, 0), (-1, -1), 0.4, line), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                     ("TOPPADDING", (0, 0), (-1, -1), 1.6 * k), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6 * k)]
            if header:
                style.append(("BACKGROUND", (0, 0), (-1, 0), wash))
            if label_col:
                style.append(("BACKGROUND", (0, 0), (0, -1), wash))
            table.setStyle(TableStyle(style))
            return table

        def column(blocks, width=col_w):
            out = []
            for block in blocks:
                kind = block[0]
                if kind == "title":
                    out.append(Paragraph(_text(block[1]), head))
                elif kind == "text":
                    out.append(para(block[1]))
                elif kind == "small":
                    if block[1]:
                        out.append(para(block[1], small))
                elif kind == "pairs":
                    if block[1]:
                        out.append(grid([[label, value] for label, value in block[1]], [width * 0.5, width * 0.5], label_col=True))
                elif kind == "bullets":
                    out += [para(f"- {item}") for item in block[1]]
                elif kind == "table":
                    out.append(grid(block[2], [width * f for f in block[3]], header=block[1]))
                elif kind == "mono":
                    mono = ParagraphStyle("mono", parent=base, fontName="Courier", fontSize=7.6 * k, leading=9.2 * k)
                    out.append(Paragraph("<br/>".join(_text(line).replace(" ", "&nbsp;") for line in block[1]), mono))
                elif kind == "blank":
                    out.append(grid([[label, ""] for label in block[1]], [width * 0.5, width * 0.5], label_col=True))
            return out

        top = [para(plain["top"][0], title), Spacer(1, 2 * k), para(plain["top"][1], small), Spacer(1, 4 * k), Paragraph(_text(plain["top"][2]), banner), Spacer(1, 3 * k), para(plain["top"][3]), Spacer(1, 2 * k)]
        body = Table([[column(plain["left"]), column(plain["right"])]], colWidths=[col_w + gap / 2, col_w + gap / 2], hAlign="LEFT")
        body.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (0, 0), gap), ("RIGHTPADDING", (1, 0), (1, 0), 0),
                                  ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
        return [*top, *column(plain.get("wide", []), inner_w), body]

    def fit_one(plain: dict) -> tuple[list, float]:
        """The page described by `plain` at the largest type size (scale 1 down to MIN_SCALE) at which it fits one page; below MIN_SCALE it is drawn at MIN_SCALE and may overflow."""
        k = 1.0
        while True:
            flowables = render(plain, k)
            height = sum(f.wrap(inner_w, 10 ** 6)[1] + f.getSpaceBefore() + f.getSpaceAfter() for f in flowables) + 4
            if height <= inner_h or k <= MIN_SCALE:
                return flowables, k
            k = round(k - 0.02, 2)

    def fit(plain: dict) -> tuple[list, float]:
        """A page may offer alternative placements of its blocks in the two columns; the one that leaves the larger type is used."""
        options = [{**plain, "left": left, "right": right} for left, right in plain.get("alternatives", [])] or [plain]
        return max((fit_one(option) for option in options), key=lambda result: result[1])

    buffer = io.BytesIO()
    ranking_page = [plain_ranking(dossiers)] if dossiers and dossiers[0].ranking_summary else []  # only when the run recorded how it ranked
    total = len(dossiers) + 2 + len(ranking_page)

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 6.5)
        canvas.setFillColor(grey)
        canvas.drawRightString(page_w - margin, 10, f"page {doc.page} of {total}")
        canvas.restoreState()

    document = BaseDocTemplate(buffer, pagesize=landscape(A4), leftMargin=margin, rightMargin=margin, topMargin=margin, bottomMargin=margin, title="Peptide review dossiers",
                               author="Peptide discovery platform")
    document.addPageTemplates([PageTemplate(id="p", frames=[Frame(margin, margin, inner_w, inner_h, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)], onPage=footer)])
    story, scales = [], []
    shared = shared_limits(dossiers)
    pages = [plain_overview(dossiers), plain_shared(dossiers), *ranking_page, *[plain_dossier(d, shared) for d in dossiers]]
    for i, plain in enumerate(pages):
        flowables, k = fit(plain)
        scales.append(k)
        story += flowables + ([PageBreak()] if i < total - 1 else [])
    document.build(story)
    return buffer.getvalue(), scales
