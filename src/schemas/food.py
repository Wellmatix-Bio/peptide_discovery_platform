"""The food pipeline's candidate: a `Candidate` that also carries where the peptide came from, how it was made, and whether it was observed or only predicted."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, computed_field, model_validator

from schemas.candidate import Candidate

OriginType = Literal["DIGESTION_DERIVED", "SYNTHETIC_DESIGN", "LITERATURE_KNOWN"]

#: THEORETICAL is a prediction (an enzyme rule or a model); OBSERVED is a measurement (e.g. LC-MS/MS). They are never merged.
EvidenceClass = Literal["THEORETICAL", "OBSERVED"]

MaterialClass = Literal["peptide", "hydrolysate", "intact_protein"]


class ParentOrigin(BaseModel):
    """One way a peptide arises from one parent protein (one digestion pathway and stage, or one generation template)."""

    protein_id: str = Field(min_length=1)  # the library record, e.g. "P02754:17-178"
    accession: str = ""
    protein_version: str | None = None  # UniProt entry version, when known
    protein_checksum: str | None = None  # of the parent sequence, so the lineage survives a registry change
    food_source: str = ""
    #: True / False when the library says so, None when it has not been assessed. None is not False.
    allergen_source: bool | None = None
    #: The two ends of the parent protein chain, from the library FASTA (a TERMINUS_MEANING code); None when the input did not say. Not the peptide's own ends (`n_term`, `c_term`).
    protein_n_terminus: str | None = None
    protein_c_terminus: str | None = None

    start_in_product: int | None = Field(default=None, ge=1)
    end_in_product: int | None = Field(default=None, ge=1)
    start_in_precursor: int | None = Field(default=None, ge=1)
    end_in_precursor: int | None = Field(default=None, ge=1)

    pathway: str = ""
    stage: str = ""
    enzymes: list[str] = Field(default_factory=list)
    ph: float | None = None
    missed_cleavages_stage: int | None = Field(default=None, ge=0)
    missed_cleavages_final: int | None = Field(default=None, ge=0)
    n_term: str = ""  # "protein_terminus" or the stage:enzyme that cuts there
    c_term: str = ""
    is_limit_product: bool | None = None
    #: The digestion run again for this origin (pipeline/food/digestion/explain.py): the stages with their enzymes and pH, the cut on each side with its flanking residues, the cuts still inside it,
    #: and final or intermediate. For a designed peptide it describes the template fragment. None for runs made before it existed.
    digestion_detail: dict | None = None

    @computed_field
    @property
    def digestion_product_type(self) -> str | None:
        """FINAL_PRODUCT when no enzyme of the pathway cuts inside the peptide any more (a limit product); INTERMEDIATE_PRODUCT when one still could; None when unknown."""
        if self.is_limit_product is None:
            return None
        return "FINAL_PRODUCT" if self.is_limit_product else "INTERMEDIATE_PRODUCT"


class FoodCandidate(Candidate):
    origin_type: OriginType
    evidence_class: EvidenceClass = "THEORETICAL"
    material_class: MaterialClass = "peptide"
    generation_method: str = ""  # e.g. "in_silico_digestion"
    parents: list[ParentOrigin] = Field(default_factory=list)
    duplicate_family_id: str | None = None  # peptides with identical sequences share it
    parent_clusters: list[str] = Field(default_factory=list)
    mono_mass: float | None = None
    avg_mass: float | None = None
    chemistry_notes: str = ""
    #: LC-MS/MS records that observed the peptide. The shape is not defined yet (no measured data enters the pipeline); `OBSERVED` is refused without at least one.
    lcms_evidence: list[dict] = Field(default_factory=list)

    @model_validator(mode="after")
    def _digestion_needs_a_parent(self) -> "FoodCandidate":
        if self.origin_type == "DIGESTION_DERIVED" and not self.parents:
            raise ValueError("a DIGESTION_DERIVED peptide needs at least one parent origin")
        return self

    @model_validator(mode="after")
    def _observed_needs_lcms_evidence(self) -> "FoodCandidate":
        if self.evidence_class == "OBSERVED" and not self.lcms_evidence:
            raise ValueError("a peptide is OBSERVED only with LC-MS/MS evidence; without it, it stays THEORETICAL")
        return self

    @computed_field
    @property
    def digestion_product_type(self) -> str | None:
        """For a digestion-derived peptide: FINAL_PRODUCT if some pathway leaves it uncut inside, otherwise INTERMEDIATE_PRODUCT. None for a designed peptide, which is not a digestion product."""
        if self.origin_type != "DIGESTION_DERIVED" or not self.parents:
            return None
        return "FINAL_PRODUCT" if any(p.is_limit_product for p in self.parents) else "INTERMEDIATE_PRODUCT"

    @computed_field
    @property
    def any_parent_allergen_source(self) -> bool | None:
        """True if any parent is an allergen source; False only if every parent is known not to be; None if that is unknown."""
        flags = [p.allergen_source for p in self.parents]
        if any(f is True for f in flags):
            return True
        if flags and all(f is False for f in flags):
            return False
        return None
