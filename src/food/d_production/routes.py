"""Which enzymes can release a candidate peptide from its parent protein, and whether they destroy it on the way."""
from __future__ import annotations

from dataclasses import dataclass

from food.a_proteins.fasta import Substrate
from food.b_digestion.engine import cumulative_cleavages
from food.b_digestion.settings import DEFAULT_SETTINGS, Settings

INTACT = "intact"  # both ends are cut and nothing inside is cut
NEEDS_MISSED_CLEAVAGES = "needs_missed_cleavages"  # both ends are cut, but the enzymes also cut inside it: it survives only if those cuts are missed
DESTROYED_LATER = "destroyed_in_later_stage"  # released at one stage, cut inside at a later stage of the same route
NOT_RELEASED = "not_released"  # no stage of the route cuts both ends


@dataclass(frozen=True)
class InternalCut:
    position_in_peptide: int  # 1-based residue after which the bond is cut
    residue: str
    cleaved_by: tuple[str, ...]  # 'stage:enzyme'


@dataclass(frozen=True)
class ReleaseRoute:
    peptide: str
    protein_id: str
    start: int  # 1-based, in the protein sequence
    end: int
    pathway: str
    release_stage: str | None  # first stage after which both ends are cut; None when no stage does
    n_boundary: tuple[str, ...]  # 'stage:enzyme' cutting before the peptide by the end of the route; ('protein_terminus',) when it starts the chain; () when uncut
    c_boundary: tuple[str, ...]
    internal_cuts_at_release: tuple[InternalCut, ...]  # inside the peptide when it is released
    internal_cuts_final: tuple[InternalCut, ...]  # inside it after the last stage
    missed_cleavages_required: int  # internal cuts at release that must be missed for the peptide to survive
    verdict: str

    @property
    def releases_intact(self) -> bool:
        return self.verdict == INTACT


def _boundary(cuts: dict[int, list[str]], site: int, terminus: bool) -> tuple[str, ...]:
    return ("protein_terminus",) if terminus else tuple(sorted(cuts.get(site, ())))


def _internal(cuts: dict[int, list[str]], start: int, end: int, sequence: str) -> tuple[InternalCut, ...]:
    return tuple(InternalCut(i - start + 1, sequence[i], tuple(sorted(cuts[i]))) for i in sorted(cuts) if start <= i < end)


def release_routes(peptide: str, protein: Substrate, settings: Settings = DEFAULT_SETTINGS) -> list[ReleaseRoute]:
    """One route per place the peptide sits in the protein and per digestion pathway of `settings` (single-enzyme and sequential).

    A route's stages run in order and cleavage is irreversible, so a bond cut at one stage stays cut. Raises ValueError when the protein does not contain the peptide.
    """
    seq = protein.sequence
    starts = [i for i in range(len(seq) - len(peptide) + 1) if seq.startswith(peptide, i)] if peptide else []
    if not starts:
        raise ValueError(f"{peptide} does not occur in {protein.substrate_id}")
    registry = settings.registry()
    routes: list[ReleaseRoute] = []
    for pathway in settings.pathways:
        by_stage = cumulative_cleavages(seq, pathway, registry)
        for start in starts:
            end = start + len(peptide) - 1  # 0-based, inclusive
            n_term, c_term = start == 0, end == len(seq) - 1
            released = next((k for k, cuts in enumerate(by_stage) if (n_term or start - 1 in cuts) and (c_term or end in cuts)), None)
            final = by_stage[-1]
            at_release = _internal(by_stage[released], start, end, seq) if released is not None else ()
            after = _internal(final, start, end, seq)
            if released is None:
                verdict = NOT_RELEASED
            elif at_release:
                verdict = NEEDS_MISSED_CLEAVAGES
            elif after:
                verdict = DESTROYED_LATER
            else:
                verdict = INTACT
            routes.append(ReleaseRoute(
                peptide=peptide, protein_id=protein.substrate_id, start=start + 1, end=end + 1, pathway=pathway.id,
                release_stage=None if released is None else pathway.stages[released].name,
                n_boundary=_boundary(final, start - 1, n_term), c_boundary=_boundary(final, end, c_term),
                internal_cuts_at_release=at_release, internal_cuts_final=after, missed_cleavages_required=len(at_release), verdict=verdict))
    return routes
