# Route C: target-interface design — structure retrieval, pocket/hotspot
# identification, RFdiffusion backbone generation, ProteinMPNN sequence design,
# AF-Multimer/OpenMM refinement and validation.
#
# Ported from Generation Stage/route_C.ipynb. Stages 0-5 (target resolution
# through hotspot identification) run against live UniProt/PDBe/AlphaFold APIs
# and locally-installed fpocket/P2Rank (via WSL) and EvoEF2/ESM2. Stages 6-8
# (RFdiffusion, ProteinMPNN, AF-Multimer/OpenMM refinement) are written against
# each tool's real CLI but require tools not installed by default — see the
# notebook's own "Setup" section for install commands. This module does not
# lower that bar; it is a faithful port, not a simplification.

from __future__ import annotations

import datetime
import hashlib
import json
import pathlib
import pickle
import re
import subprocess

import gemmi
import httpx
from openmm import app as openmm_app, LangevinMiddleIntegrator, Platform, unit
from pdbfixer import PDBFixer
from openmm.app import PDBFile
from rapidfuzz import fuzz

from common.logging import get_logger
from pipeline.s04_generation.routeA import ConstraintConfig, physicochemical_filter
from src.schemas.stage4_route import Stage4Route
from schemas.candidate import Candidate

try:
    import freesasa
except ImportError as exc:
    raise ImportError(
        "freesasa is required for pocket detection. Install it with: pip install freesasa"
    ) from exc

try:
    import torch
    import esm
except ImportError as exc:
    raise ImportError(
        "torch and fair-esm are required for hotspot identification (ESM2 masked-marginal "
        "scoring). Install them with: pip install torch fair-esm"
    ) from exc

logger = get_logger(__name__)


# ============================================================================
# 0. Generic inputs
# ============================================================================

ORGANISM_TAXON_MAP = {
    "homo sapiens": 9606,
    "mus musculus": 10090,
    "rattus norvegicus": 10116,
    "danio rerio": 7955,
}


def resolve_taxon_id(organism: str) -> int:
    key = organism.strip().lower()
    if key not in ORGANISM_TAXON_MAP:
        raise ValueError(f"Unknown organism '{organism}' — add to ORGANISM_TAXON_MAP")
    return ORGANISM_TAXON_MAP[key]


# ============================================================================
# 1. UniProt target resolution
# ============================================================================

UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_SEARCH_FIELDS = [
    "accession",
    "id",
    "gene_names",
    "protein_name",
    "organism_name",
    "length",
    "sequence",
    "xref_pdb",
    "protein_existence",
    "annotation_score",
]

PROTEIN_EXISTENCE_RANK = {
    "1: Evidence at protein level": 1,
    "2: Evidence at transcript level": 2,
    "3: Inferred from homology": 3,
    "4: Predicted": 4,
    "5: Uncertain": 5,
}

GENE_MATCH_SCORE = {"primary": 50, "synonym": 20, "none": 0}
PROTEIN_EXISTENCE_SCORE = {1: 40, 2: 30, 3: 20, 4: 10, 5: 0}


async def search_uniprot(target_input: dict, verbose: bool = False) -> list[dict]:
    params = {
        "query": f'(gene:{target_input["target_name"]}) AND (organism_name:"{target_input["organism"]}") AND (reviewed:true)',
        "fields": UNIPROT_SEARCH_FIELDS,
        "sort": "accession desc",
        "size": "20",
    }
    headers = {"accept": "application/json"}

    async with httpx.AsyncClient() as client:
        response = await client.get(UNIPROT_SEARCH_URL, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        if verbose:
            logger.info("routec.uniprot.search_raw", extra={"data": json.dumps(data)})
        return data["results"]


def filter_uniprot_results(results: list[dict], target_input: dict) -> list[dict]:
    filtered = []
    target_taxon = resolve_taxon_id(target_input["organism"])
    for res in results:
        res_organism = res.get("organism", {}).get("taxonId", None)
        if res_organism != target_taxon:
            continue

        gene_match_type = "none"
        for gene in res.get("genes", []):
            if gene.get("geneName", {}).get("value", "").lower() == target_input["target_name"].lower():
                gene_match_type = "primary"
                break
            if any(
                synonym.get("value", "").lower() == target_input["target_name"].lower()
                for synonym in gene.get("synonyms", [])
            ):
                gene_match_type = "synonym"
                break

        if gene_match_type == "none":
            continue

        res["_gene_match_type"] = gene_match_type
        filtered.append(res)
    return filtered


def score_uniprot_entry(entry: dict, target_input: dict) -> dict:
    reviewed = entry.get("entryType") == "UniProtKB reviewed (Swiss-Prot)"

    protein_existence_raw = entry.get("proteinExistence", "")
    protein_existence = PROTEIN_EXISTENCE_RANK.get(protein_existence_raw)

    annotation_score = entry.get("annotationScore")

    recommended_name = (
        entry.get("proteinDescription", {})
        .get("recommendedName", {})
        .get("fullName", {})
        .get("value", "")
    )
    protein_name_sim = fuzz.token_sort_ratio(target_input["target_name"], recommended_name)

    gene_match_type = entry.get("_gene_match_type", "none")

    score = 0
    score += 100 if reviewed else 0
    score += GENE_MATCH_SCORE[gene_match_type]
    score += PROTEIN_EXISTENCE_SCORE.get(protein_existence, 0)
    score += (annotation_score or 0) * 8
    score += (protein_name_sim / 100) * 30

    return {
        "uniprot_accession": entry.get("primaryAccession"),
        "reviewed": reviewed,
        "gene_match_type": gene_match_type,
        "protein_existence": protein_existence,
        "annotation_score": annotation_score,
        "protein_name_sim": protein_name_sim,
        "score": score,
    }


async def resolve_uniprot_target(target_input: dict) -> dict:
    """Resolve target_input (name/gene/organism) to a single UniProt entry via
    live search + scoring. Raises ValueError if nothing suitable is found."""
    if target_input.get("uniprot_id"):
        # Direct short-circuit: caller already knows the accession.
        return {"uniprot_accession": target_input["uniprot_id"]}

    results = await search_uniprot(target_input)
    filtered = filter_uniprot_results(results, target_input)
    scored = [score_uniprot_entry(entry, target_input) for entry in filtered]
    scored.sort(key=lambda s: -s["score"])

    selected = scored[0] if scored else None
    if not selected:
        raise ValueError("No suitable UniProt entry found for the target.")
    return selected


# ============================================================================
# 2. PDB structure retrieval and selection (experimental, with AlphaFold fallback)
# ============================================================================

METHOD_SCORE = {
    "X-ray diffraction": 40,
    "Electron Microscopy": 35,
    "Solution NMR": 20,
}


async def fetch_best_structures(uniprot_accession: str) -> list[dict]:
    response = httpx.get(
        f"https://www.ebi.ac.uk/pdbe/api/v2/mappings/best_structures/{uniprot_accession}",
        headers={"accept": "application/json"},
        timeout=30.0,
    )
    response.raise_for_status()
    best_structures = response.json()
    return best_structures.get(uniprot_accession, [])


def score_experimental(candidate: dict) -> dict:
    method = candidate.get("experimental_method", "")
    resolution = candidate.get("resolution")
    coverage = candidate.get("coverage", 0.0)

    method_score = METHOD_SCORE.get(method, 10)

    # lower resolution (Angstroms) is better; no resolution (e.g. NMR) treated as neutral
    if resolution is None:
        resolution_score = 20
    else:
        resolution_score = max(0, 40 - (resolution * 10))

    coverage_score = coverage * 40

    breakdown = {
        "method_score": method_score,
        "resolution_score": resolution_score,
        "coverage_score": coverage_score,
    }
    return {
        "pdb_id": candidate.get("pdb_id"),
        "af_model_id": None,
        "structure_source": "experimental",
        "method": method,
        "resolution": resolution,
        "chain_id": candidate.get("chain_id"),
        # Resolved per-entry by filter_candidates_for_complex() below (chain lettering for
        # the same biological partner is not stable across PDB entries -- VEGF-A is chain A
        # in 3v2a but could be lettered differently elsewhere), so it travels with the
        # candidate from here rather than being looked up separately downstream from a
        # static fixture value.
        "partner_chain_id": candidate.get("partner_chain_id"),
        "score_total": sum(breakdown.values()),
        "score_breakdown": breakdown,
    }


async def fetch_alphafold_structure(uniprot_accession: str, preferred_domain=None) -> dict:
    async with httpx.AsyncClient() as client:
        pred_resp = await client.get(
            f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_accession}",
            headers={"accept": "application/json"},
            timeout=30.0,
        )
        pred_resp.raise_for_status()
        prediction = pred_resp.json()[0]

        mean_plddt = prediction["globalMetricValue"]

        conf_resp = await client.get(prediction["plddtDocUrl"], timeout=30.0)
        conf_resp.raise_for_status()
        confidence = conf_resp.json()

    low_confidence_regions = [
        residue
        for residue, plddt in zip(confidence["residueNumber"], confidence["confidenceScore"])
        if plddt < 70
    ]

    domain_overlap = False
    if preferred_domain is not None:
        dom_start, dom_end = preferred_domain
        domain_overlap = any(dom_start <= r <= dom_end for r in low_confidence_regions)

    breakdown = {
        "mean_plddt": mean_plddt,
        "low_confidence_residue_count": len(low_confidence_regions),
        "low_confidence_overlaps_preferred_domain": domain_overlap,
    }

    return {
        "pdb_id": None,
        "af_model_id": prediction["entryId"],
        "structure_source": "predicted",
        "method": "AlphaFold",
        "resolution": None,
        "chain_id": prediction.get("chainId"),
        "score_total": mean_plddt,
        "score_breakdown": breakdown,
    }


async def resolve_partner_chain(pdb_id: str, own_chain_id: str, partner_gene_symbols: list[str]) -> str | None:
    """A target_complex request names the desired partner by gene identity (e.g. VEGFA for
    'VEGFR2 bound to VEGF-A'), never by chain letter -- chain lettering for the same
    biological partner is not stable across PDB entries (VEGF-A is chain A in 3v2a, but nothing
    guarantees that elsewhere), and a candidate merely HAVING a second chain proves nothing:
    5oyj has two protein chains too, but the second one is an unrelated DARPin, not VEGF-A.
    So this checks each non-target protein entity's own gene_name list against the requested
    symbols, and returns the actual chain letter that entity occupies in *this* entry -- or
    None if no entity in this structure matches, which is a real 'not a candidate' result,
    not just 'binds something'."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"https://www.ebi.ac.uk/pdbe/api/pdb/entry/molecules/{pdb_id.lower()}",
            headers={"accept": "application/json"},
            timeout=30.0,
        )
    resp.raise_for_status()
    molecules = resp.json().get(pdb_id.lower(), [])
    wanted = {g.upper() for g in partner_gene_symbols}

    for entity in molecules:
        if entity.get("molecule_type") != "polypeptide(L)":
            continue
        if own_chain_id in entity.get("in_chains", []):
            continue  # this is the target itself, not a candidate partner
        entity_genes = {g.upper() for g in (entity.get("gene_name") or [])}
        if entity_genes & wanted:
            chains = entity.get("in_chains", [])
            if chains:
                return chains[0]
    return None


async def filter_candidates_for_complex(candidates: list[dict], partner_gene_symbols: list[str]) -> list[dict]:
    """Hard filter (same treatment as preferred_domain elsewhere in this pipeline): when the
    caller explicitly asked for a target_complex, a candidate that doesn't actually deposit
    the named partner is not a candidate at all, no matter how good its resolution/coverage
    otherwise looks -- without this, score_experimental's resolution/coverage weighting alone
    reliably prefers isolated high-resolution domain structures (e.g. 3vhe at ~1.55 A) over
    the real, lower-resolution complex structure (e.g. 3v2a at 3.2 A), silently never honoring
    target_complex at all. Resolves and attaches the real per-entry partner_chain_id onto each
    surviving candidate rather than just returning a yes/no."""
    keep = []
    for candidate in candidates:
        partner_chain_id = await resolve_partner_chain(
            candidate["pdb_id"], candidate["chain_id"], partner_gene_symbols
        )
        if partner_chain_id is not None:
            keep.append({**candidate, "partner_chain_id": partner_chain_id})
    return keep


async def select_structure(uniprot_accession: str, target_input: dict) -> tuple[dict, list[dict]]:
    """Returns (structure_record, experimental_fallback_pool)."""
    filtered_candidates = await fetch_best_structures(uniprot_accession)

    target_complex = target_input.get("target_complex")
    if target_complex and target_complex.get("partner_gene_symbols") and len(filtered_candidates) > 0:
        filtered_candidates = await filter_candidates_for_complex(
            filtered_candidates, target_complex["partner_gene_symbols"]
        )

    if len(filtered_candidates) == 0:
        structure_record = await fetch_alphafold_structure(
            uniprot_accession, preferred_domain=target_input.get("preferred_domain")
        )
        structure_record["fallback_pool"] = "alphafold"
        experimental_fallback_pool = []
    else:
        scored_candidates = [score_experimental(c) for c in filtered_candidates]
        scored_candidates.sort(key=lambda s: -s["score_total"])
        structure_record = scored_candidates[0]
        structure_record["fallback_pool"] = "pdb"
        # remaining ranked candidates to fall back into if structure fetching/validation fails
        experimental_fallback_pool = scored_candidates[1:]

    return structure_record, experimental_fallback_pool


# ============================================================================
# 3. Structure fetching, cleaning, and preparation
# ============================================================================

STRUCTURE_CACHE_DIR = pathlib.Path("structure_cache")

# Common crystallization buffer / cryoprotectant heteroatoms — stripped even
# though they're technically NonPolymer entities, since they aren't biological ligands.
CRYSTALLIZATION_ARTIFACTS = {
    "SO4", "PO4", "GOL", "EDO", "PEG", "PG4", "1PE", "MPD", "TRS", "HEPES",
    "ACT", "CIT", "IMD", "BME", "DMS", "FMT", "NA", "CL", "K", "MG", "CA",
    "ZN", "MN", "NI", "CO", "CD", "BR", "IOD", "UNX",
}

GAP_FLAG_THRESHOLD = 5  # residues


async def download_structure_cif(pdb_id: str) -> pathlib.Path:
    STRUCTURE_CACHE_DIR.mkdir(exist_ok=True)
    dest = STRUCTURE_CACHE_DIR / f"{pdb_id.lower()}.cif"
    if dest.exists():
        return dest

    async with httpx.AsyncClient(follow_redirects=True) as client:
        resp = await client.get(
            f"https://files.rcsb.org/download/{pdb_id.upper()}.cif",
            timeout=30.0,
        )
    if resp.status_code == 404:
        raise FileNotFoundError(f"No mmCIF available for {pdb_id}")
    resp.raise_for_status()

    dest.write_bytes(resp.content)
    return dest


async def fetch_sifts_uniprot_mapping(pdb_id: str, uniprot_accession: str) -> list[dict]:
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id.lower()}",
            headers={"accept": "application/json"},
            timeout=30.0,
        )
    resp.raise_for_status()
    data = resp.json()
    entry = data.get(pdb_id.lower(), {}).get("UniProt", {}).get(uniprot_accession, {})
    return entry.get("mappings", [])


def subtract_chains(structure: gemmi.Structure, chain_id: str, partner_chain_id: str | None) -> gemmi.Structure:
    model = structure[0]
    keep = {chain_id}
    if partner_chain_id:
        keep.add(partner_chain_id)

    for chain_name in [c.name for c in model]:
        if chain_name not in keep:
            model.remove_chain(chain_name)

    structure.remove_empty_chains()
    return structure


def clean_structure(structure: gemmi.Structure, keep_ligand_resnames: set[str]) -> gemmi.Structure:
    structure.remove_alternative_conformations()  # keeps highest-occupancy altloc, drops rest
    structure.remove_waters()

    model = structure[0]
    for chain in model:
        # gemmi.Chain has no remove_residue()/delete-by-seqid method in the installed version
        # (0.7.5) -- it only supports index-based deletion (del chain[i]), so artifacts must
        # be found by index first, then deleted highest-index-first; deleting while iterating
        # forward would shift every later residue's index out from under the loop.
        artifact_indices = [
            i for i, res in enumerate(chain)
            if res.het_flag == "H" and res.name != "HOH"
            and res.name in CRYSTALLIZATION_ARTIFACTS and res.name not in keep_ligand_resnames
        ]
        for i in reversed(artifact_indices):
            del chain[i]

    structure.remove_empty_chains()
    return structure


def find_gaps(structure: gemmi.Structure, chain_id: str) -> list[tuple[int, int]]:
    model = structure[0]
    chain = model[chain_id]
    poly = chain.get_polymer()
    nums = sorted(r.seqid.num for r in poly)

    gaps = []
    for prev, curr in zip(nums, nums[1:]):
        if curr - prev > 1:
            gaps.append((prev, curr))
    return gaps


def flagged_gaps(gaps: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(a, b) for a, b in gaps if (b - a - 1) > GAP_FLAG_THRESHOLD]


def build_resnum_map(sifts_mappings: list[dict], chain_id: str) -> dict[int, int]:
    """author_residue_number can legitimately be null in SIFTS for some depositions --
    observed on cryo-EM entry (mouse Kdr, PDBe's own /mappings/uniprot response),
    where PDBe fell back to an internal sequential residue_number instead of the
    author-assigned one. That internal index is not the structure's real residue
    numbering (the .cif/.pdb file itself uses author numbers), so it can't be
    substituted in -- the honest behavior is to skip the segment and leave those
    residues unmapped, not to guess, and not to crash the whole candidate out of
    experimental_fallback_pool over one bad segment."""
    resnum_map = {}
    for mapping in sifts_mappings:
        if mapping.get("struct_asym_id") != chain_id and mapping.get("chain_id") != chain_id:
            continue
        unp_start = mapping["unp_start"]
        struct_start = mapping["start"]["author_residue_number"]
        struct_end = mapping["end"]["author_residue_number"]
        if struct_start is None or struct_end is None:
            continue
        span = struct_end - struct_start
        for offset in range(span + 1):
            resnum_map[struct_start + offset] = unp_start + offset
    return resnum_map


def fix_missing_atoms(cif_path: pathlib.Path) -> pathlib.Path:
    fixer = PDBFixer(filename=str(cif_path))
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)

    out_path = cif_path.with_name(cif_path.stem + "_fixed.pdb")
    with open(out_path, "w") as f:
        # keepIds=True preserves original chain/residue numbering — without it,
        # OpenMM renumbers residues starting from 1, which silently breaks
        # uniprot_to_structure_resnum_map (built against the pre-fix numbering)
        # for every downstream stage that reads local_file_path.
        PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)
    return out_path


def validate_prepared_structure(
    structure: gemmi.Structure,
    chain_id: str,
    expected_length: int,
    target_complex: dict | None,
    preferred_domain: tuple[int, int] | None,
    resnum_map: dict[int, int],
) -> list[str]:
    failures = []

    if not resnum_map:
        # An empty mapping makes the structure unusable for downstream UniProt-based steps.
        failures.append(
            "uniprot_to_structure_resnum_map is empty -- no residues could be "
            "mapped from UniProt to this structure (e.g. missing author_residue_number "
            "in SIFTS for every segment)"
        )

    model = structure[0]
    chain_length_after = len(model[chain_id].get_polymer())
    if chain_length_after < 0.8 * expected_length:
        failures.append(
            f"chain_length_after_cleaning ({chain_length_after}) < 0.8 * expected ({expected_length})"
        )

    if target_complex and target_complex.get("ligand_resname"):
        ligand_resname = target_complex["ligand_resname"]
        found = any(res.name == ligand_resname for chain in model for res in chain)
        if not found:
            failures.append(f"ligand_of_interest '{ligand_resname}' missing after cleaning")

    if preferred_domain is not None:
        dom_start, dom_end = preferred_domain
        structure_positions = set(resnum_map.values())
        if not any(dom_start <= pos <= dom_end for pos in structure_positions):
            failures.append(f"no resolvable coordinates for preferred_domain {preferred_domain}")

    return failures


async def fetch_and_prepare_structure(candidate: dict, target_input: dict, uniprot_accession: str) -> dict | None:
    """Attempt to download, extract, clean, and validate one experimental candidate.
    Returns a prepared structure record, or None if the candidate fails validation."""
    pdb_id = candidate["pdb_id"]
    chain_id = candidate["chain_id"]
    target_complex = target_input.get("target_complex")
    preferred_domain = target_input.get("preferred_domain")
    # Resolved per-entry in select_structure (filter_candidates_for_complex /
    # resolve_partner_chain), not read from target_complex directly -- chain lettering
    # for the same biological partner isn't stable across PDB entries, so target_complex
    # only names WHO the partner is (by gene symbol); WHICH chain letter that is in this
    # specific structure travels on the candidate record itself, same as chain_id already does.
    partner_chain_id = candidate.get("partner_chain_id")
    keep_ligand_resnames = (
        {target_complex["ligand_resname"]}
        if target_complex and target_complex.get("ligand_resname")
        else set()
    )

    try:
        cif_path = await download_structure_cif(pdb_id)
    except (httpx.HTTPStatusError, FileNotFoundError):
        return None

    structure = gemmi.read_structure(str(cif_path))
    structure.setup_entities()

    if chain_id not in {c.name for c in structure[0]}:
        return None

    expected_length = len(structure[0][chain_id].get_polymer())

    structure = subtract_chains(structure, chain_id, partner_chain_id)
    structure = clean_structure(structure, keep_ligand_resnames)

    gaps = find_gaps(structure, chain_id)

    sifts_mappings = await fetch_sifts_uniprot_mapping(pdb_id, uniprot_accession)
    resnum_map = build_resnum_map(sifts_mappings, chain_id)

    failures = validate_prepared_structure(
        structure, chain_id, expected_length, target_complex, preferred_domain, resnum_map
    )
    if failures:
        logger.warning("routec.structure_prep.validation_failed", extra={"pdb_id": pdb_id, "failures": failures})
        return None

    STRUCTURE_CACHE_DIR.mkdir(exist_ok=True)
    cleaned_cif_path = STRUCTURE_CACHE_DIR / f"{pdb_id.lower()}_clean.cif"
    structure.make_mmcif_document().write_file(str(cleaned_cif_path))

    fixed_path = fix_missing_atoms(cleaned_cif_path)

    return {
        "structure_id": pdb_id,
        "local_file_path": str(fixed_path),
        "chain_id": chain_id,
        "uniprot_to_structure_resnum_map": resnum_map,
        "partner_chain_id": partner_chain_id,
        "gaps": gaps,
        "source": "experimental",
        "mean_plddt": None,
        "prep_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


async def prepare_predicted_structure(af_record: dict, uniprot_accession: str) -> dict:
    af_model_id = af_record["af_model_id"]
    async with httpx.AsyncClient() as client:
        pred_resp = await client.get(
            f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_accession}",
            headers={"accept": "application/json"},
            timeout=30.0,
        )
        pred_resp.raise_for_status()
        prediction = pred_resp.json()[0]

        cif_resp = await client.get(prediction["cifUrl"], timeout=30.0)
        cif_resp.raise_for_status()

    STRUCTURE_CACHE_DIR.mkdir(exist_ok=True)
    cif_path = STRUCTURE_CACHE_DIR / f"{af_model_id}.cif"
    cif_path.write_bytes(cif_resp.content)

    structure = gemmi.read_structure(str(cif_path))
    structure.setup_entities()
    chain_id = prediction.get("chainId", "A")

    resnum_map = {r.seqid.num: r.seqid.num for r in structure[0][chain_id].get_polymer()}

    fixed_path = fix_missing_atoms(cif_path)

    return {
        "structure_id": af_model_id,
        "local_file_path": str(fixed_path),
        "chain_id": chain_id,
        "uniprot_to_structure_resnum_map": resnum_map,
        "partner_chain_id": None,
        "gaps": [],
        "source": "predicted",
        "mean_plddt": prediction["globalMetricValue"],
        "prep_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


async def prepare_structure(
    structure_record: dict, experimental_fallback_pool: list[dict], target_input: dict, uniprot_accession: str
) -> dict:
    prepared_structure = None

    if structure_record["structure_source"] == "experimental":
        candidate_pool = [structure_record] + experimental_fallback_pool
        for candidate in candidate_pool:
            prepared_structure = await fetch_and_prepare_structure(candidate, target_input, uniprot_accession)
            if prepared_structure is not None:
                break

    if prepared_structure is None:
        # every experimental candidate failed (or none existed) — fall back to AlphaFold
        if structure_record["structure_source"] != "predicted":
            af_record = await fetch_alphafold_structure(
                uniprot_accession, preferred_domain=target_input.get("preferred_domain")
            )
        else:
            af_record = structure_record
        prepared_structure = await prepare_predicted_structure(af_record, uniprot_accession)

    return prepared_structure


# ============================================================================
# 4. Pocket detection
# ============================================================================

# fpocket (conda-forge, env "pocket") and P2Rank both run inside WSL —
# neither ships a native Windows build.
WSL_MAMBA_EXE = "/home/kounen/.local/bin/micromamba"
WSL_MAMBA_ROOT_PREFIX = "/home/kounen/micromamba"
WSL_P2RANK_HOME = "/home/kounen/tools/p2rank_2.5.1"

DRUGGABILITY_THRESHOLD = 0.5
MIN_POCKET_VOLUME = 50.0    # cubic Angstrom
MAX_POCKET_VOLUME = 2000.0  # cubic Angstrom
BURIED_SASA_THRESHOLD = 1.0  # total pocket-residue SASA below this = fully buried
INTERFACE_CONTACT_CUTOFF = 5.0  # Angstrom


def run_checked(cmd: list, **kwargs):
    """subprocess.run(..., check=True) that surfaces real stdout/stderr on failure --
    plain check=True with capture_output=True raises CalledProcessError whose default
    __str__ only shows the exit code, hiding the actual tool error underneath."""
    kwargs.setdefault("capture_output", True)
    kwargs.setdefault("text", True)
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        cmd_str = " ".join(cmd)
        raise RuntimeError(
            "Command failed (exit " + str(result.returncode) + "): " + cmd_str + "\n"
            "--- stdout ---\n" + (result.stdout or "") + "\n"
            "--- stderr ---\n" + (result.stderr or "")
        )
    return result


def to_wsl_path(win_path: str) -> str:
    result = subprocess.run(
        ["wsl.exe", "-e", "wslpath", "-a", str(win_path)],
        capture_output=True, text=True, check=True, stdin=subprocess.DEVNULL,
    )
    return result.stdout.strip()


def run_fpocket(local_file_path: str, workdir_win: pathlib.Path) -> pathlib.Path:
    wsl_path = to_wsl_path(str(workdir_win))
    filename = pathlib.Path(local_file_path).name
    cmd = [
        "wsl.exe", "-e", "bash", "-lc",
        f"export MAMBA_EXE='{WSL_MAMBA_EXE}'; export MAMBA_ROOT_PREFIX='{WSL_MAMBA_ROOT_PREFIX}'; "
        f"cd '{wsl_path}' && $MAMBA_EXE run -n pocket fpocket -f '{filename}'",
    ]
    run_checked(cmd, stdin=subprocess.DEVNULL)

    out_dir = workdir_win / f"{pathlib.Path(filename).stem}_out"
    if not out_dir.exists():
        raise RuntimeError(f"fpocket did not produce expected output dir: {out_dir}")
    return out_dir


def parse_fpocket_info(info_path: pathlib.Path) -> dict[int, dict]:
    text = info_path.read_text()
    blocks = re.split(r"\nPocket (\d+) :\n", "\n" + text)
    pockets = {}
    # blocks alternates: [prelude, id, body, id, body, ...]
    for i in range(1, len(blocks), 2):
        pocket_id = int(blocks[i])
        body = blocks[i + 1]
        fields = {}
        for line in body.strip().splitlines():
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            key = key.strip().lower()
            val = val.strip()
            try:
                fields[key] = float(val)
            except ValueError:
                fields[key] = val
        pockets[pocket_id] = fields
    return pockets


def parse_fpocket_pocket_residues(pocket_atm_path: pathlib.Path) -> list[int]:
    residues = set()
    for line in pocket_atm_path.read_text().splitlines():
        if line.startswith("ATOM") or line.startswith("HETATM"):
            resnum = int(line[22:26])
            residues.add(resnum)
    return sorted(residues)


def run_fpocket_detection(local_file_path: str) -> list[dict]:
    win_path = pathlib.Path(local_file_path).resolve()
    out_dir = run_fpocket(local_file_path, win_path.parent)

    stem = win_path.stem
    info_path = out_dir / f"{stem}_info.txt"
    pockets_dir = out_dir / "pockets"

    info = parse_fpocket_info(info_path)

    results = []
    for pocket_id, fields in info.items():
        atm_path = pockets_dir / f"pocket{pocket_id}_atm.pdb"
        residue_list = parse_fpocket_pocket_residues(atm_path) if atm_path.exists() else []
        results.append({
            "pocket_id": pocket_id,
            "residue_list": residue_list,
            "druggability_score": fields.get("druggability score"),
            "volume": fields.get("volume"),
            "polarity_score": fields.get("polarity score"),
            "hydrophobicity_score": fields.get("hydrophobicity score"),
        })
    return results


def run_p2rank_detection(local_file_path: str) -> list[dict]:
    win_path = pathlib.Path(local_file_path).resolve()
    wsl_workdir = to_wsl_path(str(win_path.parent))
    filename = win_path.name
    out_subdir = "p2rank_out"

    cmd = [
        "wsl.exe", "-e", "bash", "-lc",
        f"cd '{wsl_workdir}' && '{WSL_P2RANK_HOME}/prank' predict -f '{filename}' -o {out_subdir}",
    ]
    run_checked(cmd, stdin=subprocess.DEVNULL)

    csv_path = win_path.parent / out_subdir / f"{filename}_predictions.csv"
    if not csv_path.exists():
        raise RuntimeError(f"P2Rank did not produce expected predictions CSV: {csv_path}")

    results = []
    lines = csv_path.read_text().strip().splitlines()
    header = [h.strip() for h in lines[0].split(",")]
    for line in lines[1:]:
        fields = [f.strip() for f in line.split(",")]
        row = dict(zip(header, fields))
        residue_list = sorted({
            int(tok.split("_")[1])
            for tok in row["residue_ids"].split()
            if "_" in tok
        })
        results.append({
            "pocket_id": int(row["rank"]),
            "residue_list": residue_list,
            "score": float(row["score"]),
            "probability": float(row["probability"]),
        })
    return results


def compute_residue_sasa(local_file_path: str, chain_id: str) -> dict[int, float]:
    structure = freesasa.Structure(local_file_path)
    result = freesasa.calc(structure)
    residue_areas = result.residueAreas()
    chain_areas = residue_areas.get(chain_id, {})
    return {int(resnum): area.total for resnum, area in chain_areas.items()}


def pocket_is_buried(residue_list: list[int], sasa_by_residue: dict[int, float]) -> bool:
    total_sasa = sum(sasa_by_residue.get(r, 0.0) for r in residue_list)
    return total_sasa < BURIED_SASA_THRESHOLD


def domain_overlaps(residue_list: list[int], preferred_domain: tuple[int, int] | None) -> bool:
    if preferred_domain is None:
        return False
    dom_start, dom_end = preferred_domain
    return any(dom_start <= r <= dom_end for r in residue_list)


def filter_pockets(pockets: list[dict], sasa_by_residue: dict[int, float], preferred_domain) -> list[dict]:
    survivors = []
    for pocket in pockets:
        druggability = pocket.get("druggability_score")
        if druggability is not None and druggability < DRUGGABILITY_THRESHOLD:
            continue

        if preferred_domain is not None and not domain_overlaps(pocket["residue_list"], preferred_domain):
            continue

        if pocket_is_buried(pocket["residue_list"], sasa_by_residue):
            continue

        volume = pocket.get("volume")
        if volume is not None and not (MIN_POCKET_VOLUME <= volume <= MAX_POCKET_VOLUME):
            continue

        survivors.append(pocket)
    return survivors


def normalized_volume_fit(volume: float | None) -> float:
    if volume is None:
        return 0.5
    # peptide-appropriate target range ~300-800 A^3; taper off outside it
    target_low, target_high = 300.0, 800.0
    if target_low <= volume <= target_high:
        return 1.0
    if volume < target_low:
        return max(0.0, volume / target_low)
    return max(0.0, 1.0 - (volume - target_high) / (MAX_POCKET_VOLUME - target_high))


def rank_pockets(pockets: list[dict], preferred_domain) -> list[dict]:
    scored = []
    for pocket in pockets:
        druggability = pocket.get("druggability_score") or 0.0
        vol_fit = normalized_volume_fit(pocket.get("volume"))
        match = 1.0 if domain_overlaps(pocket["residue_list"], preferred_domain) else 0.0
        score = druggability * 0.5 + vol_fit * 0.2 + match * 0.3
        scored.append({**pocket, "rank_score": score})
    scored.sort(key=lambda p: -p["rank_score"])
    return scored


def residue_sets_agree(a: list[int], b: list[int], min_overlap: float = 0.5) -> bool:
    set_a, set_b = set(a), set(b)
    if not set_a or not set_b:
        return False
    overlap = len(set_a & set_b) / len(set_a | set_b)
    return overlap >= min_overlap


def get_chain_atoms(structure: gemmi.Structure, chain_id: str):
    for chain in structure[0]:
        if chain.name == chain_id:
            for res in chain:
                for atom in res:
                    yield res.seqid.num, atom.pos


def compute_interface_contacts(local_file_path: str, chain_id: str, partner_chain_id: str, cutoff: float) -> list[int]:
    structure = gemmi.read_structure(local_file_path)
    structure.setup_entities()

    partner_positions = [pos for _, pos in get_chain_atoms(structure, partner_chain_id)]

    contact_residues = set()
    for resnum, pos in get_chain_atoms(structure, chain_id):
        if resnum in contact_residues:
            continue
        for partner_pos in partner_positions:
            if pos.dist(partner_pos) <= cutoff:
                contact_residues.add(resnum)
                break

    return sorted(contact_residues)


def detect_pocket(prepared_structure: dict, target_input: dict) -> dict:
    local_file_path = prepared_structure["local_file_path"]
    chain_id = prepared_structure["chain_id"]
    partner_chain_id = prepared_structure["partner_chain_id"]
    resnum_map = prepared_structure["uniprot_to_structure_resnum_map"]
    preferred_domain = target_input.get("preferred_domain")

    if partner_chain_id is not None:
        contact_residues = compute_interface_contacts(
            local_file_path, chain_id, partner_chain_id, INTERFACE_CONTACT_CUTOFF
        )
        return {
            "pocket_source": "interface",
            "pocket_id": None,
            "contact_residues": contact_residues,
            "contact_residues_uniprot": [resnum_map[r] for r in contact_residues if r in resnum_map],
            "druggability_score": None,
            "selection_method": "complex_interface",
            "flagged_ambiguous": False,
        }

    fpocket_results = run_fpocket_detection(local_file_path)
    p2rank_results = run_p2rank_detection(local_file_path)

    sasa_by_residue = compute_residue_sasa(local_file_path, chain_id)

    fpocket_filtered = filter_pockets(fpocket_results, sasa_by_residue, preferred_domain)
    fpocket_ranked = rank_pockets(fpocket_filtered, preferred_domain)

    if not fpocket_ranked:
        return {
            "pocket_source": "blind_pocket",
            "pocket_id": None,
            "contact_residues": [],
            "contact_residues_uniprot": [],
            "druggability_score": None,
            "selection_method": "fpocket",
            "flagged_ambiguous": True,
        }

    top_fpocket = fpocket_ranked[0]

    top_p2rank_residues = p2rank_results[0]["residue_list"] if p2rank_results else []
    agree = residue_sets_agree(top_fpocket["residue_list"], top_p2rank_residues)

    selection_method = "p2rank_consensus" if agree else "fpocket"

    return {
        "pocket_source": "blind_pocket",
        "pocket_id": top_fpocket["pocket_id"],
        "contact_residues": top_fpocket["residue_list"],
        "contact_residues_uniprot": [resnum_map[r] for r in top_fpocket["residue_list"] if r in resnum_map],
        "druggability_score": top_fpocket.get("druggability_score"),
        "selection_method": selection_method,
        "flagged_ambiguous": not agree,
    }


# ============================================================================
# 5. Hotspot identification
# ============================================================================

TARGET_REGISTRY_PATH = pathlib.Path("data/wound_healing_peptide_targets_structured.jsonl")
EVOEF2_EXE = pathlib.Path("tools/EvoEF2/EvoEF2.exe").resolve()

# Alanine-scanning convention: kcal/mol-scale "hot" cutoff
ALANINE_SCAN_DDG_THRESHOLD = 1.0
# ESM2 agreement threshold: esm_score >= this counts as "WT matters here"
ESM2_AGREEMENT_THRESHOLD = 0.5
# Typical anchor count for hotspot selection
ANCHOR_TOP_K = 6
# Advisory-only sanity check (flags flagged_ambiguous), not a hard selection constraint
MAX_PAIRWISE_CA_DISTANCE = 25.0
# Max CA-CA distance from seed anchor for another candidate to be admitted into anchor set
ANCHOR_CLUSTER_RADIUS = 14.0
# Per-residue SASA threshold (A^2) below which residues are treated as buried
SASA_EXPOSED_THRESHOLD = 20.0


def load_target_registry(path: pathlib.Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def find_registry_entry(registry: list[dict], target_input: dict) -> dict | None:
    """Match on gene_symbol/target_name against the registry's `genes` list first
    (exact, most reliable), then fall back to a token match against `target_name`
    (registry names are compound, e.g. "VEGFR2 / KDR" or "Tie2 / TEK")."""
    candidates = {target_input.get("gene_symbol", ""), target_input["target_name"]}
    candidates = {c.upper() for c in candidates if c}

    for entry in registry:
        registry_genes = {g.upper() for g in entry["target"]["genes"]}
        if candidates & registry_genes:
            return entry["target"]

    for entry in registry:
        name_tokens = {tok.upper() for tok in re.split(r"[\s/()]+", entry["target"]["target_name"]) if tok}
        if candidates & name_tokens:
            return entry["target"]

    return None


def get_literature_hotspots(target_record: dict, target_input: dict) -> list[dict]:
    """Only hotspots reported on the target's own molecule (not a binding partner)
    are usable as direct anchors on chain_id — partner-molecule hotspots describe
    the ligand, not contact_residues on our structure."""
    own_gene_names = {g.upper() for g in target_record["genes"]}
    own_gene_names.add(target_record["target_name"].upper())
    hotspots = []
    for h in target_record.get("hotspots", []):
        if h.get("molecule", "").upper() not in own_gene_names:
            continue
        for res in h.get("residues", []):
            hotspots.append({
                "uniprot_resnum": res["position"],
                "aa": res.get("aa"),
                "hotspot_strength": h.get("hotspot_strength"),
                "evidence_type": h.get("evidence_type"),
                "mutation": h.get("mutation"),
            })
    return hotspots


def run_evoef2(args: list[str], cwd: pathlib.Path) -> str:
    result = subprocess.run(
        [str(EVOEF2_EXE), *args],
        cwd=cwd, capture_output=True, text=True, timeout=120,
    )
    return result.stdout


def parse_evoef2_total(stdout: str) -> float:
    for line in stdout.splitlines():
        if line.strip().startswith("Total"):
            return float(line.split("=")[-1].strip())
    raise ValueError(f"EvoEF2 output had no Total line:\n{stdout}")


def evoef2_repair(local_file_path: str) -> pathlib.Path:
    path = pathlib.Path(local_file_path).resolve()
    run_evoef2(["--command=RepairStructure", f"--pdb={path.name}"], cwd=path.parent)
    return path.parent / f"{path.stem}_Repair.pdb"


def evoef2_binding_energy(pdb_path: pathlib.Path, chain_id: str, partner_chain_id: str) -> float:
    stdout = run_evoef2(
        ["--command=ComputeBinding", f"--split_chains={chain_id},{partner_chain_id}", f"--pdb={pdb_path.name}"],
        cwd=pdb_path.parent,
    )
    return parse_evoef2_total(stdout)


def evoef2_mutant_ddg(repaired_path: pathlib.Path, chain_id: str, partner_chain_id: str,
                       wt_aa: str, resnum: int, wt_binding: float) -> float:
    mutant_list_path = repaired_path.parent / f"mutlist_{chain_id}{resnum}.txt"
    mutant_list_path.write_text(f"{wt_aa}{chain_id}{resnum}A;\n")

    run_evoef2(
        ["--command=BuildMutant", f"--pdb={repaired_path.name}", f"--mutant_file={mutant_list_path.name}"],
        cwd=repaired_path.parent,
    )
    mutant_path = repaired_path.parent / f"{repaired_path.stem}_Model_0001.pdb"
    mutant_binding = evoef2_binding_energy(mutant_path, chain_id, partner_chain_id)
    return mutant_binding - wt_binding


_esm_model = None
_esm_alphabet = None


def get_esm_model():
    global _esm_model, _esm_alphabet
    if _esm_model is None:
        _esm_model, _esm_alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        _esm_model.eval()
    return _esm_model, _esm_alphabet


def esm2_masked_marginal_score(sequence: str, seq_position_0idx: int) -> float:
    """log P(wildtype | context) - log P(Ala | context) at a masked position.
    Positive = model thinks the wildtype residue matters more than Ala here."""
    model, alphabet = get_esm_model()
    batch_converter = alphabet.get_batch_converter()
    _, _, tokens = batch_converter([("query", sequence)])

    tok_idx = seq_position_0idx + 1  # +1 for BOS
    wt_aa = sequence[seq_position_0idx]
    masked_tokens = tokens.clone()
    masked_tokens[0, tok_idx] = alphabet.mask_idx

    with torch.no_grad():
        logits = model(masked_tokens, repr_layers=[], return_contacts=False)["logits"]
    log_probs = torch.log_softmax(logits, dim=-1)

    wt_tok = alphabet.get_idx(wt_aa)
    ala_tok = alphabet.get_idx("A")
    return (log_probs[0, tok_idx, wt_tok] - log_probs[0, tok_idx, ala_tok]).item()


def get_chain_sequence_and_map(local_file_path: str, chain_id: str) -> tuple[str, dict[int, int]]:
    """Returns (one-letter sequence, {structure_resnum: 0-indexed position in sequence})."""
    structure = gemmi.read_structure(local_file_path)
    structure.setup_entities()
    poly = structure[0][chain_id].get_polymer()
    seq = gemmi.one_letter_code(poly.extract_sequence())
    resnum_to_pos = {res.seqid.num: i for i, res in enumerate(poly)}
    return seq.upper(), resnum_to_pos


def alanine_scan_ensemble(prepared_structure: dict, contact_residues: list[int]) -> list[dict]:
    local_file_path = prepared_structure["local_file_path"]
    chain_id = prepared_structure["chain_id"]
    partner_chain_id = prepared_structure["partner_chain_id"]

    repaired_path = evoef2_repair(local_file_path)
    wt_binding = evoef2_binding_energy(repaired_path, chain_id, partner_chain_id)

    seq, resnum_to_pos = get_chain_sequence_and_map(str(repaired_path), chain_id)

    results = []
    for resnum in contact_residues:
        if resnum not in resnum_to_pos:
            continue
        seq_pos = resnum_to_pos[resnum]
        wt_aa = seq[seq_pos]
        if wt_aa == "A":
            continue  # already alanine, scanning it is meaningless

        ddg_evoef2 = evoef2_mutant_ddg(repaired_path, chain_id, partner_chain_id, wt_aa, resnum, wt_binding)
        esm_score = esm2_masked_marginal_score(seq, seq_pos)

        evoef2_hot = ddg_evoef2 >= ALANINE_SCAN_DDG_THRESHOLD
        esm_hot = esm_score >= ESM2_AGREEMENT_THRESHOLD
        agree = evoef2_hot and esm_hot

        results.append({
            "structure_resnum": resnum,
            "wt_aa": wt_aa,
            "ddg_evoef2": ddg_evoef2,
            "esm2_score": esm_score,
            "evoef2_hot": evoef2_hot,
            "esm2_hot": esm_hot,
            "agree": agree,
            "is_hotspot": evoef2_hot or esm_hot,
            "confidence": "medium-high" if agree else "medium",
        })

    return results


def burial_proxy_scan(prepared_structure: dict, contact_residues: list[int]) -> list[dict]:
    """Apo case, no partner to scan against: rank pocket-lining residues by burial alone.
    'Burial' here means engagement in the pocket relative to a fully solvent-exposed
    residue — NOT literal SASA=0. A residue with ~0 SASA is buried in the protein core,
    not contactable by a peptide binding the pocket surface, so it's excluded outright
    rather than ranked highest (the naive 1 - sasa/max_sasa formula would otherwise favor
    exactly the wrong residues — the surface-exposed check downstream would then just
    discard them). Conservation term intentionally omitted for now (no MSA pipeline
    wired up) — see get_conservation_score() TODO below."""
    local_file_path = prepared_structure["local_file_path"]
    chain_id = prepared_structure["chain_id"]

    sasa_by_residue = compute_residue_sasa(local_file_path, chain_id)
    exposed_sasas = [s for s in sasa_by_residue.values() if s > SASA_EXPOSED_THRESHOLD]
    max_sasa = max(exposed_sasas) if exposed_sasas else 1.0

    results = []
    for resnum in contact_residues:
        sasa = sasa_by_residue.get(resnum, 0.0)
        if sasa <= SASA_EXPOSED_THRESHOLD:
            normalized_burial = 0.0  # core-buried, not pocket-contactable — excluded from ranking
        else:
            normalized_burial = 1.0 - min(sasa / max_sasa, 1.0)

        conservation_score = get_conservation_score(resnum)
        conservation_term = conservation_score if conservation_score is not None else 0.0
        weight = 0.5 if conservation_score is not None else 1.0

        score_residue = weight * normalized_burial + (0.5 * conservation_term if conservation_score is not None else 0.0)

        results.append({
            "structure_resnum": resnum,
            "sasa": sasa,
            "normalized_burial": normalized_burial,
            "conservation_score": conservation_score,
            "score_residue": score_residue,
        })

    results.sort(key=lambda r: -r["score_residue"])
    return results


def get_conservation_score(structure_resnum: int) -> float | None:
    # TODO: wire up an MSA pipeline (e.g. HMMER/UniRef search + ConSurf-style scoring)
    # to score evolutionary conservation at this residue. Returns None until then,
    # in which case burial_proxy_scan() falls back to burial-only scoring.
    return None


def get_ca_position(structure: gemmi.Structure, chain_id: str, resnum: int) -> gemmi.Position | None:
    for res in structure[0][chain_id]:
        if res.seqid.num == resnum:
            atom = res.find_atom("CA", "*")
            return atom.pos if atom else None
    return None


def spatial_clustering_ok(local_file_path: str, chain_id: str, resnums: list[int]) -> tuple[bool, float]:
    structure = gemmi.read_structure(local_file_path)
    structure.setup_entities()
    positions = [p for p in (get_ca_position(structure, chain_id, r) for r in resnums) if p is not None]

    max_dist = 0.0
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            max_dist = max(max_dist, positions[i].dist(positions[j]))

    return max_dist <= MAX_PAIRWISE_CA_DISTANCE, max_dist


def select_spatially_clustered_anchors(
    local_file_path: str, chain_id: str, scored_candidates: list[dict], resnum_key: str, top_k: int,
) -> list[dict]:
    """Greedily builds a spatially coherent anchor set instead of taking the global top-K by
    score alone. Ranking candidates by score (burial, ddG, ESM2 agreement, ...) with no spatial
    term can select residues that individually look great but sit spread across the protein
    surface -- a short 8-40 residue peptide cannot straddle anchors that are, say, 15-18 A apart
    on a folded surface without its backbone cutting through the intervening protein mass to
    reach both.

    Each admitted candidate must be within ANCHOR_CLUSTER_RADIUS of EVERY already-admitted
    member, not just the seed -- checking only "close to the seed" lets the cluster sprawl into
    a star shape (two candidates both ~12-13 A from the seed but ~20+ A from each other would
    both pass a seed-only check while sitting on opposite sides of the seed), which is exactly
    as unreachable for a short linear peptide as the original unclustered top-K was."""
    structure = gemmi.read_structure(local_file_path)
    structure.setup_entities()

    candidates = [
        {**c, "_pos": get_ca_position(structure, chain_id, c[resnum_key])}
        for c in scored_candidates
    ]
    candidates = [c for c in candidates if c["_pos"] is not None]
    if not candidates:
        return []

    seed = candidates[0]  # candidates are assumed pre-sorted best-first by the caller
    cluster = [seed]
    for c in candidates[1:]:
        if len(cluster) >= top_k:
            break
        if all(c["_pos"].dist(m["_pos"]) <= ANCHOR_CLUSTER_RADIUS for m in cluster):
            cluster.append(c)

    for c in cluster:
        del c["_pos"]
    return cluster


def surface_exposed_resnums(local_file_path: str, chain_id: str, resnums: list[int]) -> set[int]:
    sasa_by_residue = compute_residue_sasa(local_file_path, chain_id)
    return {r for r in resnums if sasa_by_residue.get(r, 0.0) > SASA_EXPOSED_THRESHOLD}


def identify_hotspots(prepared_structure: dict, pocket_record: dict, target_input: dict) -> dict:
    contact_residues = pocket_record["contact_residues"]
    partner_chain_id = prepared_structure["partner_chain_id"]
    resnum_map = prepared_structure["uniprot_to_structure_resnum_map"]
    local_file_path = prepared_structure["local_file_path"]
    chain_id = prepared_structure["chain_id"]

    flagged_ambiguous = False
    ddg_values = None

    # Step 1a — literature hotspots (highest priority)
    registry = load_target_registry(TARGET_REGISTRY_PATH)
    target_record = find_registry_entry(registry, target_input)
    literature_hotspots = get_literature_hotspots(target_record, target_input) if target_record else []

    inverse_resnum_map = {v: k for k, v in resnum_map.items()}
    lit_anchor_candidates = []
    for h in literature_hotspots:
        struct_resnum = inverse_resnum_map.get(h["uniprot_resnum"])
        if struct_resnum is not None:
            lit_anchor_candidates.append({**h, "structure_resnum": struct_resnum})

    lit_on_surface = [h for h in lit_anchor_candidates if h["structure_resnum"] in contact_residues]

    if lit_anchor_candidates and lit_on_surface:
        evidence_source = "literature"
        confidence = "high"
        strength_rank = {"very_strong": 3, "strong": 2, "moderate": 1}
        lit_on_surface.sort(key=lambda h: -strength_rank.get(h.get("hotspot_strength"), 0))
        # Literature hotspots are pooled from independent papers/domains with no guarantee
        # they sit anywhere near each other in 3D -- unlike the evoef2/proxy branches below,
        # this path used to take the top-K by strength alone with no spatial check, which
        # could (and did) hand RFdiffusion anchors ~180 residues apart on the sequence and
        # nowhere near clustered in space, something no 8-40 residue peptide can contact
        # >=60% of at once. Cluster around the strongest hit (already first after the sort
        # above) exactly as select_spatially_clustered_anchors() does for the other two
        # evidence sources, instead of leaving this one evidence source unclustered.
        clustered_lit = select_spatially_clustered_anchors(
            local_file_path, chain_id, lit_on_surface, "structure_resnum", ANCHOR_TOP_K
        )
        anchor_residues = [
            {"structure_resnum": h["structure_resnum"], "uniprot_resnum": h["uniprot_resnum"], "chain_id": chain_id}
            for h in clustered_lit
        ]
    elif lit_anchor_candidates and not lit_on_surface:
        # literature residues exist but don't land on the detected pocket/interface —
        # wrong pocket, wrong isoform/state, or stale annotation. Flag and fall through.
        flagged_ambiguous = True
        evidence_source = None
        anchor_residues = []
    else:
        evidence_source = None
        anchor_residues = []

    # Step 1b / 1c — no usable literature hotspots
    if not anchor_residues:
        if partner_chain_id is not None:
            scan_results = alanine_scan_ensemble(prepared_structure, contact_residues)
            hits = [r for r in scan_results if r["is_hotspot"]]
            hits.sort(key=lambda r: (-r["agree"], -r["ddg_evoef2"]))
            ddg_values = {r["structure_resnum"]: r["ddg_evoef2"] for r in scan_results}

            evidence_source = "evoef2_esm2_ensemble"
            clustered_hits = select_spatially_clustered_anchors(
                local_file_path, chain_id, hits, "structure_resnum", ANCHOR_TOP_K
            )
            any_agree = any(r["agree"] for r in clustered_hits)
            confidence = "medium-high" if any_agree else "low"

            anchor_residues = [
                {
                    "structure_resnum": r["structure_resnum"],
                    "uniprot_resnum": resnum_map.get(r["structure_resnum"]),
                    "chain_id": chain_id,
                }
                for r in clustered_hits
            ]
        else:
            proxy_results = burial_proxy_scan(prepared_structure, contact_residues)
            evidence_source = "computational_proxy"
            confidence = "low"
            clustered_proxy = select_spatially_clustered_anchors(
                local_file_path, chain_id, proxy_results, "structure_resnum", ANCHOR_TOP_K
            )
            anchor_residues = [
                {
                    "structure_resnum": r["structure_resnum"],
                    "uniprot_resnum": resnum_map.get(r["structure_resnum"]),
                    "chain_id": chain_id,
                }
                for r in clustered_proxy
            ]

    # Step 3 — sanity checks
    anchor_resnums = [a["structure_resnum"] for a in anchor_residues]
    flagged_low_confidence = confidence == "low"

    if anchor_resnums:
        clustered_ok, max_pairwise_dist = spatial_clustering_ok(local_file_path, chain_id, anchor_resnums)
        if not clustered_ok:
            flagged_ambiguous = True

        exposed = surface_exposed_resnums(local_file_path, chain_id, anchor_resnums)
        anchor_residues = [a for a in anchor_residues if a["structure_resnum"] in exposed]

    return {
        "anchor_residues": anchor_residues,
        "evidence_source": evidence_source,
        "confidence": confidence,
        "ddG_values": ddg_values,
        "flagged_ambiguous": flagged_ambiguous,
        "flagged_low_confidence": flagged_low_confidence,
    }


# ============================================================================
# 6. Backbone generation (RFdiffusion)
# ============================================================================
#
# Not installed by default — written against RFdiffusion's real run_inference.py
# Hydra CLI (contig syntax, ppi.hotspot_res, .trb metadata format), run via Docker.
# Linear topology uses RFdiffusion's standard PPI/hotspot-conditioned binder design
# mode. Cyclic topology (RFpeptides) is a distinct, less mature path and raises
# rather than silently falling back to linear. See module docstring for install refs.

RFDIFFUSION_DIR = pathlib.Path("tools/RFdiffusion")
RFPEPTIDES_DIR = pathlib.Path("tools/RFpeptides")  # Zenodo release, not mainline RFdiffusion —
                                                     # see run_rfpeptides_cyclic()'s docstring
RFDIFFUSION_DOCKER_IMAGE = "rfdiffusion"  # built from tools/RFdiffusion/docker/Dockerfile
RFDIFFUSION_COMPLEX_CKPT_CONTAINER = "/app/RFdiffusion/models/Complex_base_ckpt.pt"  # in-container path

DIFFUSION_STEPS_DEFAULT = 25   # final candidates -- lowered from 50 for speed; quality tradeoff
DIFFUSION_STEPS_TRIAGE = 15    # cheap first-pass batches -- RFdiffusion hard-asserts T >= 15

RG_MIN, RG_MAX = 4.0, 20.0      # Angstrom, radius-of-gyration sanity bounds for an 8-40 residue peptide
CLASH_DISTANCE = 2.0            # Angstrom, backbone heavy-atom clash cutoff against target
CONTACT_DISTANCE = 8.0          # Angstrom, anchor-contact cutoff
CONTACT_ANCHOR_FRACTION = 0.6   # must sit near >=60% of anchor_residues.
                                 # Reverted to 0.6 after fixing the actual root cause upstream:
                                 # hotspot anchor selection previously picked top-scored residues
                                 # with no spatial-clustering constraint, producing anchor sets
                                 # spread up to ~17.6 A apart -- too far for a short peptide to
                                 # contact most of them without cutting through the target (see
                                 # select_spatially_clustered_anchors() / ANCHOR_CLUSTER_RADIUS).
                                 # 0.6 is only realistic now that anchors are constrained to sit
                                 # within one reachable local patch; re-lower this only if a
                                 # properly clustered anchor set still can't clear it.


def docker_path(host_path: pathlib.Path) -> str:
    """Docker Desktop (Windows) accepts host paths on the -v flag directly and translates
    them, but the in-container path we reference in later args must be a POSIX path under
    wherever we mounted it — so every host path that crosses into the container needs both
    forms tracked together, not just converted once."""
    return str(host_path.resolve()).replace("\\", "/")


def build_contig_string(prepared_structure: dict, peptide_length_range: tuple[int, int]) -> str:
    """RFdiffusion contig syntax: '<chain><start>-<end>/0 <min>-<max>' — fixed target region,
    a chain break, then a sampled-length designable segment. Target length here spans the full
    resolved chain; RFdiffusion samples the binder length uniformly from the given range per trajectory."""
    chain_id = prepared_structure["chain_id"]
    resnums = sorted(prepared_structure["uniprot_to_structure_resnum_map"].keys())
    target_start, target_end = resnums[0], resnums[-1]
    min_len, max_len = peptide_length_range
    return f"{chain_id}{target_start}-{target_end}/0 {min_len}-{max_len}"


def build_hotspot_res_list(hotspot_record: dict) -> list[str]:
    return [f"{a['chain_id']}{a['structure_resnum']}" for a in hotspot_record["anchor_residues"]]


def run_rfdiffusion_linear(
    prepared_structure: dict,
    hotspot_record: dict,
    target_input: dict,
    n_designs: int,
    n_diffusion_steps: int,
    output_dir: pathlib.Path,
) -> list[dict]:
    """RFdiffusion PPI/hotspot-conditioned binder design mode (linear backbones only).
    Real Hydra CLI, verified against examples/design_ppi.sh in the RFdiffusion repo:

        run_inference.py inference.output_prefix=<prefix> inference.input_pdb=<pdb>
            'contigmap.contigs=[<contig>]' 'ppi.hotspot_res=[<hotspots>]'
            inference.num_designs=<n> diffuser.T=<steps>
            denoiser.noise_scale_ca=0 denoiser.noise_scale_frame=0

    denoiser.noise_scale_*=0 is the repo's own recommendation for PPI/binder design —
    lower-noise, more deterministic trajectories than the default de novo monomer setting.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = output_dir / "design"

    contig = build_contig_string(prepared_structure, target_input["peptide_length_range"])
    hotspots = build_hotspot_res_list(hotspot_record)

    # Mount the output dir and the input PDB's parent dir separately: they don't share a
    # common host ancestor in general (backbone_cache/ vs. wherever structure prep wrote
    # the prepared PDB), so a single bind mount can't cover both.
    input_pdb_host = pathlib.Path(prepared_structure["local_file_path"]).resolve()
    output_dir_host = output_dir.resolve()
    models_dir_host = (RFDIFFUSION_DIR / "models").resolve()

    input_pdb_container = f"/mnt/input/{input_pdb_host.name}"
    output_prefix_container = f"/mnt/output/{output_prefix.name}"

    # Named (not auto-named) so a cancelled/crashed run can still be found and stopped
    # afterward -- "docker run" without -d blocks in the foreground, but killing the
    # parent subprocess.run() call does not send docker a stop signal, so the container
    # itself keeps running orphaned on the daemon otherwise.
    container_name = f"rfdiffusion_{prepared_structure['structure_id']}_{output_dir.name}".replace("/", "_")

    # A crashed/interrupted previous run can leave this same name held by an exited or
    # still-running container (--rm only removes it on a clean exit) -- "docker run
    # --name" then fails outright with a Conflict error before ever starting, rather than
    # just reusing/replacing it. Force-remove any stale container under this name first;
    # -f also stops it if still running. Harmless no-op if nothing is there.
    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, text=True, check=False)

    cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "--name", container_name,
        "-v", f"{docker_path(input_pdb_host.parent)}:/mnt/input",
        "-v", f"{docker_path(output_dir_host)}:/mnt/output",
        "-v", f"{docker_path(models_dir_host)}:/app/RFdiffusion/models",
        RFDIFFUSION_DOCKER_IMAGE,
        f"inference.output_prefix={output_prefix_container}",
        f"inference.input_pdb={input_pdb_container}",
        f"inference.ckpt_override_path={RFDIFFUSION_COMPLEX_CKPT_CONTAINER}",
        f"contigmap.contigs=[{contig}]",
        f"ppi.hotspot_res=[{','.join(hotspots)}]",
        f"inference.num_designs={n_designs}",
        f"diffuser.T={n_diffusion_steps}",
        "denoiser.noise_scale_ca=0",
        "denoiser.noise_scale_frame=0",
    ]
    try:
        run_checked(cmd)
    finally:
        # Always attempt cleanup, not just on a caught exception -- an interrupt delivers
        # KeyboardInterrupt on the next bytecode boundary after the blocking subprocess.run()
        # call returns, which is not guaranteed to be "mid-wait" in every Python/OS
        # combination, so catching specific exceptions here is not reliable. "docker stop"
        # on a container that already exited and self-removed (--rm) just fails harmlessly
        # (check=False swallows it), so this is safe to run unconditionally every time
        # rather than only in a failure path.
        subprocess.run(["docker", "stop", container_name], capture_output=True, text=True, check=False)

    backbones = []
    for i in range(n_designs):
        pdb_path = output_dir / f"design_{i}.pdb"
        trb_path = output_dir / f"design_{i}.trb"
        with open(trb_path, "rb") as f:
            trb = pickle.load(f)

        backbones.append({
            "backbone_id": f"{prepared_structure['structure_id']}_bb{i}",
            "target_id": prepared_structure["structure_id"],
            "coordinates_file": str(pdb_path),
            "contig_used": trb.get("sampled_mask", contig),
            "hotspot_residues_targeted": hotspots,
            "n_diffusion_steps": n_diffusion_steps,
            "trajectory_seed": trb.get("seed"),
            "con_hal_pdb_idx": trb.get("con_hal_pdb_idx"),
            "con_ref_pdb_idx": trb.get("con_ref_pdb_idx"),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        })
    return backbones


def run_rfpeptides_cyclic(*args, **kwargs):
    """Cyclic backbone design via RFpeptides (Rettie, Juergens, Adebomi et al.,
    Nat Chem Biol 2025 — 'Accurate de novo design of high-affinity protein-binding
    macrocycles using deep learning'). Not a separate tool: a modified cyclic
    relative-position-encoding mode layered onto RFdiffusion + RoseTTAFold2, run
    alongside ProteinMPNN. BSD-3-Clause (same license as RFdiffusion itself, which
    explicitly covers the model weights too).

    CAVEAT: as of writing, RFdiffusion GitHub issue #333 (opened March 2025, still
    open) reports the paper's described cyclic-mode code does not appear to be in
    the mainline RFdiffusion repo despite the supplement's claim it was merged.
    A separate Zenodo release (doi.org/10.5281/zenodo.15264344) is cited as an
    alternative source. Target that Zenodo release when actually installing this —
    do not assume mainline `run_inference.py` has cyclic contig support without
    checking first.
    """
    raise NotImplementedError(
        "Cyclic backbone generation (RFpeptides) is not implemented — its integration "
        "point in RFdiffusion is unconfirmed (see docstring). Implement once the Zenodo "
        "release's actual CLI/config surface has been verified, rather than guessing at "
        "flag names the way run_rfdiffusion_linear() could be verified against a real example."
    )


def _read_cached_backbones(output_dir: pathlib.Path, prepared_structure: dict, hotspot_record: dict,
                            target_input: dict, n_designs: int, n_diffusion_steps: int) -> list[dict]:
    """Rebuilds the same backbone-record shape run_rfdiffusion_linear() returns, from
    design_*.pdb/.trb files already sitting in output_dir -- used only on a debug-cache hit,
    so we don't have to invoke Docker again to re-derive records we already have on disk."""
    contig = build_contig_string(prepared_structure, target_input["peptide_length_range"])
    hotspots = build_hotspot_res_list(hotspot_record)

    backbones = []
    for i in range(n_designs):
        pdb_path = output_dir / f"design_{i}.pdb"
        trb_path = output_dir / f"design_{i}.trb"
        with open(trb_path, "rb") as f:
            trb = pickle.load(f)

        backbones.append({
            "backbone_id": f"{prepared_structure['structure_id']}_bb{i}",
            "target_id": prepared_structure["structure_id"],
            "coordinates_file": str(pdb_path),
            "contig_used": trb.get("sampled_mask", contig),
            "hotspot_residues_targeted": hotspots,
            "n_diffusion_steps": n_diffusion_steps,
            "trajectory_seed": trb.get("seed"),
            "con_hal_pdb_idx": trb.get("con_hal_pdb_idx"),
            "con_ref_pdb_idx": trb.get("con_ref_pdb_idx"),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        })
    return backbones


def generate_backbones(
    prepared_structure: dict,
    hotspot_record: dict,
    target_input: dict,
    n_designs_requested: int,
    triage: bool = True,
    use_cache: bool = False,
) -> list[dict]:
    """use_cache=False (default): every call gets its own fresh, never-reused output_dir --
    the only way to guarantee two runs (different inputs, or two concurrent processes) can
    never silently overwrite each other's design_*.pdb/.trb files.

    use_cache=True: opt-in fast-iteration mode for once the pipeline is trusted. Keys
    output_dir on a hash of the inputs that actually determine RFdiffusion's output (contig,
    hotspots, topology, n_steps, n_designs) instead of a timestamp, so re-running with the
    *exact* same inputs reuses the existing designs instead of paying for another GPU run.
    Reuse only fires if that exact directory already holds >= n_designs_requested design
    files -- anything less is treated as an interrupted/partial prior run and regenerated
    fresh into the same (still input-specific, so still collision-safe) directory, never
    silently topped up or mixed with a different config's output.
    """
    topology = target_input["backbone_topology"]
    n_steps = DIFFUSION_STEPS_TRIAGE if triage else DIFFUSION_STEPS_DEFAULT

    if use_cache:
        contig_for_key = build_contig_string(prepared_structure, target_input["peptide_length_range"])
        hotspots_for_key = build_hotspot_res_list(hotspot_record)
        run_key_material = repr((
            contig_for_key, hotspots_for_key, topology, n_steps, n_designs_requested,
        )).encode("utf-8")
        run_tag = "cache_" + hashlib.sha1(run_key_material).hexdigest()[:12]
    else:
        run_tag = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%f")

    output_dir = pathlib.Path("backbone_cache") / prepared_structure["structure_id"] / run_tag

    if use_cache and output_dir.exists():
        existing = sorted(output_dir.glob("design_*.pdb"))
        if len(existing) >= n_designs_requested:
            logger.info(
                "routec.backbone.cache_hit",
                extra={"output_dir": str(output_dir), "n_cached": len(existing)},
            )
            backbones = _read_cached_backbones(
                output_dir, prepared_structure, hotspot_record, target_input,
                n_designs_requested, n_steps,
            )
            for b in backbones:
                b["hotspot_confidence"] = hotspot_record["confidence"]
            return backbones
        else:
            logger.info(
                "routec.backbone.cache_partial",
                extra={"output_dir": str(output_dir), "n_cached": len(existing), "n_requested": n_designs_requested},
            )

    if topology == "linear":
        backbones = run_rfdiffusion_linear(
            prepared_structure, hotspot_record, target_input,
            n_designs_requested, n_steps, output_dir,
        )
    elif topology == "cyclic":
        backbones = run_rfpeptides_cyclic(
            prepared_structure, hotspot_record, target_input,
            n_designs_requested, n_steps, output_dir,
        )
    else:
        raise ValueError(f"Unknown backbone_topology: {topology!r}")

    for b in backbones:
        b["hotspot_confidence"] = hotspot_record["confidence"]  # carried through for downstream scrutiny
    return backbones


def _peptide_chain(structure: gemmi.Structure, target_chain_id: str):
    """RFdiffusion writes out the full PPI complex per design (fixed target chain +
    newly-diffused binder chain), not just the generated backbone -- so every geometric
    check here must operate on the binder chain alone. Selecting "whichever chain isn't
    the target's" rather than hardcoding a letter, since RFdiffusion's own chain lettering
    for the generated segment isn't guaranteed stable across versions/runs."""
    other_chains = [chain for chain in structure[0] if chain.name != target_chain_id]
    if len(other_chains) != 1:
        names = [c.name for c in structure[0]]
        raise ValueError(
            f"Expected exactly one non-target chain (the generated peptide) alongside "
            f"target chain '{target_chain_id}', found chains {names}"
        )
    return other_chains[0]


def radius_of_gyration(local_file_path: str, target_chain_id: str) -> float:
    structure = gemmi.read_structure(local_file_path)
    structure.setup_entities()
    peptide = _peptide_chain(structure, target_chain_id)
    positions = [atom.pos for res in peptide for atom in res if atom.name == "CA"]
    if not positions:
        return 0.0
    cx = sum(p.x for p in positions) / len(positions)
    cy = sum(p.y for p in positions) / len(positions)
    cz = sum(p.z for p in positions) / len(positions)
    centroid = gemmi.Position(cx, cy, cz)
    return (sum(p.dist(centroid) ** 2 for p in positions) / len(positions)) ** 0.5


BACKBONE_ATOM_NAMES = {"N", "CA", "C", "O"}


def backbone_clashes_with_target(backbone_file: str, target_file: str, target_chain_id: str,
                                   cutoff: float = CLASH_DISTANCE) -> bool:
    """Backbone-atom-only clash check, deliberately not full-atom. At this stage RFdiffusion
    has only generated backbone geometry (peptide residues come back as placeholder glycines --
    real side chains don't exist until ProteinMPNN); comparing that bare backbone against the
    target's fully-sidechained structure produces sub-1A "clashes" against target side-chain
    atoms that have no peptide-side counterpart yet and are not a real defect. Backbone-vs-backbone
    overlap, though, is a genuine and unrecoverable failure (no amount of later sequence design
    or relaxation un-threads two overlapping backbones), so it's still worth catching cheaply
    here rather than deferring all sterics to after refinement."""
    bb = gemmi.read_structure(backbone_file)
    bb.setup_entities()
    tgt = gemmi.read_structure(target_file)
    tgt.setup_entities()

    peptide = _peptide_chain(bb, target_chain_id)
    bb_atoms = [atom.pos for res in peptide for atom in res if atom.name in BACKBONE_ATOM_NAMES]
    tgt_atoms = [atom.pos for chain in tgt[0] for res in chain for atom in res if atom.name in BACKBONE_ATOM_NAMES]

    for bp in bb_atoms:
        for tp in tgt_atoms:
            if bp.dist(tp) < cutoff:
                return True
    return False


def backbone_anchor_contact_fraction(backbone_file: str, target_file: str, target_chain_id: str,
                                       anchor_resnums: list[int], cutoff: float = CONTACT_DISTANCE) -> float:
    bb = gemmi.read_structure(backbone_file)
    bb.setup_entities()
    tgt = gemmi.read_structure(target_file)
    tgt.setup_entities()

    peptide = _peptide_chain(bb, target_chain_id)
    bb_atoms = [atom.pos for res in peptide for atom in res]

    contacted = 0
    unmatched_anchors = []
    for resnum in anchor_resnums:
        found = False
        for res in tgt[0][target_chain_id]:
            if res.seqid.num != resnum:
                continue
            found = True
            anchor_atoms = [atom.pos for atom in res]
            if any(a.dist(b) <= cutoff for a in anchor_atoms for b in bb_atoms):
                contacted += 1
            break
        if not found:
            unmatched_anchors.append(resnum)

    if unmatched_anchors:
        # Anchor resnums that don't exist on target_chain_id can never register a contact --
        # silently drags the fraction down (e.g. a UniProt-vs-structure numbering mismatch
        # feeding hotspot_record). Surface it instead of letting it masquerade as "no contact".
        logger.warning(
            "routec.backbone.anchors_not_found",
            extra={"target_chain_id": target_chain_id, "unmatched_anchors": unmatched_anchors},
        )

    return contacted / len(anchor_resnums) if anchor_resnums else 0.0


def filter_backbones(backbones: list[dict], prepared_structure: dict, hotspot_record: dict,
                      verbose: bool = True) -> list[dict]:
    """Cheap pre-ProteinMPNN geometric filter with rejection diagnostics."""
    target_file = prepared_structure["local_file_path"]
    target_chain_id = prepared_structure["chain_id"]
    anchor_resnums = [a["structure_resnum"] for a in hotspot_record["anchor_residues"]]

    rejection_counts = {"rg": 0, "clash": 0, "contact": 0}
    rejection_detail = []

    survivors = []
    for b in backbones:
        rg = radius_of_gyration(b["coordinates_file"], target_chain_id)
        if not (RG_MIN <= rg <= RG_MAX):
            rejection_counts["rg"] += 1
            rejection_detail.append({"backbone_id": b["backbone_id"], "gate": "rg", "radius_of_gyration": rg})
            if verbose:
                logger.info(
                    "routec.backbone.reject",
                    extra={"backbone_id": b["backbone_id"], "gate": "rg", "rg": round(rg, 2), "bounds": [RG_MIN, RG_MAX]},
                )
            continue

        if backbone_clashes_with_target(b["coordinates_file"], target_file, target_chain_id):
            rejection_counts["clash"] += 1
            rejection_detail.append({"backbone_id": b["backbone_id"], "gate": "clash", "radius_of_gyration": rg})
            if verbose:
                logger.info(
                    "routec.backbone.reject",
                    extra={"backbone_id": b["backbone_id"], "gate": "clash", "clash_distance": CLASH_DISTANCE, "rg": round(rg, 2)},
                )
            continue

        contact_fraction = backbone_anchor_contact_fraction(
            b["coordinates_file"], target_file, target_chain_id, anchor_resnums
        )
        if contact_fraction < CONTACT_ANCHOR_FRACTION:
            rejection_counts["contact"] += 1
            rejection_detail.append({
                "backbone_id": b["backbone_id"], "gate": "contact",
                "radius_of_gyration": rg, "anchor_contact_fraction": contact_fraction,
            })
            if verbose:
                logger.info(
                    "routec.backbone.reject",
                    extra={
                        "backbone_id": b["backbone_id"], "gate": "contact",
                        "contact_fraction": round(contact_fraction, 2), "threshold": CONTACT_ANCHOR_FRACTION,
                        "n_anchors": len(anchor_resnums), "rg": round(rg, 2),
                    },
                )
            continue

        b["radius_of_gyration"] = rg
        b["anchor_contact_fraction"] = contact_fraction
        survivors.append(b)
        if verbose:
            logger.info(
                "routec.backbone.pass",
                extra={"backbone_id": b["backbone_id"], "rg": round(rg, 2), "contact_fraction": round(contact_fraction, 2)},
            )

    if verbose:
        n = len(backbones)
        logger.info(
            "routec.backbone.filter_summary",
            extra={
                "total": n,
                "survived": len(survivors),
                "rejection_counts": rejection_counts,
                "anchor_resnums": anchor_resnums,
            },
        )

    filter_backbones.last_run_diagnostics = {
        "rejection_counts": rejection_counts,
        "rejection_detail": rejection_detail,
        "anchor_resnums": anchor_resnums,
    }
    return survivors


# ============================================================================
# 7. Sequence design (ProteinMPNN)
# ============================================================================
#
# Not installed by default — written against ProteinMPNN's real single-PDB CLI
# (protein_mpnn_run.py --pdb_path --pdb_path_chains) and its actual FASTA output
# format (score/global_score in the header). Applies the shared physicochemical
# filter (from routeA) plus a ProteinMPNN-confidence cutoff before AF-Multimer.

PROTEINMPNN_DIR = pathlib.Path("tools/ProteinMPNN")

SAMPLING_TEMP_DEFAULT = 0.1     # conservative; raise toward 0.3 for more diverse sampling
N_SEQUENCES_PER_BACKBONE = 6     # 4-8 per spec
MEAN_CONFIDENCE_THRESHOLD = 0.5  # drop low-confidence ProteinMPNN outputs early — tune per target


def binder_chain_id(backbone_coordinates_file: str, target_chain_id: str) -> str:
    """RFdiffusion emits the fixed target chain plus a second chain for the designed
    binder — the binder is whichever chain in the output isn't the target's own id."""
    structure = gemmi.read_structure(backbone_coordinates_file)
    structure.setup_entities()
    chain_ids = [c.name for c in structure[0]]
    others = [c for c in chain_ids if c != target_chain_id]
    if len(others) != 1:
        raise ValueError(f"Expected exactly one non-target chain in {backbone_coordinates_file}, got {others}")
    return others[0]


def parse_proteinmpnn_fasta(fasta_path: pathlib.Path) -> list[dict]:
    """ProteinMPNN writes seqs/<name>.fa with per-sequence headers carrying inline scores:
    '>T=0.1, sample=1, score=1.2345, global_score=1.5678, seq_recovery=0.4321'
    (fields confirmed against the real ProteinMPNN output format). First record is the
    input/native sequence used as context, not a design — skipped."""
    text = fasta_path.read_text()
    records = [r for r in text.split(">")[1:] if r.strip()]

    parsed = []
    for i, record in enumerate(records):
        if i == 0:
            continue  # native/input sequence, not a ProteinMPNN design
        header, seq = record.split("\n", 1)
        seq = seq.strip()

        score_match = re.search(r"score=([\d.]+)", header)
        sample_match = re.search(r"sample=(\d+)", header)
        temp_match = re.search(r"T=([\d.]+)", header)

        score = float(score_match.group(1)) if score_match else None
        # ProteinMPNN's "score" is a per-residue negative log-likelihood — lower is more
        # confident. Converted to a bounded 0-1 confidence via exp(-score) for thresholding.
        mean_confidence = pow(2.718281828, -score) if score is not None else None

        parsed.append({
            "sequence": seq,
            "score": score,
            "mean_sequence_confidence": mean_confidence,
            "sample": int(sample_match.group(1)) if sample_match else i,
            "sampling_temperature": float(temp_match.group(1)) if temp_match else None,
        })
    return parsed


def run_proteinmpnn(backbone: dict, target_chain_id: str, n_sequences: int, sampling_temp: float) -> list[dict]:
    binder_chain = binder_chain_id(backbone["coordinates_file"], target_chain_id)
    out_dir = pathlib.Path("mpnn_cache") / backbone["backbone_id"]
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", str(PROTEINMPNN_DIR / "protein_mpnn_run.py"),
        "--pdb_path", backbone["coordinates_file"],
        "--pdb_path_chains", binder_chain,
        "--out_folder", str(out_dir),
        "--num_seq_per_target", str(n_sequences),
        "--sampling_temp", str(sampling_temp),
        "--batch_size", "1",
    ]
    run_checked(cmd)

    pdb_stem = pathlib.Path(backbone["coordinates_file"]).stem
    fasta_path = out_dir / "seqs" / f"{pdb_stem}.fa"
    designs = parse_proteinmpnn_fasta(fasta_path)

    candidates = []
    for d in designs:
        candidates.append({
            "candidate_id": f"{backbone['backbone_id']}_seq{d['sample']}",
            "backbone_id": backbone["backbone_id"],
            "sequence": d["sequence"],
            "per_residue_logprob": None,  # only in --score_only mode's .npz, not default FASTA output
            "mean_sequence_confidence": d["mean_sequence_confidence"],
            "sampling_temperature": d["sampling_temperature"] or sampling_temp,
            "model_version": "proteinmpnn_v_48_020",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        })
    return candidates


def design_sequences(surviving_backbones: list[dict], target_chain_id: str) -> list[dict]:
    all_candidates = []
    for backbone in surviving_backbones:
        all_candidates.extend(
            run_proteinmpnn(backbone, target_chain_id, N_SEQUENCES_PER_BACKBONE, SAMPLING_TEMP_DEFAULT)
        )
    return all_candidates


def prevalidate_candidates(candidates: list[dict]) -> list[dict]:
    survivors = []
    for c in candidates:
        if c["mean_sequence_confidence"] is None or c["mean_sequence_confidence"] < MEAN_CONFIDENCE_THRESHOLD:
            continue
        filter_result = physicochemical_filter(
            c["sequence"], ConstraintConfig(min_length=6, max_length=35)
        )
        if not filter_result.passed:
            continue
        c["physicochemical_attributes"] = filter_result.computed_attributes
        survivors.append(c)
    return survivors


# ============================================================================
# 8. Refinement + validation
# ============================================================================
#
# Threads each designed sequence onto its RFdiffusion backbone (real side chains
# via PDBFixer.applyMutations), locally minimizes it, independently re-folds the
# complex with AlphaFold-Multimer (ColabFold local + public MMseqs2 API), rescores
# survivors with the same EvoEF2+ESM2 ensemble used for hotspot discovery, then
# runs gated MD on the top survivors only. Not installed by default (ColabFold) —
# see module docstring for install refs.

AF_MULTIMER_MODEL_TYPE = "alphafold2_multimer_v3"
AF_MULTIMER_NUM_MODELS = 1          # single model per candidate — raise for a final confirmatory pass
AF_MULTIMER_NUM_RECYCLE = 3
IPTM_THRESHOLD = 0.6
INTERFACE_OVERLAP_MIN_FRACTION = 0.5  # AF-Multimer's predicted interface must substantially
                                        # overlap the anchor_residues it was designed against

RESIDUE_1TO3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU",
    "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE",
    "P": "PRO", "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}


def thread_sequence_onto_backbone(candidate: dict, backbone: dict, binder_chain_id: str) -> pathlib.Path:
    """RFdiffusion backbones carry a placeholder sequence (typically poly-glycine/poly-alanine),
    not the ProteinMPNN-designed one — PDBFixer.applyMutations rebuilds real side-chain rotamers
    for the designed identity before minimization."""
    structure = gemmi.read_structure(backbone["coordinates_file"])
    structure.setup_entities()

    binder_residues = sorted(
        (r.seqid.num, r.name) for r in structure[0][binder_chain_id]
    )
    if len(binder_residues) != len(candidate["sequence"]):
        raise ValueError(
            f"Backbone binder chain has {len(binder_residues)} residues, "
            f"designed sequence has {len(candidate['sequence'])} — cannot thread."
        )

    mutations = [
        f"{orig_name}-{resnum}-{RESIDUE_1TO3[aa]}"
        for (resnum, orig_name), aa in zip(binder_residues, candidate["sequence"])
        if orig_name != RESIDUE_1TO3[aa]
    ]

    threaded_dir = pathlib.Path("threaded_cache")
    threaded_dir.mkdir(exist_ok=True)
    threaded_path = threaded_dir / f"{candidate['candidate_id']}_threaded.pdb"

    fixer = PDBFixer(filename=backbone["coordinates_file"])
    if mutations:
        fixer.applyMutations(mutations, binder_chain_id)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)

    with open(threaded_path, "w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)

    return threaded_path


def minimize_structure(threaded_pdb_path: pathlib.Path) -> dict:
    """Short local energy minimization — implicit solvent, backbone restraints relaxed
    gradually is the spec's intent; OpenMM's own minimizeEnergy() with an implicit-solvent
    force field covers this without hand-rolling a restraint schedule."""
    pdb = openmm_app.PDBFile(str(threaded_pdb_path))
    forcefield = openmm_app.ForceField("amber14-all.xml", "implicit/gbn2.xml")

    system = forcefield.createSystem(
        pdb.topology, nonbondedMethod=openmm_app.NoCutoff, constraints=openmm_app.HBonds
    )
    integrator = LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds)
    platform = (
        Platform.getPlatformByName("CUDA")
        if "CUDA" in [Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())]
        else Platform.getPlatformByName("CPU")
    )

    simulation = openmm_app.Simulation(pdb.topology, system, integrator, platform)
    simulation.context.setPositions(pdb.positions)
    simulation.minimizeEnergy(maxIterations=2000)

    state = simulation.context.getState(getEnergy=True, getPositions=True)
    potential_energy = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)

    minimized_path = threaded_pdb_path.with_name(threaded_pdb_path.stem + "_min.pdb")
    with open(minimized_path, "w") as f:
        openmm_app.PDBFile.writeFile(pdb.topology, state.getPositions(), f)

    return {"minimized_structure_file": str(minimized_path), "potential_energy": potential_energy}


def write_complex_fasta(candidate: dict, target_sequence: str, out_path: pathlib.Path) -> None:
    """ColabFold multimer convention: single FASTA record, chains colon-separated
    (verified against colabfold/input.py — 'sequences = query_sequence.upper().split(":")')."""
    out_path.write_text(f">{candidate['candidate_id']}\n{target_sequence}:{candidate['sequence']}\n")


def run_colabfold_multimer(candidate: dict, target_sequence: str, out_dir: pathlib.Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = out_dir / f"{candidate['candidate_id']}.fasta"
    write_complex_fasta(candidate, target_sequence, fasta_path)

    cmd = [
        "colabfold_batch",
        str(fasta_path), str(out_dir),
        "--host-url", COLABFOLD_HOST_URL,
        "--model-type", AF_MULTIMER_MODEL_TYPE,
        "--num-models", str(AF_MULTIMER_NUM_MODELS),
        "--num-recycle", str(AF_MULTIMER_NUM_RECYCLE),
        "--num-relax", "0",
    ]
    run_checked(cmd)

    scores_files = sorted(out_dir.glob(f"{candidate['candidate_id']}_scores_rank_001_*.json"))
    pdb_files = sorted(out_dir.glob(f"{candidate['candidate_id']}_unrelaxed_rank_001_*.pdb"))
    if not scores_files or not pdb_files:
        raise RuntimeError(f"ColabFold did not produce expected rank_001 outputs in {out_dir}")

    with open(scores_files[0]) as f:
        scores = json.load(f)

    return {
        "predicted_complex_file": str(pdb_files[0]),
        "iptm": scores.get("iptm"),
        "ptm": scores.get("ptm"),
        "plddt": scores.get("plddt"),  # per-residue list
    }


def run_highfold_cyclic(*args, **kwargs):
    """Cyclic-aware AF2-Multimer validation (Duan lab, Briefings in Bioinformatics 2024,
    GPL-3.0). Layers a Cyclic Position Offset Encoding Matrix onto LocalColabFold —
    run via the same colabfold_batch entry point with --model-type alphafold2 once
    HighFold's source is installed alongside it. Reported RMSD 0.4-4.5A on known cyclic
    PDB structures. NOTE: GPL-3.0 is copyleft — fine for internal/research use, a
    constraint only if this integration were ever bundled/distributed as closed source."""
    raise NotImplementedError(
        "HighFold cyclic validation is not implemented — see module docstring for setup. "
        "This raises rather than silently validating a cyclic candidate with plain "
        "AF-Multimer, whose linear position encoding cannot score macrocycle closure."
    )


def interface_overlap_ok(predicted_complex_file: str, plddt: list[float], target_chain_id: str,
                          binder_chain_id: str, anchor_resnums: list[int],
                          contact_cutoff: float = CONTACT_DISTANCE) -> bool:
    """Reject if AF-Multimer folds the binder onto a different surface than the one it
    was designed against — RFdiffusion/ProteinMPNN can produce a sequence that folds
    'confidently' onto the wrong interface."""
    structure = gemmi.read_structure(predicted_complex_file)
    structure.setup_entities()

    binder_atoms = [atom.pos for res in structure[0][binder_chain_id] for atom in res]

    predicted_contacts = set()
    for res in structure[0][target_chain_id]:
        res_atoms = [atom.pos for atom in res]
        if any(a.dist(b) <= contact_cutoff for a in res_atoms for b in binder_atoms):
            predicted_contacts.add(res.seqid.num)

    overlap = len(predicted_contacts & set(anchor_resnums))
    return (overlap / len(anchor_resnums) if anchor_resnums else 0.0) >= INTERFACE_OVERLAP_MIN_FRACTION


def validate_with_af_multimer(candidate: dict, target_sequence: str, prepared_structure: dict,
                                hotspot_record: dict, binder_chain_id: str, backbone_topology: str) -> dict | None:
    out_dir = pathlib.Path("af_multimer_cache") / candidate["candidate_id"]

    if backbone_topology == "linear":
        result = run_colabfold_multimer(candidate, target_sequence, out_dir)
    elif backbone_topology == "cyclic":
        result = run_highfold_cyclic(candidate, target_sequence, out_dir)
    else:
        raise ValueError(f"Unknown backbone_topology: {backbone_topology!r}")

    if result["iptm"] is None or result["iptm"] < IPTM_THRESHOLD:
        return None

    anchor_resnums = [a["structure_resnum"] for a in hotspot_record["anchor_residues"]]
    if not interface_overlap_ok(
        result["predicted_complex_file"], result["plddt"],
        prepared_structure["chain_id"], binder_chain_id, anchor_resnums,
    ):
        return None

    return result


# Step 6 (optional): EvoEF2+ESM2 rescoring, reusing the exact ensemble from hotspot
# discovery's alanine-scan machinery -- same evoef2_repair/evoef2_binding_energy,
# just scoring the AF-Multimer-validated complex directly rather than a point mutation,
# so both stages report a consistent ddG-style number.
def evoef2_esm2_rescore(predicted_complex_file: str, target_chain_id: str, binder_chain_id: str) -> float:
    repaired_path = evoef2_repair(predicted_complex_file)
    return evoef2_binding_energy(repaired_path, target_chain_id, binder_chain_id)


MD_TOP_K = 15          # top 10-20 per spec
MD_DURATION_NS = 20     # 10-50ns range
MD_STEP_SIZE_FS = 2.0
MD_REPORT_INTERVAL_PS = 100


def run_gated_md(minimized_structure_file: str, duration_ns: float = MD_DURATION_NS) -> dict:
    """Extends the minimization run with unrestrained dynamics, tracking whether
    the designed interface actually stays together rather than drifting apart."""
    pdb = openmm_app.PDBFile(minimized_structure_file)
    forcefield = openmm_app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
    system = forcefield.createSystem(
        pdb.topology, nonbondedMethod=openmm_app.NoCutoff, constraints=openmm_app.HBonds
    )
    integrator = LangevinMiddleIntegrator(
        300 * unit.kelvin, 1 / unit.picosecond, MD_STEP_SIZE_FS * unit.femtoseconds
    )
    platform = (
        Platform.getPlatformByName("CUDA")
        if "CUDA" in [Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())]
        else Platform.getPlatformByName("CPU")
    )

    simulation = openmm_app.Simulation(pdb.topology, system, integrator, platform)
    simulation.context.setPositions(pdb.positions)
    simulation.context.setVelocitiesToTemperature(300 * unit.kelvin)

    n_steps = int(duration_ns * 1_000_000 / MD_STEP_SIZE_FS)
    report_every = int(MD_REPORT_INTERVAL_PS * 1000 / MD_STEP_SIZE_FS)

    reference_positions = pdb.positions
    rmsd_trajectory = []
    for step in range(0, n_steps, report_every):
        simulation.step(report_every)
        state = simulation.context.getState(getPositions=True)
        positions = state.getPositions(asNumpy=True)
        ref = reference_positions.value_in_unit(unit.nanometer)
        cur = positions.value_in_unit(unit.nanometer)
        rmsd = (sum((c[i] - r[i]) ** 2 for c, r in zip(cur, ref) for i in range(3)) / len(cur)) ** 0.5
        rmsd_trajectory.append(rmsd)

    mean_late_rmsd = sum(rmsd_trajectory[len(rmsd_trajectory) // 2:]) / max(1, len(rmsd_trajectory) // 2)
    binding_stability_flag = "stable" if mean_late_rmsd < 0.5 else "unstable"  # nm, heuristic cutoff

    return {
        "rmsd_trajectory": rmsd_trajectory,
        "contact_persistence": None,  # TODO: track anchor-residue contact fraction per frame
        "binding_stability_flag": binding_stability_flag,
    }


def refine_and_validate(surviving_candidates: list[dict], surviving_backbones: list[dict],
                          prepared_structure: dict, hotspot_record: dict, target_input: dict) -> list[dict]:
    backbones_by_id = {b["backbone_id"]: b for b in surviving_backbones}
    # prepared_structure carries coordinates, not a sequence string — derive it from the chain
    target_sequence, _ = get_chain_sequence_and_map(prepared_structure["local_file_path"], prepared_structure["chain_id"])

    validated = []
    for candidate in surviving_candidates:
        backbone = backbones_by_id[candidate["backbone_id"]]
        binder_chain = binder_chain_id(backbone["coordinates_file"], prepared_structure["chain_id"])

        # Step 4 — OpenMM minimization
        threaded_path = thread_sequence_onto_backbone(candidate, backbone, binder_chain)
        min_result = minimize_structure(threaded_path)

        # Step 5 — AF-Multimer / HighFold validation
        af_result = validate_with_af_multimer(
            candidate, target_sequence, prepared_structure, hotspot_record,
            binder_chain, target_input["backbone_topology"],
        )
        if af_result is None:
            continue

        candidate = {
            **candidate,
            "minimized_structure_file": min_result["minimized_structure_file"],
            "potential_energy": min_result["potential_energy"],
            "af_multimer_iptm": af_result["iptm"],
            "af_multimer_ptm": af_result["ptm"],
            "predicted_complex_file": af_result["predicted_complex_file"],
        }
        validated.append(candidate)

    # Step 6 — EvoEF2+ESM2 rescoring (same ensemble as hotspot identification)
    for candidate in validated:
        backbone = backbones_by_id[candidate["backbone_id"]]
        binder_chain = binder_chain_id(backbone["coordinates_file"], prepared_structure["chain_id"])
        candidate["evoef2_esm2_score"] = evoef2_esm2_rescore(
            candidate["predicted_complex_file"], prepared_structure["chain_id"], binder_chain
        )

    # Step 7 — gated MD, top-K survivors only (ranked by AF-Multimer ipTM)
    validated.sort(key=lambda c: -c["af_multimer_iptm"])
    for candidate in validated[:MD_TOP_K]:
        md_result = run_gated_md(candidate["minimized_structure_file"])
        candidate["md_refined"] = True
        candidate["md_stability_flag"] = md_result["binding_stability_flag"]
    for candidate in validated[MD_TOP_K:]:
        candidate["md_refined"] = False
        candidate["md_stability_flag"] = None

    return validated


COLABFOLD_HOST_URL = "https://api.colabfold.com"  # ColabFold's own default; only needed if overriding


# ============================================================================
# 9. Final Route C output
# ============================================================================

def model_version_pipeline(target_input: dict) -> dict:
    return {
        "uniprot_resolution": "rest.uniprot.org (live)",
        "structure_source": "RCSB PDBe best_structures / AlphaFold DB",
        "pocket_detection": "fpocket + P2Rank 2.5.1",
        "hotspot_scoring": "EvoEF2 (source build) + ESM2 (esm2_t12_35M_UR50D)",
        "backbone_generation": (
            "RFdiffusion (PPI/hotspot-conditioned)" if target_input["backbone_topology"] == "linear" else "RFpeptides"
        ),
        "sequence_design": "ProteinMPNN v_48_020",
        "structure_validation": (
            "ColabFold / AlphaFold2-Multimer v3" if target_input["backbone_topology"] == "linear" else "HighFold"
        ),
        "md_engine": "OpenMM 8.6, amber14-all + implicit/gbn2",
    }


def to_shared_schema(candidate: dict, prepared_structure: dict, hotspot_record: dict, target_input: dict) -> dict:
    return {
        "candidate_id": candidate["candidate_id"],
        "sequence": candidate["sequence"],
        "route": "C",
        "target_id": prepared_structure["structure_id"],
        "backbone_id": candidate["backbone_id"],
        "anchor_residues_used": [a["structure_resnum"] for a in hotspot_record["anchor_residues"]],
        "mean_sequence_confidence": candidate["mean_sequence_confidence"],
        "af_multimer_iptm": candidate["af_multimer_iptm"],
        "af_multimer_ptm": candidate["af_multimer_ptm"],
        "evoef2_esm2_score": candidate["evoef2_esm2_score"],
        "md_refined": candidate["md_refined"],
        "md_stability_flag": candidate["md_stability_flag"],
        "model_version_pipeline": model_version_pipeline(target_input),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


# ============================================================================
# Route C driver
# ============================================================================

async def run_route_c(target_input: dict) -> list[dict]:
    """Runs the full Route C pipeline (stages 1-9) for one target_input record.
    Stages 2 ("peptide_length_range"/"backbone_topology" defaults) and 6-8
    (RFdiffusion/ProteinMPNN/ColabFold) mirror the notebook's own placeholders
    and "not installed" caveats -- see module docstring."""
    target_input = dict(target_input)
    target_input.setdefault("peptide_length_range", (8, 40))
    target_input.setdefault("backbone_topology", "linear")

    selected = await resolve_uniprot_target(target_input)
    uniprot_accession = selected["uniprot_accession"]

    structure_record, experimental_fallback_pool = await select_structure(uniprot_accession, target_input)
    prepared_structure = await prepare_structure(
        structure_record, experimental_fallback_pool, target_input, uniprot_accession
    )

    pocket_record = detect_pocket(prepared_structure, target_input)
    hotspot_record = identify_hotspots(prepared_structure, pocket_record, target_input)

    n_designs_requested = target_input.get("n_designs_requested", 25)
    backbones = generate_backbones(prepared_structure, hotspot_record, target_input, n_designs_requested, False)
    surviving_backbones = filter_backbones(backbones, prepared_structure, hotspot_record)
    logger.info(
        "routec.backbone.generated",
        extra={"n_generated": len(backbones), "n_survived": len(surviving_backbones)},
    )

    candidates = design_sequences(surviving_backbones, prepared_structure["chain_id"])
    surviving_candidates = prevalidate_candidates(candidates)
    logger.info(
        "routec.sequence_design.done",
        extra={"n_designed": len(candidates), "n_survived": len(surviving_candidates)},
    )

    validated_candidates = refine_and_validate(
        surviving_candidates, surviving_backbones, prepared_structure, hotspot_record, target_input
    )
    logger.info("routec.validation.done", extra={"n_validated": len(validated_candidates)})

    return [
        to_shared_schema(c, prepared_structure, hotspot_record, target_input) for c in validated_candidates
    ]


class RouteC(Stage4Route):
    """Target-interface candidate design: structure retrieval, pocket/hotspot
    identification, RFdiffusion backbone generation, ProteinMPNN sequence
    design, and AF-Multimer/OpenMM refinement (Stage 4, route C). See
    CLAUDE.md's multi-route generation convention. Requires config["target_input"]
    (Stage 1/2 target definition) and locally-installed structural biology
    tooling (fpocket, P2Rank, EvoEF2, RFdiffusion, ProteinMPNN, ColabFold) —
    see module docstring for install references."""

    def run(self) -> list[Candidate]:
        target_input = self.config.get("target_input")
        if not target_input:
            raise ValueError(
                "RouteC requires config['target_input'] (target-interface design "
                "needs a resolved target to design against)."
            )

        import asyncio

        route_c_output = asyncio.run(run_route_c(target_input))

        return [
            Candidate(
                id=record["candidate_id"],
                sequence=record["sequence"],
                predictions={k: v for k, v in record.items() if k not in ("candidate_id", "sequence")},
            )
            for record in route_c_output
        ]
