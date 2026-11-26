"""The food protein library: known food proteins (whey, casein, soy, pea, wheat) with their source provenance, for the food pipeline.

Two uses:
- Library discovery: a customer picks library proteins and runs the pipeline without uploading anything.
- Library matching: an uploaded protein that is exactly a library protein gets the library's family, allergen-source flag, version and UniProt release. An uploaded protein that
  is not in the library is never rejected for that; it is reported as "not in the library" and keeps what the customer supplied.

A library protein is a digestion substrate: one mature chain of a UniProt protein, the unit the digestion stage cuts. The data is a copy of the Phase 1 protein library, with its
UniProt release recorded in `data/MANIFEST.json`. Standard library only, so the API can use it without the pipeline.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from common import storage
from common.protein_fasta import Substrate, read_fasta

BUNDLED_DATA = Path(__file__).resolve().parent / "data"
DATA = BUNDLED_DATA  # the copy that ships with the code


LIBRARY_FILES = ("digestion_substrates.fasta", "source_records.csv", "MANIFEST.json")
_downloaded: dict[str, Path] = {}


class LibraryFilesMissing(RuntimeError):
    """A library file is not in the folder FOOD_LIBRARY_DIR points to."""


def data_dir() -> Path:
    """The folder holding the three library files. FOOD_LIBRARY_DIR is a local folder (the setting under DEV_MODE) or a gs:// folder (the setting in the cloud, downloaded once per process);
    empty means the copy bundled with the code."""
    setting = os.environ.get("FOOD_LIBRARY_DIR", "").strip()
    if not setting:
        return BUNDLED_DATA
    if not storage.is_gcs_path(setting):
        return Path(setting)
    if setting not in _downloaded:
        target = Path(tempfile.mkdtemp(prefix="food_library_"))
        for name in LIBRARY_FILES:
            source = storage.join(setting, name)
            try:
                (target / name).write_text(storage.read_text(source), encoding="utf-8")
            except Exception as exc:  # a missing object, no permission or no network: say which file and where
                raise LibraryFilesMissing(f"food library file {name} could not be read from {source}: {type(exc).__name__}: {exc}") from exc
        _downloaded[setting] = target
    return _downloaded[setting]

STATUS_EXACT = "exact_sequence"
STATUS_ACCESSION_ONLY = "accession_sequence_differs"
STATUS_NOT_IN_LIBRARY = "not_in_library"

NOT_IN_LIBRARY_MESSAGE = "not in the library: family and allergen source are not established; the run uses what was supplied with the upload"
ACCESSION_ONLY_MESSAGE = "the accession is in the library but the sequence differs (another chain, isoform or variant), so no library data is applied"


class UnknownLibraryProteins(ValueError):
    def __init__(self, missing: list[str]):
        super().__init__(f"not in the library: {', '.join(missing[:10])}{' ...' if len(missing) > 10 else ''}")
        self.missing = missing


@dataclass(frozen=True)
class LibraryProtein:
    substrate_id: str
    accession: str
    name: str
    food_source: str
    kind: str
    length: int
    sequence: str
    header: str
    protein_family: str | None
    organism: str | None
    allergen_source: bool | None
    sequence_version: str | None
    uniprot_release: str | None
    retrieved_at: str | None
    n_terminus: str | None = None  # the chain's N-terminal end (a TERMINUS_MEANING code); None when the FASTA header has none
    c_terminus: str | None = None

    def info(self) -> dict:
        """The protein without its sequence, for listings."""
        return {k: v for k, v in self.__dict__.items() if k not in ("sequence", "header")}


@dataclass(frozen=True)
class Match:
    id: str  # the uploaded record's substrate id
    status: str
    library_substrate_id: str | None = None
    protein_family: str | None = None
    allergen_source: bool | None = None
    message: str = ""


def _flag(value: str | None) -> bool | None:
    return {"True": True, "False": False}.get(value or "")


class FoodLibrary:
    def __init__(self, folder: Path | None = None):
        data_dir = folder or globals()["data_dir"]()
        substrates, issues = read_fasta(data_dir / "digestion_substrates.fasta")
        if issues:
            raise ValueError(f"the library FASTA has QC issues: {issues[:3]}")
        with open(data_dir / "source_records.csv", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            self.record_columns = list(reader.fieldnames or [])
            self.records = {r["accession"]: r for r in reader}
        self.manifest = json.loads((data_dir / "MANIFEST.json").read_text(encoding="utf-8"))
        self._substrates = {s.substrate_id: s for s in substrates}
        self.proteins = [self._protein(s) for s in substrates]
        self.by_id = {p.substrate_id: p for p in self.proteins}
        self._by_sequence: dict[str, LibraryProtein] = {}
        for p in self.proteins:
            self._by_sequence.setdefault(hashlib.sha256(p.sequence.encode()).hexdigest(), p)
        self._accessions = {p.accession for p in self.proteins}

    def _protein(self, s: Substrate) -> LibraryProtein:
        r = self.records.get(s.accession, {})
        return LibraryProtein(
            substrate_id=s.substrate_id, accession=s.accession, name=r.get("protein_name") or s.name.replace("_", " "), food_source=s.food_source or r.get("food_source", ""),
            kind=s.kind, length=len(s.sequence), sequence=s.sequence, header=s.header, protein_family=r.get("protein_family") or None, organism=r.get("organism") or None,
            n_terminus=s.n_terminus or None, c_terminus=s.c_terminus or None, allergen_source=_flag(r.get("allergen_source_flag")), sequence_version=r.get("sequence_version") or None, uniprot_release=r.get("uniprot_release") or None,
            retrieved_at=r.get("retrieved_at") or None)

    # ------------------------------------------------------------------ discovery
    def summary(self) -> dict:
        families: dict[tuple[str, str], int] = {}
        for p in self.proteins:
            key = (p.food_source, p.protein_family or "unclassified")
            families[key] = families.get(key, 0) + 1
        by_source: dict[str, int] = {}
        for p in self.proteins:
            by_source[p.food_source] = by_source.get(p.food_source, 0) + 1
        flags = {"allergen_source": sum(p.allergen_source is True for p in self.proteins), "not_flagged": sum(p.allergen_source is False for p in self.proteins),
                 "not_recorded": sum(p.allergen_source is None for p in self.proteins)}
        return {"manifest": {k: self.manifest[k] for k in ("name", "uniprot_release", "retrieved_at", "note")}, "total": len(self.proteins), "by_food_source": by_source,
                "families": [{"food_source": s, "family": f, "count": n} for (s, f), n in sorted(families.items())], "allergen_source_flags": flags}

    def search(self, food_source: str | None = None, family: str | None = None, q: str | None = None, limit: int = 50, offset: int = 0) -> tuple[int, list[LibraryProtein]]:
        """(total matching, one page). `q` matches the accession, substrate id, protein name or family, ignoring case."""
        needle = (q or "").strip().lower()
        hits = [p for p in self.proteins
                if (not food_source or p.food_source.lower() == food_source.lower())
                and (not family or (p.protein_family or "").lower() == family.lower())
                and (not needle or needle in " ".join([p.accession, p.substrate_id, p.name, p.protein_family or ""]).lower())]
        return len(hits), hits[offset:offset + limit]

    def resolve(self, substrate_ids: list[str]) -> list[LibraryProtein]:
        """The proteins for these substrate ids, in the order given and without repeats; UnknownLibraryProteins when any is not in the library."""
        unique = list(dict.fromkeys(substrate_ids))
        missing = [i for i in unique if i not in self.by_id]
        if missing:
            raise UnknownLibraryProteins(missing)
        return [self.by_id[i] for i in unique]

    def substrates(self, proteins: list[LibraryProtein]) -> list[Substrate]:
        """The parsed FASTA records of these proteins, for the digestion engine."""
        return [self._substrates[p.substrate_id] for p in proteins]

    # ------------------------------------------------------------------ what a run is given
    def fasta_text(self, proteins: list[LibraryProtein]) -> str:
        return "".join(f">{p.header}\n{p.sequence}\n" for p in proteins)

    def records_csv(self, proteins: list[LibraryProtein], rekey: dict[str, str] | None = None) -> str:
        """The source records of these proteins, one per accession, in the library's own columns plus `library_substrate_id`.
        `rekey` maps a library substrate id to the accession an uploaded protein uses, so the pipeline's join by accession finds the record."""
        out = io.StringIO()
        writer = csv.DictWriter(out, fieldnames=[*self.record_columns, "library_substrate_id"], lineterminator="\n")
        writer.writeheader()
        seen: set[str] = set()
        for p in proteins:
            row = dict(self.records.get(p.accession, {}))
            if not row:
                continue
            if rekey and p.substrate_id in rekey:
                row["accession"] = rekey[p.substrate_id]
            if row["accession"] in seen:
                continue
            seen.add(row["accession"])
            writer.writerow({**row, "library_substrate_id": p.substrate_id})
        return out.getvalue()

    # ------------------------------------------------------------------ matching an upload
    def match_upload(self, uploaded: list[Substrate]) -> list[Match]:
        """Each uploaded record against the library: the same sequence, the same accession with another sequence, or neither. Never a rejection."""
        out = []
        for s in uploaded:
            hit = self._by_sequence.get(hashlib.sha256(s.sequence.encode()).hexdigest())
            if hit is not None:
                out.append(Match(s.substrate_id, STATUS_EXACT, hit.substrate_id, hit.protein_family, hit.allergen_source,
                                 f"same sequence as library protein {hit.substrate_id}: its family, allergen-source flag and release are applied"))
            elif s.accession in self._accessions:
                out.append(Match(s.substrate_id, STATUS_ACCESSION_ONLY, message=ACCESSION_ONLY_MESSAGE))
            else:
                out.append(Match(s.substrate_id, STATUS_NOT_IN_LIBRARY, message=NOT_IN_LIBRARY_MESSAGE))
        return out

    def matched_records_csv(self, uploaded: list[Substrate], matches: list[Match]) -> str | None:
        """Source records for the exactly matched uploads, keyed by the upload's accession; None when nothing matched."""
        rekey = {m.library_substrate_id: s.accession for s, m in zip(uploaded, matches) if m.status == STATUS_EXACT}
        if not rekey:
            return None
        proteins = [self.by_id[i] for i in rekey]
        return self.records_csv(proteins, rekey)


@lru_cache(maxsize=1)
def library() -> FoodLibrary:
    return FoodLibrary()
