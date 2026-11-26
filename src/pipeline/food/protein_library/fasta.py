"""FASTA reading with QC for the food protein library. The reader lives in `common.protein_fasta` so the API can check a submission without importing the pipeline."""
from common.protein_fasta import STANDARD, QCIssue, Substrate, read_fasta

__all__ = ["STANDARD", "QCIssue", "Substrate", "read_fasta"]
