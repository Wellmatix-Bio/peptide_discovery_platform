# Unit tests for common.sequence_utils.parse_sequences.
from __future__ import annotations

import pytest

from common.sequence_utils import parse_sequence_items, parse_sequences


def reasons(result):
    return [(r.id, r.reason) for r in result.rejected]


def test_fasta_headers_and_wrapped_lines():
    r = parse_sequences(">a first\nACDEF\nGHIK\n>b\nLMNPQR\n")
    assert [(p.id, p.sequence) for p in r.accepted] == [("a first", "ACDEFGHIK"), ("b", "LMNPQR")]


def test_plain_lines_get_generated_ids():
    r = parse_sequences("ACDEFG\n\n  hikLMN  \n")
    assert [(p.id, p.sequence) for p in r.accepted] == [("seq_1", "ACDEFG"), ("seq_2", "HIKLMN")]


def test_whitespace_inside_and_bom_are_removed():
    assert parse_sequences("﻿>x\nAC DE\tFG\r\n").sequences == ["ACDEFG"]


def test_invalid_residues_are_rejected_not_repaired():
    r = parse_sequences(">x\nACDEFX\n>y\nACDE*G\n>z\nACDEF1\n")
    assert not r.accepted
    assert [x.detail for x in r.rejected] == ["X", "*", "1"]


def test_duplicates_removed_case_insensitively_first_wins():
    r = parse_sequences(">a\nACDEFG\n>b\nacdefg\n>c\nACDEFH\n")
    assert [p.id for p in r.accepted] == ["a", "c"]
    assert reasons(r) == [("b", "duplicate")]


@pytest.mark.parametrize("seq,ok", [("A" * 5, False), ("A" * 6, True), ("A" * 50, True), ("A" * 51, False)])
def test_default_length_limits(seq, ok):
    assert bool(parse_sequences(seq).accepted) is ok


def test_limits_are_configurable():
    assert parse_sequences("ACDEFGHIKL", min_length=8, max_length=10).sequences == ["ACDEFGHIKL"]
    assert reasons(parse_sequences("ACDE", min_length=2, max_length=3)) == [("seq_1", "too_long")]


def test_header_without_sequence_is_reported_empty():
    assert reasons(parse_sequences(">a\n>b\nACDEFG")) == [("a", "empty")]


@pytest.mark.parametrize("lo,hi", [(0, 5), (10, 5)])
def test_bad_limits_raise(lo, hi):
    with pytest.raises(ValueError):
        parse_sequences("ACDEFG", min_length=lo, max_length=hi)


def test_items_are_read_one_by_one_so_a_header_does_not_swallow_its_neighbours():
    r = parse_sequence_items(["ALPMHIR", ">named" + chr(10) + "AMKPWIQPK", "alpmhir", "ACDEFX"])
    assert [(p.id, p.sequence) for p in r.accepted] == [("seq_1", "ALPMHIR"), ("named", "AMKPWIQPK")]
    assert [(x.id, x.reason) for x in r.rejected] == [("seq_3", "duplicate"), ("seq_4", "invalid_residues")]
