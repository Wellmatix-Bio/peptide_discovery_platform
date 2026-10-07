"""Stage 3: standardising peptide records from the knowledge base.

**Stage 3 is a stub, and these tests say so rather than disguising it.** Its own docstring reads
"Placeholder implementation; replace with real source integration", and what it returns is one
record echoing the knowledge base's `source_path` -- not peptide data. `KnowledgeBase` is
likewise a stub holding only that path.

Testing a stub is worth it for one reason: the stage is force-disabled on the deployed path, so
nobody meets this behaviour day to day, and the next person to enable it needs to know what it
does and does NOT return before building on it. These tests fail the moment it becomes real,
which is the intended signal to come back and write proper ones.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from pipeline.s03_data_integration.stage import Stage3  # noqa: E402
from schemas.knowledge_base import KnowledgeBase  # noqa: E402
from schemas.run_config import StageConfig  # noqa: E402


class FakeCtx:
    def __init__(self, knowledge_base=None):
        self.knowledge_base = knowledge_base


def test_no_knowledge_base_yields_no_records():
    """An empty list, not None and not a crash: a run without a knowledge base is a supported
    configuration, and later stages receive something iterable."""
    assert Stage3().run(StageConfig(), FakeCtx(knowledge_base=None)) == []


def test_a_knowledge_base_yields_one_record_echoing_its_path():
    kb = KnowledgeBase(source_path="/data/known_peptides.json")
    got = Stage3().run(StageConfig(), FakeCtx(knowledge_base=kb))
    assert got == [{"source_path": "/data/known_peptides.json"}]


def test_the_stage_is_still_a_stub_and_returns_no_peptide_data():
    """THE POINT OF THIS FILE. The record carries a file path, not sequences, properties or
    anything a later stage could screen. When this stage is implemented for real, this test
    fails -- which is the reminder to replace these tests rather than trust them.
    """
    kb = KnowledgeBase(source_path="/data/known_peptides.json")
    [record] = Stage3().run(StageConfig(), FakeCtx(knowledge_base=kb))
    assert list(record) == ["source_path"], (
        "Stage 3 now returns more than a source path. It is no longer the stub these tests "
        "describe; write tests for what it actually produces."
    )


def test_run_delegates_to_integrate_data():
    """`run` adds nothing of its own, so `integrate_data` can be called directly by anything
    building on this stage."""
    kb = KnowledgeBase(source_path="/data/x.json")
    ctx = FakeCtx(knowledge_base=kb)
    assert Stage3().run(StageConfig(), ctx) == Stage3().integrate_data(ctx)


def test_config_params_are_not_consulted():
    """Pinned so a later reader does not assume a knob exists. The stage ignores its config
    entirely; anything passed in params has no effect."""
    kb = KnowledgeBase(source_path="/data/x.json")
    plain = Stage3().run(StageConfig(), FakeCtx(knowledge_base=kb))
    with_params = Stage3().run(
        StageConfig(params={"sources": ["dbaasp", "apd3"], "limit": 10}),
        FakeCtx(knowledge_base=kb),
    )
    assert plain == with_params
