# Stage-level rejection counters and the run-level tally that sums them from stage 5 on.

from __future__ import annotations

from types import SimpleNamespace

from pipeline.base import CandidateStage, RejectionTally


class _Recorder:
    def record_stage(self, result, *, run_id):
        self.last = result

    def record_failure(self, *args, **kwargs):
        raise AssertionError("unexpected failure")


class _Boundary:
    def write(self, *args, **kwargs):
        pass


def _stage(stage_name: str, drops: dict[str, list[str]]):
    """A stage that removes one candidate per entry of `drops`, noting each entry's reasons."""

    class Dropper(CandidateStage):
        name = stage_name

        def run(self, candidates, config, ctx):
            for reasons in drops.values():
                self.note_rejection(*reasons)
            return candidates[len(drops):]

    return Dropper()


def _ctx():
    return SimpleNamespace(run_id="r", audit=_Recorder(), boundary=_Boundary(), rejections=RejectionTally())


def _candidates(n: int):
    return [SimpleNamespace(id=str(i), sequence="A", predictions={}) for i in range(n)]


def test_each_stage_starts_a_fresh_counter_and_the_run_tally_sums_them() -> None:
    ctx = _ctx()
    first = _stage("s05_x", {"a": ["too_hydrophobic", "too_long"], "b": ["too_long"]})
    second = _stage("s08_x", {"c": ["hemolysis"]})
    r1 = first.execute(_candidates(5), SimpleNamespace(), ctx)
    r2 = second.execute(r1.candidates, SimpleNamespace(), ctx)
    assert r1.rejection_reasons == {"too_hydrophobic": 1, "too_long": 2}
    assert r2.rejection_reasons == {"hemolysis": 1}
    # One peptide failed two checks, so the reasons (4) exceed the number removed (3).
    assert ctx.rejections.as_dict() == {"n_removed": 3, "reasons": {"too_hydrophobic": 1, "too_long": 2, "hemolysis": 1}}


def test_stages_before_five_are_not_counted() -> None:
    ctx = _ctx()
    result = _stage("s04_x", {"a": ["duplicate"]}).execute(_candidates(3), SimpleNamespace(), ctx)
    assert result.rejection_reasons == {"duplicate": 1}
    assert ctx.rejections.as_dict() == {"n_removed": 0, "reasons": {}}
