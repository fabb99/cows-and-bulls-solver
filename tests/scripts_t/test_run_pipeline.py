import pytest

from scripts.run_pipeline import (
    MAX_ITERATIONS,
    Stage,
    build_stages,
    decide,
    execute,
    fingerprint,
    load_manifest,
)


def test_fingerprint_tracks_content_dirs_and_missing(tmp_path):
    f, d = tmp_path / "f.txt", tmp_path / "d"
    d.mkdir()
    f.write_text("a")
    (d / "x.txt").write_text("1")
    before = (fingerprint([f]), fingerprint([d]), fingerprint([tmp_path / "nope"]))
    assert before == (fingerprint([f]), fingerprint([d]), fingerprint([tmp_path / "nope"]))
    f.write_text("b")
    (d / "x.txt").write_text("2")
    assert fingerprint([f]) != before[0] and fingerprint([d]) != before[1]
    assert before[2] != before[0]


def test_decide_threshold_and_iteration_cap():
    games = 5040
    assert decide(1000, 999, games, 0)[0] is True       # una mossa in meno: 1.98e-4 >= 1e-4
    assert decide(1000, 1000, games, 0)[0] is False     # nessun miglioramento
    assert decide(1000, 1005, games, 0)[0] is False     # peggiora
    assert decide(1000, 900, games, MAX_ITERATIONS - 1)[0] is False  # tetto di iterazioni


def _stages(tmp, calls):
    def make(name):
        out = tmp / f"{name}.out"

        def act():
            calls.append(name)
            out.write_text(name)

        return Stage(name, outputs=(out,), action=act,
                     inputs=(tmp / "in.txt",) if name == "a" else ())

    return [make("a"), make("b")]


def test_execute_skip_cascade_from_only_and_dry_run(tmp_path):
    (tmp_path / "in.txt").write_text("1")
    calls, directory = [], tmp_path / "run"
    stages = _stages(tmp_path, calls)

    execute(stages, directory)
    assert calls == ["a", "b"]
    execute(stages, directory)
    assert calls == ["a", "b"]                      # tutto saltato

    (tmp_path / "in.txt").write_text("2")           # input esterno cambiato
    execute(stages, directory)
    assert calls == ["a", "b", "a", "b"]

    execute(stages, directory, from_stage="b")
    assert calls[-1] == "b" and len(calls) == 5

    execute(stages, directory, only="a", force=True)
    assert calls[-1] == "a" and len(calls) == 6
    assert "b" not in load_manifest(directory)["stages"]   # a rieseguito: b invalidato
    execute(stages, directory)
    assert calls[-1] == "b" and len(calls) == 7

    execute(stages, directory, force=True, dry_run=True)
    assert len(calls) == 7
    with pytest.raises(SystemExit):
        execute(stages, directory, only="zzz")


def test_failed_stage_is_not_recorded(tmp_path):
    def boom():
        raise RuntimeError("boom")

    stage = Stage("x", outputs=(tmp_path / "x.out",), action=boom)
    with pytest.raises(SystemExit):
        execute([stage], tmp_path / "run")
    assert "x" not in load_manifest(tmp_path / "run")["stages"]
    optional = Stage("y", outputs=(tmp_path / "y.out",), action=boom, optional=True)
    execute([optional], tmp_path / "run")           # non solleva
    assert "y" not in load_manifest(tmp_path / "run")["stages"]


def test_build_stages_shapes():
    s0, s1 = build_stages(0), build_stages(1)
    names0, names1 = [s.name for s in s0], [s.name for s in s1]
    assert "baseline" in names0 and "baseline" not in names1
    assert names0.index("mine") < names0.index("augment") < names0.index("refine")
    assert names0[-1] == "verdict" and names0[0] == "init"
    refine1 = next(s for s in s1 if s.name == "refine")
    assert "estimated_evm_selected" in refine1.args and "--strict-models" in refine1.args
    assert any("iter_00" in str(p) for p in refine1.inputs)
    refine0 = next(s for s in s0 if s.name == "refine")
    assert "estimated_evm_selected" not in refine0.args