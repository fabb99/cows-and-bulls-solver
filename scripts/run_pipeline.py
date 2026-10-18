"""
Modulo: run_pipeline (Script)
Orchestra UNA iterazione della pipeline, in stadi con manifest e skip idempotente.
Si ferma a fine iterazione, mostra il verdetto e aspetta il comando dell'utente.

Uso (dalla radice del progetto):
  python -m scripts.run_pipeline --iter 0 [--only STADIO] [--from STADIO] [--force]
                                 [--dry-run] [--tournament-max-n 300]

Regola di stop: si prosegue con l'iterazione k+1 solo se la media dell'albero
estimated_evm_selected migliora di almeno MIN_IMPROVEMENT rispetto a quella precedente
(k=0: rispetto all'albero con i coefficienti legacy), con al massimo MAX_ITERATIONS iterazioni.
Il modello finale è l'iterazione con la media minima (data/runs/final.json).
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from src.analysis.best_model import pick_best_model, tree_stats
from src.analysis.metrics import calculate_total_moves_and_leaves, max_depth
from src.analysis.tree_io import load_tree_from_json
from src.paths import DATA_DIR, LEGACY_DIR

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = DATA_DIR / "runs"
MAX_ITERATIONS = 3
MIN_IMPROVEMENT = 1e-4  # con 5040 partite equivale a >= 1 mossa totale in meno

LEGACY_TREE = "estimated_evm_legacy.json"


def run_dir(k: int) -> Path:
    """POST: cartella dell'iterazione k (data/runs/iter_KK)."""
    return RUNS_DIR / f"iter_{k:02d}"


# ------------------------------------------------------------------- stadi
@dataclass(frozen=True)
class Stage:
    """
    name:     identificativo dello stadio.
    args:     argomenti di `python` (es. ("-m", "scripts.generate_trees", ...)); vuoto se action.
    action:   funzione interna, alternativa a args.
    inputs:   file/cartelle ESTERNI all'iterazione: se cambiano, lo stadio va rieseguito.
    outputs:  file la cui esistenza è condizione per lo skip.
    optional: se True, un fallimento è segnalato ma non ferma la pipeline.
    """

    name: str
    outputs: tuple[Path, ...]
    args: tuple[str, ...] = ()
    action: Callable[[], None] | None = None
    inputs: tuple[Path, ...] = ()
    optional: bool = False


def _hash_file(h: "hashlib._Hash", path: Path) -> None:
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            h.update(chunk)


def fingerprint(paths: Iterable[Path]) -> str:
    """POST: sha256 del contenuto di file e cartelle (ricorsivo, ordine stabile); un percorso assente conta come MISSING."""
    h = hashlib.sha256()
    for path in sorted(paths, key=str):
        h.update(str(path).encode())
        if path.is_dir():
            for f in sorted(x for x in path.rglob("*") if x.is_file()):
                h.update(str(f.relative_to(path)).encode())
                _hash_file(h, f)
        elif path.is_file():
            _hash_file(h, path)
        else:
            h.update(b"MISSING")
    return h.hexdigest()


def git_info() -> tuple[str | None, bool | None]:
    """POST: (commit, working tree sporco) oppure (None, None) se git non è disponibile."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=ROOT
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True, check=True, cwd=ROOT
            ).stdout.strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


# ---------------------------------------------------------------- manifest
def load_manifest(directory: Path) -> dict:
    path = directory / "manifest.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"stages": {}}


def save_manifest(directory: Path, manifest: dict) -> None:
    """POST: scrittura atomica."""
    directory.mkdir(parents=True, exist_ok=True)
    tmp = directory / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(tmp, directory / "manifest.json")


def is_done(stage: Stage, manifest: dict) -> bool:
    """POST: True se lo stadio è nel manifest, gli input esterni sono invariati e gli output esistono."""
    entry = manifest["stages"].get(stage.name)
    return bool(
        entry
        and entry["inputs_sha256"] == fingerprint(stage.inputs)
        and all(p.exists() for p in stage.outputs)
    )


def run_stage(stage: Stage, directory: Path) -> list[str] | None:
    """
    POST: esegue lo stadio. Con action la chiama (ritorna None); altrimenti lancia
          `python <args>` con COWS_RUN_DIR=directory, copiando l'output su schermo e in
          logs/<stadio>.log. Solleva RuntimeError se il comando termina con errore.
    """
    if stage.action is not None:
        stage.action()
        return None
    cmd = [sys.executable, *stage.args]
    log_dir = directory / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "COWS_RUN_DIR": str(directory), "PYTHONUNBUFFERED": "1"}
    with open(log_dir / f"{stage.name}.log", "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd, env=env, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        assert proc.stdout is not None
        pending = ""  # riga corrente per il log (l'ultimo \r vince)
        while True:
            chunk = proc.stdout.read(1)
            if not chunk:
                break
            char = chunk.decode("utf-8", errors="replace")
            sys.stdout.write(char)
            sys.stdout.flush()
            if char == "\n":
                log.write(pending + "\n")
                pending = ""
            elif char == "\r":
                pending = ""  # un aggiornamento di barra sostituisce il precedente
            else:
                pending += char
        if pending:
            log.write(pending + "\n")
        if proc.wait() != 0:
            raise RuntimeError(f"Stadio '{stage.name}' fallito (codice {proc.returncode})")


def execute(
    stages: list[Stage],
    directory: Path,
    *,
    only: str | None = None,
    from_stage: str | None = None,
    force: bool = False,
    dry_run: bool = False,
    runner: Callable[[Stage, Path], list[str] | None] = run_stage,
) -> None:
    """
    PRE:  stages nell'ordine di esecuzione.
    POST: esegue gli stadi selezionati (tutti, oppure solo `only`), saltando quelli già
          completati salvo `force`. `from_stage` invalida quello stadio e i successivi.
          Dopo l'esecuzione di uno stadio, i successivi vengono invalidati nel manifest.
          Con dry_run stampa solo lo stato. Uno stadio non opzionale che fallisce ferma tutto.
    """
    names = [s.name for s in stages]
    for label, name in (("--only", only), ("--from", from_stage)):
        if name is not None and name not in names:
            raise SystemExit(f"{label}: stadio sconosciuto '{name}'. Stadi: {names}")

    if force and only is None and from_stage is None:
        raise SystemExit("--force richiede --only STADIO oppure --from STADIO")
    forced = set(names[names.index(from_stage):]) if from_stage else set()
    if only is not None:
        forced = {only}
    
    directory.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(directory)
    if from_stage is not None:
        for name in names[names.index(from_stage):]:
            manifest["stages"].pop(name, None)
        if not dry_run:
            save_manifest(directory, manifest)

    for stage in stages:
        if only is not None and stage.name != only:
            continue
        done = is_done(stage, manifest) and not (force and stage.name in forced)
        if dry_run:
            print(f"[{'salta' if done else 'esegui'}] {stage.name}  {' '.join(stage.args)}")
            continue
        if done:
            print(f"[salta] {stage.name} (già completato)")
            continue

        print(f"\n=== STADIO {stage.name} ===")
        inputs_hash = fingerprint(stage.inputs)
        commit, dirty = git_info()
        started = datetime.now(UTC).isoformat(timespec="seconds")
        t0 = time.time()
        try:
            cmd = runner(stage, directory)
        except Exception as error:
            if stage.optional:
                print(f"ATTENZIONE: stadio opzionale '{stage.name}' fallito: {error}")
                continue
            raise SystemExit(f"{error}\nCorreggi e rilancia con --from {stage.name}") from error

        for later in names[names.index(stage.name) + 1:]:
            manifest["stages"].pop(later, None)
        manifest["stages"][stage.name] = {
            "command": cmd or "action interna",
            "git_commit": commit,
            "git_dirty": dirty,
            "started": started,
            "seconds": round(time.time() - t0, 1),
            "inputs_sha256": inputs_hash,
            "outputs_sha256": {str(p): fingerprint([p]) for p in stage.outputs},
        }
        save_manifest(directory, manifest)


# ---------------------------------------------------------------- azioni
def init_iteration(k: int, prev_best: str = "selected", prev_hybrid: bool = False) -> None:
    """
    POST: crea la cartella dell'iterazione k. k=0: seed = raw_states legacy. k>0: raw_states
          di k-1 e, come sorgenti di stati, gli alberi migliori di k-1
          (prev_estimated_evm_<best>.json e, se prev_hybrid, prev_hybrid_best_minimax.json).
          NON sovrascrive mai un raw_states.json già presente e non vuoto.
    """
    root = run_dir(k)
    for sub in ("trees", "models", "logs"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    if k == 0:
        seed = LEGACY_DIR / "raw_states.json"
    else:
        prev = run_dir(k - 1)
        seed = prev / "raw_states.json"
        names = [f"estimated_evm_{prev_best}"] + (["hybrid_best_minimax"] if prev_hybrid else [])
        for name in names:
            src = prev / "trees" / f"{name}.json"
            if src.exists():
                shutil.copyfile(src, root / "trees" / f"prev_{name}.json")
    target = root / "raw_states.json"
    if target.exists() and target.stat().st_size > 2:
        return
    if seed.exists():
        shutil.copyfile(seed, target)
    else:
        target.write_text("{}", encoding="utf-8")


def decide(ref_moves: int, moves: int, games: int, k: int) -> tuple[bool, str]:
    """
    POST: (continuare?, motivo). Si continua solo se (ref - corrente) / games >= MIN_IMPROVEMENT
          e k + 1 < MAX_ITERATIONS.
    """
    improvement = (ref_moves - moves) / games
    if improvement < MIN_IMPROVEMENT - 1e-12:
        return False, f"miglioramento {improvement:.6f} sotto la soglia {MIN_IMPROVEMENT}"
    if k + 1 >= MAX_ITERATIONS:
        return False, f"raggiunto il massimo di {MAX_ITERATIONS} iterazioni"
    return True, f"miglioramento {improvement:.6f} >= {MIN_IMPROVEMENT}"

def write_verdict(k: int) -> None:
    """
    PRE:  esistono alberi estimated_evm_<kind> di k. Riferimento: k=0 albero legacy;
          k>0 il miglior albero EVM di k-1 (verdict.json di k-1).
    POST: verdict.json con il miglior modello puro (usato per la regola di stop) e, se
          esiste, le statistiche dell'ibrido (solo informative: pesi tarati sugli stessi segreti).
    """
    root = run_dir(k)
    best_model, current = pick_best_model(root / "trees")
    if k == 0:
        ref_name, reference = "legacy", tree_stats(root / "trees" / LEGACY_TREE)
    else:
        prev = json.loads((run_dir(k - 1) / "verdict.json").read_text(encoding="utf-8"))
        ref_name, reference = f"iter_{k - 1:02d}", prev["best"]
    cont, reason = decide(reference["total_moves"], current["total_moves"], current["games"], k)

    hybrid_path = root / "trees" / "hybrid_best_minimax.json"
    hybrid = tree_stats(hybrid_path) if hybrid_path.exists() else None
    verdict = {
        "iteration": k, "best_model": best_model, "best": current,
        "reference_name": ref_name, "reference": reference,
        "improvement_avg": reference["avg"] - current["avg"],
        "hybrid": hybrid, "continue": cont, "reason": reason,
    }
    (root / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print(f"\nIterazione {k}: miglior modello {best_model}: media {current['avg']:.5f} "
          f"(caso peggiore {current['worst']}) | riferimento {ref_name}: {reference['avg']:.5f}")
    if hybrid:
        print(f"Ibrido (informativo): media {hybrid['avg']:.5f}, caso peggiore {hybrid['worst']}")
    print(f"Verdetto: {'CONTINUARE' if cont else 'FERMARSI'} - {reason}")
    update_final()


def update_final() -> None:
    """POST: data/runs/final.json = iterazione con meno mosse totali (miglior modello puro)."""
    verdicts = [json.loads(p.read_text(encoding="utf-8"))
                for p in sorted(RUNS_DIR.glob("iter_*/verdict.json"))]
    if not verdicts:
        return
    best = min(verdicts, key=lambda v: ((v.get("best") or v["selected"])["total_moves"], v["iteration"]))
    stats = best.get("best") or best["selected"]
    kind = best.get("best_model", "selected")
    final = {
        "best_iteration": best["iteration"], "best_model": kind,
        "avg": stats["avg"], "worst": stats["worst"],
        "model": str(run_dir(best["iteration"]) / "models" / f"{kind}.json"),
    }
    (RUNS_DIR / "final.json").write_text(json.dumps(final, indent=2), encoding="utf-8")

# ------------------------------------------------------------ definizione stadi
def build_stages(
    k: int, tournament_max_n: int = 1500, prev_best: str = "selected", prev_hybrid: bool = False
) -> list[Stage]:
    """
    POST: gli stadi dell'iterazione k. prev_best/prev_hybrid descrivono l'iterazione k-1
          (miglior modello e presenza dell'ibrido): entrano nel torneo di refine e come sorgenti di stati.
    """
    R = run_dir(k)
    P = run_dir(k - 1) if k > 0 else None
    trees, models = R / "trees", R / "models"
    gen = ("-m", "scripts.generate_trees", "--models-dir", str(models), "--strict-models", "--strategies")

    stages = [
        Stage(
            "init",
            outputs=(R / "raw_states.json",),
            action=lambda: init_iteration(k, prev_best, prev_hybrid),
            inputs=(LEGACY_DIR / "raw_states.json",)
            if P is None
            else (P / "raw_states.json", P / "trees" / f"estimated_evm_{prev_best}.json"),
        )
    ]
    if k == 0:
        stages.append(Stage("baseline", outputs=(trees / LEGACY_TREE,), args=(*gen, "estimated_evm_legacy")))

    base_trees = ("max_entropy", "greedy_minimax", "hybrid_entropy_minimax")
    stages += [
        Stage("trees_base", outputs=tuple(trees / f"{n}.json" for n in base_trees), args=(*gen, *base_trees)),
        Stage("mine", outputs=(R / "raw_states.json",), args=("-m", "experiments.oracle_lab.data_miner")),
    ]
    if k == 0:  # augment costa molto e non migliora il target dei bucket grandi: solo in iter 0
        stages.append(Stage("augment", outputs=(R / "raw_states.json",),
                            args=("-m", "experiments.oracle_lab.augment_states", "--seed", str(1000 + k))))

    competitors = ["max_entropy", "greedy_minimax"]
    refine = ["-m", "experiments.oracle_lab.dataset_refiner", "--max-tournament-n", str(tournament_max_n)]
    refine_inputs: tuple[Path, ...] = ()
    if P is not None:
        competitors.append(f"estimated_evm_{prev_best}")
        if prev_hybrid:
            competitors.append("hybrid_best_minimax")
        refine += ["--models-dir", str(P / "models"), "--strict-models"]
        refine_inputs = (P / "models",)
    refine += ["--competitors", *competitors]
    stages.append(Stage("refine", outputs=(R / "raw_states.json",), args=tuple(refine), inputs=refine_inputs))

    train = ("-m", "experiments.oracle_lab.train_model")
    stages += [
        Stage("features", outputs=(R / "features_dataset.json",),
              args=("-m", "experiments.oracle_lab.feature_builder")),
        Stage("eda", outputs=(R / "logs" / "eda.log",), optional=True,
              args=("-m", "experiments.oracle_lab.exploratory_analysis")),
        Stage("select_model", outputs=(models / "selected.json",),
              args=("-m", "experiments.oracle_lab.select_model", "--out", str(models / "selected.json"))),
        Stage("train_geometric", outputs=(models / "geometric.json",),
              args=(*train, "--splits", "--out", str(models / "geometric.json"))),
        Stage("train_piecewise", outputs=(models / "piecewise.json",),
              args=(*train, "--out", str(models / "piecewise.json"))),
        Stage("regret", outputs=(R / "regret.json",), optional=True,
              args=("-m", "experiments.oracle_lab.evaluate_regret", "--max-states", "300",
                    "--models", "legacy_piecewise", str(models / "geometric.json"),
                    str(models / "piecewise.json"), str(models / "selected.json"),
                    "--out", str(R / "regret.json"))),
    ]

    model_trees = ("estimated_evm_selected", "estimated_evm_piecewise",
                   "estimated_evm_geometric", "hybrid_evm_minimax")
    stages += [
        Stage("trees_model", outputs=tuple(trees / f"{n}.json" for n in model_trees), args=(*gen, *model_trees)),
        Stage("tune_hybrid", outputs=(models / "hybrid_best.json",),
              args=("-m", "experiments.tune_best_hybrid")),
        # opzionale: se tune_hybrid ha saltato, la strategia non esiste e lo stadio fallisce senza danni
        Stage("trees_hybrid", outputs=(trees / "hybrid_best_minimax.json",), optional=True,
              args=(*gen, "hybrid_best_minimax")),
        Stage("verdict", outputs=(R / "verdict.json",), action=lambda: write_verdict(k)),
    ]
    return stages


def main() -> None:
    parser = argparse.ArgumentParser(description="Esegue una iterazione della pipeline.")
    parser.add_argument("--iter", type=int, required=True, dest="iteration")
    parser.add_argument("--only", default=None, help="esegue solo questo stadio")
    parser.add_argument("--from", default=None, dest="from_stage", help="invalida e riesegue da questo stadio")
    parser.add_argument("--force", action="store_true", help="non salta gli stadi già completati")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tournament-max-n", type=int, default=1500)
    args = parser.parse_args()

    k = args.iteration
    if not 0 <= k < MAX_ITERATIONS:
        raise SystemExit(f"--iter deve essere tra 0 e {MAX_ITERATIONS - 1}")
    if k > 0 and not (run_dir(k - 1) / "verdict.json").exists():
        raise SystemExit(f"Manca {run_dir(k - 1) / 'verdict.json'}: completa prima l'iterazione {k - 1}.")
    prev_best, prev_hybrid = "selected", False
    if k > 0:
        previous = json.loads((run_dir(k - 1) / "verdict.json").read_text(encoding="utf-8"))
        prev_best = previous.get("best_model", "selected")
        prev_hybrid = previous.get("hybrid") is not None
        if not previous["continue"]:
            print(f"ATTENZIONE: il verdetto di iter_{k - 1:02d} era FERMARSI ({previous['reason']}).")

    os.chdir(ROOT)
    execute(build_stages(k, args.tournament_max_n, prev_best, prev_hybrid), run_dir(k),
            only=args.only, from_stage=args.from_stage, force=args.force, dry_run=args.dry_run)

    verdict_path = run_dir(k) / "verdict.json"
    if args.only is None and not args.dry_run and verdict_path.exists():
        verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
        if verdict["continue"]:
            print(f"\nPer proseguire, dopo aver guardato i numeri:\n"
                  f"  python -m scripts.run_pipeline --iter {k + 1}")
        else:
            print("\nPipeline iterativa conclusa. Modello migliore: data/runs/final.json")


if __name__ == "__main__":
    main()