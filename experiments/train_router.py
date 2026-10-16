"""
Modulo: train_router (Script)
Router a programmazione dinamica per stato. In ogni stato i competitor propongono una
mossa; per ogni mossa distinta i figli si risolvono ricorsivamente con lo stesso router
(memoizzazione per stato). Due obiettivi:
  - best_average:     minimizza (mosse totali, caso peggiore)
  - pareto_defensive: minimizza (caso peggiore, mosse totali)
Garanzia: il risultato non è mai peggiore del miglior competitor sull'obiettivo primario.
Nota: per la difesa il totale è un'approssimazione (ogni figlio ottimizza la propria
coppia lessicograficamente); il caso peggiore è invece esatto.

Uso: python -m experiments.train_router [--competitors NOME ...]
"""
import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from tqdm import tqdm

from src.core.constants import WIN_FEEDBACK
from src.core.engine import calculate_buckets
from src.core.generator import generate_all_permutations
from src.core.lookup_table import FeedbackLookup
from src.core.state import state_hash
from src.paths import MODELS_DIR, ROUTING_POLICY_FILEPATH
from src.strategies.base import BaseStrategy
from src.strategies.zoo import build_base_strategies

DEFAULT_COMPETITORS = [
    "max_entropy",
    "estimated_evm_piecewise",
    "estimated_evm_geometric",
    "hybrid_evm_minimax",
    "hybrid_entropy_minimax",
]


@dataclass(frozen=True)
class Plan:
    """Mossa scelta in uno stato e suo costo: mosse totali su tutte le partite e profondità."""

    total: int
    depth: int
    guess: str


class RouterSolver:
    def __init__(
        self,
        competitors: dict[str, BaseStrategy],
        lookup: FeedbackLookup,
        on_new_state: Callable[[], None] | None = None,
    ) -> None:
        """PRE: competitors non vuoto; tutti condividono lookup e universo."""
        if not competitors:
            raise ValueError("Serve almeno un competitor")
        self.competitors = competitors
        self.lookup = lookup
        self.on_new_state = on_new_state
        self._memo: dict[str, tuple[Plan, Plan]] = {}

    @property
    def n_states(self) -> int:
        return len(self._memo)

    def solve(self, solutions: list[str]) -> tuple[Plan, Plan]:
        """
        PRE:  solutions non vuota.
        POST: (piano d'attacco, piano difensivo) dello stato. Con N <= 2 la mossa è la
              prima soluzione (come in build_decision_tree): costo 1 mossa / 3 mosse totali.
        """
        n = len(solutions)
        if n == 1:
            plan = Plan(1, 1, solutions[0])
            return plan, plan
        if n == 2:
            plan = Plan(3, 2, solutions[0])
            return plan, plan

        key = state_hash(solutions)
        if key in self._memo:
            return self._memo[key]

        in_solutions = set(solutions)
        proposals = sorted({s.choose_guess(solutions, []) for s in self.competitors.values()})

        best_avg: tuple[tuple, Plan] | None = None
        best_def: tuple[tuple, Plan] | None = None
        for guess in proposals:
            buckets = calculate_buckets(guess, solutions, self.lookup)
            if len(buckets) == 1 and WIN_FEEDBACK not in buckets:
                continue  # non partiziona
            tot_a, dep_a, tot_d, dep_d = n, 1, n, 1
            for fb, bucket in buckets.items():
                if fb == WIN_FEEDBACK:
                    continue
                child_a, child_d = self.solve(bucket)
                tot_a += child_a.total
                dep_a = max(dep_a, 1 + child_a.depth)
                tot_d += child_d.total
                dep_d = max(dep_d, 1 + child_d.depth)
            not_solution = guess not in in_solutions
            key_a = (tot_a, dep_a, not_solution, guess)
            key_d = (dep_d, tot_d, not_solution, guess)
            if best_avg is None or key_a < best_avg[0]:
                best_avg = (key_a, Plan(tot_a, dep_a, guess))
            if best_def is None or key_d < best_def[0]:
                best_def = (key_d, Plan(tot_d, dep_d, guess))

        if best_avg is None or best_def is None:
            raise RuntimeError(f"Nessuna mossa valida proposta per uno stato di N={n}")
        result = (best_avg[1], best_def[1])
        self._memo[key] = result
        if self.on_new_state is not None:
            self.on_new_state()
        return result

    def plan(self, solutions: list[str], objective: str) -> Plan:
        """PRE: objective in {'avg', 'def'}. POST: piano memoizzato dello stato."""
        avg_plan, def_plan = self.solve(solutions)
        return avg_plan if objective == "avg" else def_plan


def extract_policy(solver: RouterSolver, root: list[str], objective: str) -> dict[str, str]:
    """
    PRE:  root già risolto da solver (o risolvibile).
    POST: {state_hash: mossa} per i soli stati con N >= 3 raggiungibili seguendo la policy.
    """
    policy: dict[str, str] = {}
    stack = [root]
    while stack:
        solutions = stack.pop()
        if len(solutions) <= 2:
            continue
        key = state_hash(solutions)
        if key in policy:
            continue
        plan = solver.plan(solutions, objective)
        policy[key] = plan.guess
        for fb, bucket in calculate_buckets(plan.guess, solutions, solver.lookup).items():
            if fb != WIN_FEEDBACK:
                stack.append(bucket)
    return policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Router a programmazione dinamica.")
    parser.add_argument("--competitors", nargs="+", default=DEFAULT_COMPETITORS)
    parser.add_argument("--out", type=str, default=str(ROUTING_POLICY_FILEPATH))
    args = parser.parse_args()

    print("Inizializzazione ambiente e Lookup Table...")
    all_perms = generate_all_permutations()
    lookup = FeedbackLookup(all_perms)
    lookup.get_feedback(all_perms[0], all_perms[0])

    base = build_base_strategies(all_perms, lookup, MODELS_DIR)
    unknown = [c for c in args.competitors if c not in base]
    if unknown:
        raise SystemExit(f"Competitor sconosciuti: {unknown}. Disponibili: {sorted(base)}")
    competitors = {name: base[name] for name in args.competitors}
    print(f"Competitor: {list(competitors)}\n")

    with tqdm(desc="Stati risolti", unit=" stati") as pbar:
        solver = RouterSolver(competitors, lookup, on_new_state=pbar.update)
        plan_avg, plan_def = solver.solve(all_perms)

    n = len(all_perms)
    print(f"\nAttacco : media {plan_avg.total / n:.5f} | caso peggiore {plan_avg.depth}")
    print(f"Difesa  : media {plan_def.total / n:.5f} | caso peggiore {plan_def.depth}")

    output = {
        "meta": {
            "created": datetime.now(UTC).isoformat(timespec="seconds"),
            "competitors": list(competitors),
            "states_solved": solver.n_states,
            "attack": {"avg": plan_avg.total / n, "worst": plan_avg.depth},
            "defense": {"avg": plan_def.total / n, "worst": plan_def.depth},
        },
        "best_average": extract_policy(solver, all_perms, "avg"),
        "pareto_defensive": extract_policy(solver, all_perms, "def"),
    }
    from pathlib import Path

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=1)
    print(f"Policy salvata in {out}")
    print(f"- stati in policy (attacco): {len(output['best_average'])}")
    print(f"- stati in policy (difesa) : {len(output['pareto_defensive'])}")


if __name__ == "__main__":
    main()