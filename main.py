"""
Punto d'ingresso per Cows and Bulls (Mastermind a 4 cifre distinte).
Uso:
  python main.py --list
  python main.py --strategy max_entropy               # partita interattiva
  python main.py --strategy max_entropy --secret 9835 # autoplay
"""
import argparse
import time

from src.cli.game_loop import play_autoplay, play_interactive
from src.core.generator import generate_all_permutations
from src.core.lookup_table import FeedbackLookup
from src.paths import MODELS_DIR, ROUTING_POLICY_FILEPATH
from src.strategies.zoo import build_base_strategies, build_routed_strategies


def main() -> None:
    parser = argparse.ArgumentParser(description="Cows and Bulls solver")
    parser.add_argument("--strategy", default="max_entropy")
    parser.add_argument("--secret", default=None, help="se indicato, gioca in autoplay")
    parser.add_argument("--list", action="store_true", help="elenca le strategie disponibili")
    args = parser.parse_args()

    start = time.time()
    all_perms = generate_all_permutations()
    lookup = FeedbackLookup(all_perms)
    lookup.get_feedback(all_perms[0], all_perms[0])
    print(f"Sistema pronto ({time.time() - start:.1f} s)")

    available = build_base_strategies(all_perms, lookup, MODELS_DIR)
    available.update(build_routed_strategies(all_perms, lookup, available, ROUTING_POLICY_FILEPATH))

    if args.list:
        print("\n".join(sorted(available)))
        return
    if args.strategy not in available:
        raise SystemExit(f"Strategia sconosciuta: {args.strategy}. Usa --list.")

    strategy = available[args.strategy]
    print(f"--- Strategia: {args.strategy} ---")
    if args.secret:
        play_autoplay(strategy, args.secret, all_perms)
    else:
        play_interactive(strategy, all_perms)


if __name__ == "__main__":
    main()