"""
Punto d'ingresso principale per Cows and Bulls (Mastermind Solver).
"""
import time
from src.core.generator import generate_all_permutations
from src.core.lookup_table import FeedbackLookup
from src.cli.game_loop import play_autoplay, play_interactive

# Import delle strategie
from src.strategies.basic import NaiveStrategy, MaxFrequencyStrategy
from src.strategies.greedy_minimax import GreedyMinimaxStrategy
from src.strategies.entropy import EntropyStrategy
from src.strategies.hybrid import HybridWeightedStrategy

def main():
    print("Avvio del Sistema Cows and Bulls...")
    
    # 1. Generazione Universo e Lookup Table
    print("Generazione permutazioni e Lookup Table in corso (Lazy)...")
    start_time = time.time()
    
    all_perms = generate_all_permutations()
    lookup = FeedbackLookup(all_perms)
    
    # OPZIONALE: Forziamo la creazione della tabella subito così non rallenta la prima mossa
    # (Altrimenti il Turno 1 della partita ci metterebbe 1 secondo in più per caricare)
    lookup.get_feedback(all_perms[0], all_perms[0]) 
    
    print(f"Sistema pronto! ({time.time() - start_time:.2f} sec)\n")


    # 2. Scelta della Strategia (De-commenta quella che vuoi usare!)
    
    # strategy = NaiveStrategy(all_perms, lookup)
    # strategy = MaxFrequencyStrategy(all_perms, lookup)
    # strategy = GreedyMinimaxStrategy(all_perms, lookup)
    strategy = EntropyStrategy(all_perms, lookup)
    
    # ESEMPIO IBRIDA:
    # strategy_a = EntropyStrategy(all_perms, lookup)
    # strategy_b = GreedyMinimaxStrategy(all_perms, lookup)
    # strategy = HybridWeightedStrategy(strategy_a, strategy_b, True, False, 0.94, 0.06)


    # 3. Avvio della Partita
    print(f"--- Strategia selezionata: {strategy.__class__.__name__} ---")
    
    # Scegli se giocare interattivamente o in autoplay
    
    # play_interactive(strategy, all_perms)
    
    segreto_da_indovinare = "9835"
    print(f"Inizio Autoplay contro il segreto: {segreto_da_indovinare}")
    mosse = play_autoplay(strategy, segreto_da_indovinare, all_perms)


if __name__ == "__main__":
    main()