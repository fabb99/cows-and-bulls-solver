"""
Modulo: train_oracle (Script)
Laboratorio di Data Science per addestrare l'Oracolo Entropico.
Raccoglie stati reali di gioco, calcola l'Entropia e il Costo Esatto,
e applica una regressione lineare per trovare il moltiplicatore 'K' ottimale.
"""
from typing import List, Tuple
import math
import random
import time

from tqdm import tqdm
# Importa qui le tue strategie: Naive, EntropyStrategy, PureDeepEVMStrategy (o simili)
# Importa generator e lookup
from src.core.generator import generate_all_permutations
from src.core.lookup_table import FeedbackLookup
from src.core.engine import filter_solutions
from src.strategies.basic import (NaiveStrategy,MaxFrequencyStrategy)
from src.strategies.deep_emv import PureDeepEVMStrategy
from src.strategies.entropy import EntropyStrategy

# Il tipo di dato che estrarremo. Tuple[N, H, M]
# N = Dimensione Bucket, H = Entropia della miglior mossa, M = Costo Esatto (EVM)
DataPoint = Tuple[int, float, float]

def generate_random_states(
    num_samples: int, 
    max_n: int, 
    lookup, 
    all_perms: List[str]
) -> List[List[str]]:
    """
    Simula partite casuali usando una strategia Naive per generare stati di gioco "veri".    
    PRE: num_samples indica quante partite simulare. max_n è il limite di soluzioni 
         accettabili per il campione (es. scartiamo N=5000 perché il DeepEVM ci metterebbe anni).
    POST: Ritorna una lista di insiemi di soluzioni (stati). Ogni stato ha lunghezza <= max_n e > 2.
    Raccoglie stati unici simulando partite, evitando doppioni tramite frozenset.
    """
    unique_states_keys = set()
    states = []
    strategy = NaiveStrategy(all_perms, lookup)

    with tqdm(total=num_samples, desc="Campionamento stati unici") as pbar:
        while len(states) < num_samples:
            secret = random.choice(all_perms)
            solutions = all_perms
            
            while True:
                guess = strategy.choose_guess(solutions)
                if guess == secret:
                    break

                solutions = filter_solutions(guess, lookup.get_feedback(guess, secret), solutions, lookup)
                num_solutions = len(solutions)

                if 2 < num_solutions <= max_n:
                    key = frozenset(solutions)
                    if key not in unique_states_keys:
                        unique_states_keys.add(key)
                        states.append(solutions)
                        pbar.update(1)
                    break 

    return states

def generate_random_states(
    num_samples: int, 
    max_n: int, 
    lookup, 
    all_perms: List[str]
) -> List[List[str]]:
    """
    Simula partite casuali usando una strategia Naive per generare stati di gioco "veri".
    
    PRE: num_samples indica quante partite simulare. max_n è il limite di soluzioni 
         accettabili per il campione (es. scartiamo N=5000 perché il DeepEVM ci metterebbe anni).
    POST: Ritorna una lista di insiemi di soluzioni (stati). Ogni stato ha lunghezza <= max_n e > 2.
    """
    unique_keys = set()
    states = []
    strategy = EntropyStrategy(all_perms, lookup)

    with tqdm(total=num_samples, desc="1/2 Raccolta Stati", unit="stati") as pbar:
        while len(states) < num_samples:
            secret = random.choice(all_perms)
            solutions = all_perms
            
            while True:
                guess = strategy.choose_guess(solutions)
                if guess == secret:
                    break

                solutions = filter_solutions(guess, lookup.get_feedback(guess, secret), solutions, lookup)
                num_solutions = len(solutions)

                # Se abbiamo trovato un bucket della dimensione desiderata
                if 2 < num_solutions <= max_n:
                    key = frozenset(solutions)
                    if key not in unique_keys:
                        unique_keys.add(key)
                        states.append(solutions)
                        pbar.update(1)
                    break # Passa alla prossima partita

    return states

def extract_metrics_for_state_old(
    state: List[str], 
    entropy_strategy, 
    exact_strategy
) -> DataPoint:
    """
    Per un dato stato di gioco (es. 15 soluzioni), calcola i parametri dell'equazione.
    
    PRE: state è una lista di soluzioni rimaste. Entrambe le strategie sono istanziate.
    POST: Ritorna la tripla (N, H, M).
          - N = len(state)
          - H = l'entropia della miglior mossa scelta da entropy_strategy.
          - M = il costo esatto calcolato da exact_strategy._compute_exact_cost(state).
    """
    #return (DataPoint (len(state), max(entropy_strategy.evaluate_candidates(state,entropy_strategy.all_permutations).values()), exact_strategy._compute_exact_cost(state)))

    candidates_scores = entropy_strategy.evaluate_candidates(entropy_strategy.all_permutations, state)
    max_entropy = max(candidates_scores.values())
    exact_cost = exact_strategy._compute_exact_cost(state)
    return (len(state), max_entropy, exact_cost)
    
def calculate_k_constant_old(dataset: List[DataPoint]) -> float:
    """
    Applica il metodo dei Minimi Quadrati per trovare la costante K che minimizza
    la differenza tra [1 + K * (log2(N) - H)] e [M].
    
    PRE: dataset è la lista delle triple raccolte.
    POST: Ritorna la costante 'K' (float) ottimizzata e stampa le statistiche di errore (MSE).
    """
    sum_XY = 0
    sum_X2 = 0

    for n, h, m in dataset:
        y = m - 1
        x = math.log2(n)-h

        sum_XY += x*y
        sum_X2 += x**2

    return sum_XY/sum_X2

def extract_metrics_for_state(
    state: List[str], 
    entropy_strategy: EntropyStrategy, 
    exact_strategy: PureDeepEVMStrategy,
    pbar=None
) -> DataPoint:
    """
    Estrae le 4 metriche aggiornando lo stato testuale sulla barra principale.
    
    PRE: state, strategie valide.
    POST: Ritorna la tupla (N, Entropia_Residua, N_max, M).
          - N: len(state).
          - Entropia_Residua: (log2(N) - H) calcolata sulla mossa migliore.
          - N_max: La len() massima tra i bucket della mossa migliore.
          - M: il costo esatto dal DeepEVM.
    """
    # 1. Trova N e notifica la barra
    
    # 2. Ottieni la mossa suggerita dall'entropia
    # NOTA: Devi usare .choose_guess() per trovare *quale* candidata è la migliore,
    # poi usi calculate_buckets su quella candidata per calcolare H e N_max.
    
    # 3. Calcola l'Entropia_Residua e N_max per quella specifica mossa
    
    # 4. Calcola M esatto con exact_strategy e ritorna la tupla
    pass

    entropy_strategy.choose_guess(state,["0123"])#voglio evitare che faccia le firme 
    exact_cost = exact_strategy._compute_exact_cost(state)

    return (len(state), candidates_scores, )

def calculate_k_constants(dataset: List[DataPoint]) -> Tuple[float, float]:
    """
    Applica i Minimi Quadrati Multipli (NumPy) per risolvere:
    (M - 1) = K1 * EntropiaResidua + K2 * N_max
    
    PRE: dataset è una lista valida di DataPoints.
    POST: Ritorna (K1, K2).
    """
    # 1. Crea le liste (vettori) per Y (M - 1.0)
    # 2. Crea la matrice X dove ogni riga è [EntropiaResidua, N_max]
    # 3. Usa np.linalg.lstsq(X, Y, rcond=None)[0] per ottenere K1 e K2
    pass

def main():
    print("Inizializzazione ambiente...")
    all_perms = generate_all_permutations()
    lookup = FeedbackLookup(all_perms)
    lookup.get_feedback(all_perms[0], all_perms[0]) 

    entropy_strategy = EntropyStrategy(all_perms, lookup)
    exact_strategy = PureDeepEVMStrategy(all_perms, lookup, entropy_strategy)

    # 1. Raccolta di stati unici con barra
    target_samples = 80
    max_solutions_threshold = 20
    random_states = generate_random_states(target_samples, max_solutions_threshold, lookup, all_perms)

    # 2. Estrazione metriche con barra di avanzamento
    dataset: List[DataPoint] = []
    for state in tqdm(random_states, desc="Analisi EVM ed Entropia"):
        metrics = extract_metrics_for_state(state, entropy_strategy, exact_strategy)
        dataset.append(metrics)

    # 3. Risoluzione della costante
    k_value = calculate_k_constant(dataset)
    print(f"\n==========================================")
    print(f" Costante K Ottimale Trovata: {k_value:.4f}")
    print(f" Formula: Stima = 1.0 + {k_value:.4f} * (log2(N) - H)")
    print(f"==========================================")


if __name__ == "__main__":
    main()