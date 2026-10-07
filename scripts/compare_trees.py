"""
Modulo: compare_trees (Script)
Legge gli alberi salvati in formato JSON e stampa una tabella comparativa estensibile.
"""
from pathlib import Path
from src.analysis.tree_io import load_tree_from_json
from src.analysis.metrics import (max_depth, average_depth, get_depth_distribution, get_depth_distribution_next_move)
from src.cli.game_loop import fb_tuple_to_string


# Definiamo una lista di "Metriche" da calcolare. 
# Sono semplicemente le funzioni che hai già scritto in src/analysis/metrics.py!
# Se in futuro ne aggiungi una, ti basta aggiungerla a questa lista.
METRICS_PIPELINE = [
    ("Caso Peggiore", max_depth),
    ("Media Mosse", average_depth),
    ("Distribuzione", get_depth_distribution),
    ("Distribuzione dopo prima mossa", get_depth_distribution_next_move)
]

def analyze_tree(tree: dict) -> dict[str, any]:
    """
    PRE: tree è un albero caricato da JSON.
    POST: Esegue tutte le metriche definite in METRICS_PIPELINE sull'albero
          e restituisce un dizionario report.
          Es: {"Caso Peggiore": 7, "Media Mosse": 5.52, ...}
    """
    report = {}
    for name, metric in METRICS_PIPELINE:
        report[name] = metric(tree) 
    return report

def print_comparison_dashboard(reports: dict[str, dict[str, any]]):
    """
    PRE: reports è un dizionario {nome_strategia: report_calcolato}.
    POST: Stampa in console una tabella ASCII formattata e allineata.
    """
    if not reports:
        print("Nessun albero trovato da confrontare!")
        return

    # Troviamo la massima profondità registrata tra tutte le strategie per dimensionare le colonne dei turni
    max_turns = max(report["Caso Peggiore"] for report in reports.values())

    print("\n" + "=" * 80)
    print(" DASHBOARD COMPARATIVA STRATEGIE (Cows and Bulls)")
    print("=" * 80)

    # 1. TABELLA RIASSUNTIVA DEI KPI
    print(f"{'Strategia':<20} | {'Caso Peggiore':<15} | {'Media Mosse':<15}")
    print("-" * 56)
    for name, data in reports.items():
        print(f"{name:<20} | {data['Caso Peggiore']:<15} | {data['Media Mosse']:<15.3f}")

    print("\n" + "=" * 80)
    print(" DETTAGLIO PARTITE RISOLTE PER NUMERO DI MOSSE")
    print("=" * 80)

    # 2. TABELLA DI DISTRIBUZIONE TURNO PER TURNO
    # Intestazione: Strategia | T1 | T2 | T3 | ...
    header = f"{'Strategia':<20} | " + " | ".join([f"T{i:<3}" for i in range(1, max_turns + 1)])
    print(header)
    print("-" * len(header))

    for name, data in reports.items():
        dist = data["Distribuzione"]
        # Estraiamo quante partite sono state vinte a ciascun turno (0 se nessuna)
        row_values = [f"{dist.get(i, 0):<4}" for i in range(1, max_turns + 1)]
        print(f"{name:<20} | " + " | ".join(row_values))
        
    print("=" * len(header) + "\n")

    print_comparison_dashboard_extend(reports)
    #print_vertical_distribution_dashboard(reports)

def print_comparison_dashboard_extend(reports: dict[str, dict[str, any]]):

    first_strategy_data = next(iter(reports.values()))
    available_feedbacks = first_strategy_data["Distribuzione dopo prima mossa"].keys()

    
    for fb in available_feedbacks:

        this_max_turns = 8
        print("\n" + "=" * 80)
        print(f" DETTAGLIO PARTITE RISOLTE PER NUMERO DI MOSSE DOPO LA VALUTAZIONE |{fb_tuple_to_string(fb)}|")
        print("=" * 80)

        # 2. TABELLA DI DISTRIBUZIONE TURNO PER TURNO
        # Intestazione: Strategia | T1 | T2 | T3 | ...
        header = f"{'Strategia':<20} | " + " | ".join([f"T{i+1:<3}" for i in range(1, this_max_turns )])
        print(header)
        print("-" * len(header))

        for name, data in reports.items():
            dist = data["Distribuzione dopo prima mossa"][fb]
            # Estraiamo quante partite sono state vinte a ciascun turno (0 se nessuna)
            row_values = [f"{dist.get(i, 0):<4}" for i in range(1, this_max_turns )]
            print(f"{name:<20} | " + " | ".join(row_values))
            
        print("=" * len(header) + "\n")

def main():
    trees_dir = Path("data/trees")
    
    if not trees_dir.exists():
        print(f"Errore: La cartella {trees_dir} non esiste!")
        return

    # Trova tutti i file .json nella cartella
    json_files = sorted(trees_dir.glob("*.json"))

    if not json_files:
        print("Nessun albero trovato in data/trees/. Esegui prima scripts/generate_trees.py!")
        return

    reports = {}
    print(f"Caricamento e analisi di {len(json_files)} alberi in corso...")

    for file_path in json_files:
        strategy_name = file_path.stem # Prende il nome del file senza estensione .json
        print(f"- Analisi di {strategy_name}...")
        tree = load_tree_from_json(str(file_path))
        reports[strategy_name] = analyze_tree(tree)

    # Mostra i risultati
    print_comparison_dashboard(reports)


if __name__ == "__main__":
    main()