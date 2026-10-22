# Cows and Bulls Solver

Un laboratorio per studiare Cows and Bulls (Mastermind a 4 cifre distinte): una collezione di strategie, gli strumenti per misurare le performance su tutte le 5040 partite possibili e per guardarne gli alberi decisionali.

<!-- TODO: immagine di copertina: un sunburst di max_entropy (vedi sezione "Vedere gli alberi") -->

## Com'è nato

La domanda da cui parte il progetto è semplice: **quanto può essere buona una strategia greedy?** Una strategia greedy sceglie la mossa guardando solo un passo avanti, valutando come quella mossa divide le soluzioni ancora possibili.[non voglio ricalcolare un albero perfetto per poi giocare in base alle mosse che mi sono calcolato dellalbero decisionale]

Ogni strategia viene giudicata con due numeri: **media delle mosse** e **caso peggiore**.

## Cosa contiene

| Parte | Cosa fa | Dove |
|---|---|---|
| **Strategie** | Naive, frequenze, entropia, minimax, ibridi pesati, EVM esatta, EVM stimata, router per stato, ricetta "Frankenstein" | `src/strategies/` |
| **Generazione di alberi** | Gioca una strategia contro tutti i 5040 segreti e salva l'albero decisionale completo in JSON | `scripts/generate_trees.py` |
| **Analisi** | Confronto tra alberi (media, caso peggiore, distribuzione dei turni, ramo per ramo), autopsia di un ramo, invarianza rispetto alla prima mossa | `scripts/compare_trees.py`, `experiments/investigate_branch.py`, `experiments/first_move_invariance.py` |
| **Visualizzazione** | Sunburst, icicle e treemap interattivi degli alberi, con confronto tra due strategie | `scripts/visualize_tree.py` |
| **Laboratorio dei regressori lineari** | Estrae gli stati dagli alberi, calcola feature, seleziona soglie e feature per fase, addestra e valuta i modelli (OLS ed expectile) | `experiments/oracle_lab/` |
| **Pipeline** | Orchestra un'iterazione del laboratorio in stadi, con manifest e skip idempotente | `scripts/run_pipeline.py` |
| **Tuning e router** | Ricerca dei pesi degli ibridi; programmazione dinamica per scegliere la strategia stato per stato | `experiments/tune_*.py`, `experiments/train_policy_router.py` |

Le strategie condividono un'unica interfaccia (`BaseStrategy`): per aggiungerne una basta implementare `evaluate_candidates` e registrarla in `src/strategies/zoo.py`.

```mermaid
flowchart LR
    S[Strategia] --> T["Albero decisionale<br/>5040 partite"]
    T --> A[Confronto e analisi]
    T --> V[Visualizzazione]
    T --> D["Stati e costi noti"]
    D --> R["Regressore lineare<br/>(oracolo)"]
    R --> S2["Nuova strategia<br/>EVM stimata"]
    S2 -- "iterazione successiva" --> T
```

## Quick start

Serve Python 3.10 o superiore.

```bash
python -m venv venv
source venv/bin/activate
pip install numpy tqdm plotly
pip install pytest ruff        # sviluppo e test
python -m pytest
```

Tutti i comandi si lanciano dalla radice con `python -m ...`; il progetto non si installa come pacchetto.

Giocare contro il solver:

```bash
python main.py --list
python main.py --strategy max_entropy                 # partita interattiva
python main.py --strategy max_entropy --secret 9835   # partita automatica
```

In modalità interattiva si risponde con `+` per ogni toro, `-` per ogni mucca, `0` per nessuna corrispondenza, `q` per uscire.

> **Prima di partire.** [la funzionalità di gioco non è la principale...questa "ottimizzazione" deriva dal fatto che il progetto non nasce tanto per giocare ma per calcolare/trovare buone strategie]Al primo avvio il programma costruisce la tabella dei feedback di tutte le coppie: qualche decina di secondi e circa 1 GB di RAM. Non è bloccato.
> **Strategie disponibili.** Senza aver eseguito la pipeline hai le strategie base (`naive`, `max_frequency`, `greedy_minimax`, `max_entropy`, `hybrid_entropy_minimax`, `estimated_evm_legacy`). Quelle con modelli, router e ricetta compaiono in `--list` solo se i rispettivi file esistono in `data/`, che non è versionata.

Esempio:
```bash
> python main.py --strategy max_entropy --secret 9876
Sistema pronto (16.0 s)
--- Strategia: max_entropy ---
1. 0123 -> 0
2. 4567 -> --
3. 5689 -> ---
4. 8975 -> +--
5. 8796 -> +---
6. 9876 -> ++++
Risolto in 6 tentativi
```

## Generare e confrontare gli alberi

```bash
python -m scripts.generate_trees --strategies max_entropy greedy_minimax hybrid_entropy_minimax
python -m scripts.compare_trees --dir data/runs/iter_00/trees
python -m scripts.compare_trees --dir data/runs/iter_00/trees --branches
```

`--branches` ripete il confronto per ogni feedback del primo turno, con turni relativi al sotto-albero: è un modo rapido per vedere *dove* due strategie divergono.

```bash
> python -m scripts.compare_trees --trees max_entropy naive max_frequency

==================================================================================
 CONFRONTO STRATEGIE 
==================================================================================
Strategia                    | Caso peggiore | Media mosse
----------------------------------------------------------
max_entropy                  | 8             | 5.24286    
max_frequency                | 8             | 5.41647    
naive                        | 9             | 5.56032    

Strategia                    | T1    | T2    | T3    | T4    | T5    | T6    | T7    | T8    | T9   
-----------------------------------------------------------------------------------------------------
max_entropy                  | 1     | 5     | 62    | 611   | 2483  | 1779  | 98    | 1     | 0    
max_frequency                | 1     | 13    | 108   | 615   | 1881  | 1893  | 507   | 22    | 0    
naive                        | 1     | 13    | 108   | 596   | 1668  | 1768  | 752   | 129   | 5 
```


## Vedere gli alberi

`visualize_tree` trasforma un albero in una figura interattiva (HTML): ogni nodo è uno spicchio o un rettangolo la cui **area** è proporzionale al numero di soluzioni rimaste e il cui **colore** indica quanto è difficile il sotto-albero. Un bordo spesso segnala i rami che contengono le partite più lunghe.

```bash
python -m scripts.visualize_tree max_entropy
```

### Una strategia, con i nodi colorati per media delle mosse

<!-- TODO: assets/sunburst_max_entropy.png -->
![Sunburst di max_entropy](assets/sunburst_max_entropy.png)

```bash
python -m scripts.visualize_tree max_entropy --kind sunburst --color absolute
```

I rami più chiari si chiudono presto; quelli scuri sono dove le partite si allungano.

### Il caso peggiore

<!-- TODO: assets/icicle_max_entropy_worst.png -->
![Icicle di max_entropy, caso peggiore](assets/icicle_max_entropy_worst.png)

```bash
python -m scripts.visualize_tree max_entropy --kind sunburst --metric worst --label feedback
```

### Due strategie a confronto

Passando due alberi, il colore mostra la differenza di costo dello *stesso stato* nei due alberi: blu dove vince la prima, rosso dove vince la seconda, bianco dove pareggiano.

<!-- TODO: assets/delta_entropy_vs_evm.png -->
![Confronto max_entropy contro un EVM stimato](assets/delta_entropy_vs_evm.png)

```bash
python -m scripts.visualize_tree max_entropy estimated_evm_geometric
```

<!-- TODO: una frase su cosa si vede in questo confronto, dopo aver guardato la figura
le startegie hanno un tempo medio molto simile alla radice ma hanno performance diverse in base al feedback ricevuto
 -->
[correggi:
il percorso di lettura degli alberi: lo script cerca in data/trees_to_draws la pipeline scrive in data/runs/iter_XX/trees(perchè e separata )]

## Il laboratorio dei regressori

Per le strategie EVM stimate serve un oracolo: un modello che stimi quanto costa chiudere un insieme di soluzioni. Il laboratorio svolge i seguenti passi:

1. **Estrarre** gli stati dagli alberi, con il miglior costo noto per ciascuno (`data_miner`, `augment_states`).
2. **Raffinare** i costi: esatti per gli stati piccoli, mini-torneo tra strategie per quelli medi (`dataset_refiner`).
3. **Calcolare le feature** di ogni stato: `log_n`, entropia per colonna, deviazione standard delle cifre, ecc. (`feature_builder`).
4. **Selezionare** soglie su N e feature di ogni fase con programmazione dinamica e cross-validation (`select_model`), oppure addestrare modelli a soglie fisse (`train_model`).
5. **Valutare** con il regret rispetto all'ottimo esatto (`evaluate_regret`).
6. **Rigiocare**: i modelli diventano strategie, i loro alberi alimentano l'iterazione successiva.

```bash
python -m scripts.run_pipeline --iter 0 --dry-run     # cosa farebbe
python -m scripts.run_pipeline --iter 0               # esegue (richiede tempo)
```

I costi sopra N ≈ 10 vengono da strategie già esistenti, quindi l'oracolo impara in gran parte a imitarle: è un limite da tenere presente.

## Struttura

```
main.py            partita interattiva o automatica
src/
  core/            feedback, generatore, tabella dei feedback, stato
  strategies/      tutte le strategie e lo zoo che le costruisce per nome
  estimators/      stimatori del costo di un bucket
  features/        registro delle feature
  analysis/        costruzione, salvataggio e metriche degli alberi
  cli/             ciclo di gioco da terminale
scripts/           generazione alberi, confronto, visualizzazione, pipeline
experiments/       laboratorio dei regressori, router, tuning, diagnostica
tests/             suite pytest
data/              alberi, modelli e dataset generati (non versionata)
```

