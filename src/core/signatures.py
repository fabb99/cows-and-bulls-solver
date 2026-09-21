"""
Modulo: signatures
Gestisce le classi di equivalenza (Isomorfismi) per ottimizzare la ricerca delle IA.
Raggruppa le candidate in "Firme Strutturali" in base al loro rapporto con lo storico delle mosse.
"""

def calculate_signature(candidate: str, history: list[str]) -> tuple[tuple[int, ...], ...]:
    """
    Calcola il "curriculum" (la firma) di una singola candidata rispetto allo storico.
    
    PRE: candidate è una stringa (es. "0123"). history è la lista delle mosse giocate.
    POST: Ritorna una tupla ordinata di tuple. Ogni sotto-tupla rappresenta 
          la "storia" di una singola cifra della candidata (0=Nuova, 1=Miss, 2=Home).
    """
    signature = []
    for i, c in enumerate(candidate):
        num_history = []
        for move in history:
            if c in move:
                if c == move[i]:
                    num_history.append(2) # miss
                else :
                    num_history.append(1) # home
            else: 
                num_history.append(0)     # Nuova

        signature.append(tuple(num_history))
    return tuple(sorted(signature))

def group_candidates_by_signature(candidates: list[str], history: list[str]) -> dict[tuple, list[str]]:
    """
    Raggruppa tutti i candidati in base alla loro firma strutturale.
    
    PRE: candidates lista di permutazioni da testare. history lista di mosse giocate.
    POST: Ritorna un dizionario dove la CHIAVE è la Firma (Tupla) e il VALORE è 
          la lista dei candidati che possiedono quella firma.
    """
    signatures = {}
    for candidate in candidates:
        signature = calculate_signature(candidate, history)
        if not signature in signatures:
            signatures[signature]=[]
        signatures[signature].append(candidate)
    return signatures

def get_representatives(candidates: list[str], history: list[str]) -> list[str]:
    """
    Estrae UN SOLO rappresentante per ogni classe di equivalenza (Firma).
    
    PRE: candidates e history validi.
    POST: Ritorna una lista ridotta di candidati da valutare (uno per ogni firma).
          Se history è vuota (Turno 1), ritorna un solo candidato (es. "0123").
    """
    representatives = []
    for _, c in group_candidates_by_signature(candidates,history).items():
        representatives.append(c[0])
    return representatives

    