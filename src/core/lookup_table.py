from src.core.feedback import calculate_feedback


class FeedbackLookup:
    """
    Gestisce la tabella di lookup dei feedback in memoria.
    Inizializzazione Lazy (viene calcolata solo la prima volta che serve).
    """
    
    def __init__(self, domain: list[str]):
        """ PRE: domain è la lista di tutte le permutazioni possibili. """
        self._domain = domain
        self._table: dict[str, dict[str, tuple[int, int]]] = None
                   
    def _build_table(self) -> None:
        """
        PRE: Nessuna. Metodo privato (segnalato dall'underscore).
        POST: Popola self._table con tutte le combinazioni incrociate 
              sfruttando la simmetria (guess vs secret == secret vs guess).
        """
        self._table = {}

        for x in self._domain:
            self._table[x] = {}

        len_domain = len(self._domain)
        
        for i in range(0,len_domain):
            for j in range(i,len_domain):
                fb = calculate_feedback(self._domain[i], self._domain[j])        
                self._table[self._domain[i]][self._domain[j]] = fb
                self._table[self._domain[j]][self._domain[i]] = fb

    def get_feedback(self, guess: str, secret: str) -> tuple[int, int]:
        """ POST: Ritorna il feedback usando la matrice in memoria O(1). """
        if self._table is None:
            self._build_table()

        return self._table[guess][secret]