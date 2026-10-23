"""
Modulo: lookup_table
Tabella dei feedback precalcolata in memoria, con costruzione lazy.
"""

from src.core.feedback import Feedback, calculate_feedback


class FeedbackLookup:
    """
    Tabella {guess: {secret: feedback}} sull'intero dominio.

    Costruita alla prima richiesta (lazy). Con 5040 permutazioni contiene ~25 milioni
    di voci: occupa oltre 1 GB di RAM e qualche decina di secondi di calcolo.
    """

    def __init__(self, domain: list[str]) -> None:
        """PRE: domain è la lista di tutte le combinazioni ammesse (senza duplicati)."""
        self._domain = domain
        self._table: dict[str, dict[str, Feedback]] | None = None

    def _build_table(self) -> None:
        """
        POST: self._table è popolata per tutte le coppie, sfruttando la simmetria
              feedback(a, b) == feedback(b, a).
        """
        table: dict[str, dict[str, Feedback]] = {x: {} for x in self._domain}
        n = len(self._domain)
        for i in range(n):
            a = self._domain[i]
            for j in range(i, n):
                b = self._domain[j]
                fb = calculate_feedback(a, b)
                table[a][b] = fb
                table[b][a] = fb
        self._table = table

    def get_feedback(self, guess: str, secret: str) -> Feedback:
        """
        PRE:  guess e secret appartengono al dominio.
        POST: feedback (bulls, cows) in O(1) dopo la prima chiamata.
        """
        if self._table is None:
            self._build_table()
        return self._table[guess][secret]