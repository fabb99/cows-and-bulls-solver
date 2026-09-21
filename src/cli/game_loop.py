"""
Modulo: game_loop
Gestisce l'interazione con l'utente (tramite terminale) e simula le partite.
"""
from typing import List
from src.core.engine import filter_solutions
from src.core.constants import DIM
from src.strategies.base import BaseStrategy

def fb_string_to_tuple(fb_string: str)->tuple[int, int]:
    return (fb_string.count("+"), fb_string.count("-"))

def fb_tuple_to_string(fb_tuple: tuple[int, int])-> str:
    return "+"*fb_tuple[0]+"-"*fb_tuple[1]

def play_interactive(strategy: BaseStrategy, possible_solutions: List[str]) -> None:
    """
    PRE: strategy è inizializzata. possible_solutions è l'insieme iniziale.
    POST: Avvia un loop interattivo. Stampa l'ipotesi della strategia, legge il feedback
          dell'utente e filtra le soluzioni finché non si inserisce "++++" (DIM, 0).
    """
    # 1. Inizializza history = []
    # 2. While True o un ciclo simile:
    # 3.   guess = strategy.choose_guess(...)
    # 4.   print("L'IA propone:", guess)
    # 5.   leggi il feedback con input() (nota: l'utente scriverà "++--", tu dovrai tradurlo in tupla!)
    # 6.   se tupla == (DIM, 0): print("Vittoria!") e break
    # 7.   possible_solutions = filter_solutions(...)
    # 8.   history.append(guess)
    history = []
    
    while True:
        guess = strategy.choose_guess(possible_solutions, history)
        history.append(guess)
        print(f"Ipotesi : {guess}")
        fb=fb_string_to_tuple(input("Feedback: "))
        if fb == (DIM,0):
            print(f"Risolto in {len(history)} tentativi")
            break
        possible_solutions=filter_solutions(guess,fb,possible_solutions,strategy.lookup)
        
 
def play_autoplay(strategy: BaseStrategy, secret: str, possible_solutions: List[str]) -> int:
    """
    PRE: secret è la soluzione da indovinare.
    POST: Simula la partita in automatico e stampa i passaggi. Restituisce il numero di turni usati.
    """
    # Usa strategy.lookup.get_feedback(guess, secret) per ottenere il feedback in automatico!
    history = []
    while True:
        guess = strategy.choose_guess(possible_solutions, history)
        history.append(guess)
        print("------------------")
        print(f"Ipotesi : {guess}")
        fb=strategy.lookup.get_feedback(guess,secret)
        print(f"Feedback: {fb_tuple_to_string(fb)}")
        print("------------------")
        if fb == (DIM,0):
            print(f"Risolto in {len(history)} tentativi")
            print("------------------")
            break
        possible_solutions=filter_solutions(guess,fb,possible_solutions,strategy.lookup)
        
 