"""Sistemi ridotti — covering designs.

Dati v numeri scelti, genera il minor numero possibile di colonne da k numeri
tali che OGNI combinazione di t numeri tra i v scelti sia contenuta in almeno
una colonna (copertura C(v, k, t)). Garanzia: se almeno t dei numeri vincenti
sono tra i tuoi v, almeno una colonna realizza la sorte di grado t.

La costruzione è greedy deterministica: a ogni passo sceglie la colonna che
copre il maggior numero di combinazioni ancora scoperte (a parità, la prima
in ordine lessicografico). Non sempre raggiunge l'ottimo assoluto dei design
teorici, ma si avvicina e la garanzia è verificata per costruzione.
"""
from itertools import combinations
from math import comb

SORTI = {1: "ambata", 2: "ambo", 3: "terno", 4: "quaterna",
         5: "cinquina", 6: "sestina"}

MAX_CANDIDATE = 60_000   # limite di C(v, k) per tempi di calcolo ragionevoli


def genera_sistema(numeri, colonna, garanzia):
    """Ritorna {'colonne': [...], 'integrale': n, 'risparmio': %, ...}."""
    numeri = sorted(set(int(n) for n in numeri))
    if any(n < 1 or n > 90 for n in numeri):
        raise ValueError("I numeri devono essere tra 1 e 90")
    v = len(numeri)
    colonna = int(colonna)
    garanzia = int(garanzia)
    if v < 3:
        raise ValueError("Servono almeno 3 numeri")
    if not 2 <= colonna <= min(10, v - 1):
        raise ValueError(f"La colonna deve avere tra 2 e {min(10, v - 1)} "
                         "numeri (e meno dei numeri del sistema)")
    if not 1 <= garanzia <= colonna:
        raise ValueError("La garanzia deve essere tra 1 e la dimensione "
                         "della colonna")
    if comb(v, colonna) > MAX_CANDIDATE:
        raise ValueError(
            f"Troppe colonne candidate (C({v},{colonna}) = "
            f"{comb(v, colonna):,}): riduci i numeri del sistema o "
            "la dimensione della colonna")

    bersagli = list(combinations(numeri, garanzia))
    indice = {s: i for i, s in enumerate(bersagli)}
    candidate = list(combinations(numeri, colonna))
    coperture = [frozenset(indice[s] for s in combinations(c, garanzia))
                 for c in candidate]

    da_coprire = set(range(len(bersagli)))
    colonne = []
    while da_coprire:
        migliore, guadagno = None, 0
        for i, copre in enumerate(coperture):
            g = len(copre & da_coprire)
            if g > guadagno:
                migliore, guadagno = i, g
        if migliore is None:    # non dovrebbe accadere
            raise ValueError("Copertura impossibile con questi parametri")
        colonne.append(list(candidate[migliore]))
        da_coprire -= coperture[migliore]

    integrale = comb(v, colonna)
    return {
        "numeri": numeri, "v": v, "colonna": colonna, "garanzia": garanzia,
        "sorte": SORTI.get(garanzia, f"{garanzia} punti"),
        "colonne": colonne,
        "combinazioni_coperte": len(bersagli),
        "integrale": integrale,
        "risparmio": round(100 * (1 - len(colonne) / integrale), 1),
    }
