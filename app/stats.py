"""Analisi statistiche sullo storico delle estrazioni."""
from collections import Counter
from itertools import combinations


# ---------------------------------------------------------------- caricamento

def estrazioni_lotto(con, ruota, finestra=None):
    """Estrazioni di una ruota, dalla più recente: [(data, [n1..n5]), ...]."""
    q = "SELECT data, n1, n2, n3, n4, n5 FROM lotto WHERE ruota = ? ORDER BY data DESC"
    par = [ruota]
    if finestra:
        q += " LIMIT ?"
        par.append(finestra)
    return [(r["data"], [r["n1"], r["n2"], r["n3"], r["n4"], r["n5"]])
            for r in con.execute(q, par)]


def estrazioni_superenalotto(con, finestra=None):
    q = ("SELECT data, concorso, n1, n2, n3, n4, n5, n6, jolly, superstar "
         "FROM superenalotto ORDER BY data DESC")
    par = []
    if finestra:
        q += " LIMIT ?"
        par.append(finestra)
    return [(r["data"], [r["n1"], r["n2"], r["n3"], r["n4"], r["n5"], r["n6"]])
            for r in con.execute(q, par)]


def _estrazioni(con, gioco, ruota=None, finestra=None):
    if gioco == "lotto":
        return estrazioni_lotto(con, ruota, finestra)
    return estrazioni_superenalotto(con, finestra)


# ---------------------------------------------------------------- statistiche

def frequenze(con, gioco, ruota=None, finestra=None):
    """Conteggio uscite per ogni numero 1-90 (dalla più recente, su `finestra` estrazioni)."""
    conta = Counter()
    for _, nums in _estrazioni(con, gioco, ruota, finestra):
        conta.update(nums)
    return {n: conta.get(n, 0) for n in range(1, 91)}


def ritardi(con, gioco, ruota=None):
    """Per ogni numero: ritardo attuale e ritardo massimo storico (in estrazioni)."""
    estr = _estrazioni(con, gioco, ruota)  # già in ordine decrescente
    attuale, massimo, visto = {}, {}, set()

    # ritardo attuale: estrazioni trascorse dall'ultima uscita
    for i, (_, nums) in enumerate(estr):
        for n in nums:
            if n not in visto:
                attuale[n] = i
                visto.add(n)
        if len(visto) == 90:
            break
    for n in range(1, 91):
        attuale.setdefault(n, len(estr))

    # ritardo massimo storico: gap più lungo tra due uscite consecutive
    ultima_pos = {}
    for i, (_, nums) in enumerate(reversed(estr)):  # in ordine cronologico
        for n in nums:
            gap = i - ultima_pos.get(n, -1) - 1
            if gap > massimo.get(n, 0):
                massimo[n] = gap
            ultima_pos[n] = i
    for n in range(1, 91):
        massimo[n] = max(massimo.get(n, 0), attuale[n])

    return {n: {"attuale": attuale[n], "massimo": massimo[n]} for n in range(1, 91)}


def combinazioni_frequenti(con, gioco, ruota=None, finestra=None, k=2, top=20):
    """Coppie (k=2) o terzine (k=3) uscite più spesso insieme."""
    conta = Counter()
    for _, nums in _estrazioni(con, gioco, ruota, finestra):
        conta.update(combinations(sorted(nums), k))
    return conta.most_common(top)


def distribuzione(con, gioco, ruota=None, finestra=None):
    """Pattern d'insieme: pari/dispari, bassi/alti, somma, decine."""
    estr = _estrazioni(con, gioco, ruota, finestra)
    tot = pari = bassi = 0
    somme = []
    decine = Counter()
    for _, nums in estr:
        somme.append(sum(nums))
        for n in nums:
            tot += 1
            pari += n % 2 == 0
            bassi += n <= 45
            decine[(n - 1) // 10] += 1
    if not tot:
        return None
    return {
        "estrazioni": len(estr),
        "pct_pari": round(100 * pari / tot, 1),
        "pct_bassi": round(100 * bassi / tot, 1),
        "somma_media": round(sum(somme) / len(somme), 1),
        "somma_min": min(somme),
        "somma_max": max(somme),
        "decine": [decine.get(d, 0) for d in range(9)],  # 1-10, 11-20, ... 81-90
    }


def numeri_spia(con, ruota, numero, top=10):
    """Lotto: numeri usciti più spesso sulla ruota nell'estrazione successiva
    a un'uscita di `numero` (analisi dei 'numeri spia')."""
    estr = list(reversed(estrazioni_lotto(con, ruota)))  # cronologico
    conta = Counter()
    occorrenze = 0
    for prec, succ in zip(estr, estr[1:]):
        if numero in prec[1]:
            occorrenze += 1
            conta.update(succ[1])
    return {"occorrenze": occorrenze, "seguenti": conta.most_common(top)}


def riepilogo(con):
    """Dati di copertura archivio per la dashboard."""
    out = {}
    r = con.execute("SELECT MIN(data) a, MAX(data) b, COUNT(*) c FROM lotto").fetchone()
    out["lotto"] = {"dal": r["a"], "al": r["b"], "righe": r["c"]}
    r = con.execute("SELECT MIN(data) a, MAX(data) b, COUNT(*) c FROM superenalotto").fetchone()
    out["superenalotto"] = {"dal": r["a"], "al": r["b"], "righe": r["c"]}
    return out


def ultima_estrazione_lotto(con):
    """Tutte le ruote dell'ultima data disponibile."""
    data = con.execute("SELECT MAX(data) d FROM lotto").fetchone()["d"]
    if not data:
        return None, []
    righe = con.execute(
        "SELECT ruota, n1, n2, n3, n4, n5 FROM lotto WHERE data = ? ORDER BY ruota",
        (data,)).fetchall()
    return data, righe


def ultima_estrazione_superenalotto(con):
    return con.execute(
        "SELECT * FROM superenalotto ORDER BY data DESC LIMIT 1").fetchone()
