"""Generazione delle proposte di giocata e verifica degli esiti.

Nota: le estrazioni sono eventi indipendenti — nessuna strategia aumenta la
probabilità di vincita. Le strategie servono a costruire giocate secondo
criteri statistici espliciti e ripetibili.
"""
import json
import random
from collections import Counter

from . import stats

STRATEGIE = {
    "ritardatari": "Numeri con il ritardo attuale più alto",
    "frequenti": "Numeri più frequenti nelle ultime estrazioni",
    "mista": "Metà ritardatari e metà frequenti",
    "bilanciata": "Estrazione pesata sulle frequenze, bilanciata pari/dispari e bassi/alti",
    "spia": "Numeri usciti più spesso dopo i numeri dell'ultima estrazione (solo Lotto)",
    "ml": "Punteggi del modello ML (vedi tab Analisi ML; parametri nella tab Setup)",
}

FINESTRA_FREQ = 100  # estrazioni recenti considerate per le frequenze


def genera(con, gioco, strategia, ruota=None, quanti=None):
    """Ritorna {'numeri': [...], 'spiegazione': str}."""
    if gioco == "superenalotto":
        quanti, ruota = 6, None
    quanti = max(1, min(int(quanti or 5), 10))

    if strategia == "ritardatari":
        rit = stats.ritardi(con, gioco, ruota)
        scelti = sorted(range(1, 91), key=lambda n: -rit[n]["attuale"])[:quanti]
        spieg = "Ritardi attuali: " + ", ".join(
            f"{n} ({rit[n]['attuale']})" for n in sorted(scelti))

    elif strategia == "frequenti":
        freq = stats.frequenze(con, gioco, ruota, FINESTRA_FREQ)
        scelti = sorted(range(1, 91), key=lambda n: -freq[n])[:quanti]
        spieg = (f"Uscite nelle ultime {FINESTRA_FREQ} estrazioni: "
                 + ", ".join(f"{n} ({freq[n]})" for n in sorted(scelti)))

    elif strategia == "mista":
        rit = stats.ritardi(con, gioco, ruota)
        freq = stats.frequenze(con, gioco, ruota, FINESTRA_FREQ)
        per_rit = sorted(range(1, 91), key=lambda n: -rit[n]["attuale"])
        per_freq = sorted(range(1, 91), key=lambda n: -freq[n])
        scelti, da_rit = [], (quanti + 1) // 2
        for n in per_rit:
            if len(scelti) < da_rit:
                scelti.append(n)
        for n in per_freq:
            if len(scelti) >= quanti:
                break
            if n not in scelti:
                scelti.append(n)
        spieg = (f"{da_rit} tra i più ritardatari + {quanti - da_rit} "
                 f"tra i più frequenti (ultime {FINESTRA_FREQ})")

    elif strategia == "bilanciata":
        freq = stats.frequenze(con, gioco, ruota, FINESTRA_FREQ)
        pesi = {n: freq[n] + 1 for n in range(1, 91)}
        scelti = _estrazione_bilanciata(pesi, quanti)
        pari = sum(1 for n in scelti if n % 2 == 0)
        bassi = sum(1 for n in scelti if n <= 45)
        spieg = (f"Scelta casuale pesata sulle frequenze recenti — "
                 f"{pari} pari / {quanti - pari} dispari, "
                 f"{bassi} bassi (1-45) / {quanti - bassi} alti (46-90)")

    elif strategia == "spia":
        if gioco != "lotto":
            raise ValueError("La strategia 'spia' vale solo per il Lotto")
        ultima = stats.estrazioni_lotto(con, ruota, 1)
        if not ultima:
            raise ValueError("Nessuna estrazione in archivio")
        conta = Counter()
        for n in ultima[0][1]:
            for seg, c in stats.numeri_spia(con, ruota, n)["seguenti"]:
                conta[seg] += c
        scelti = [n for n, _ in conta.most_common(quanti)]
        spieg = ("Numeri storicamente più frequenti nell'estrazione successiva "
                 f"ai numeri usciti il {ultima[0][0]}: "
                 + ", ".join(map(str, ultima[0][1])))

    elif strategia == "ml":
        from . import ml
        risultati = ml.risultati_salvati(con, gioco, ruota)
        if not risultati:
            risultati = ml.analizza_e_salva(con, gioco, ruota)
        scelti = [p["numero"] for p in risultati["punteggi"][:quanti]]
        spieg = (f"Punteggi {ml.NOMI_MODELLI[risultati['config']['modello']]} "
                 f"(analisi del {risultati['calcolata']}, AUC {risultati['auc']}): "
                 + ", ".join(f"{p['numero']} ({p['punteggio']})"
                             for p in risultati["punteggi"][:quanti]))

    else:
        raise ValueError(f"Strategia sconosciuta: {strategia}")

    return {"numeri": sorted(scelti), "spiegazione": spieg}


def _estrazione_bilanciata(pesi, quanti, tentativi=200):
    """Campiona `quanti` numeri pesati, cercando equilibrio pari/dispari e bassi/alti."""
    migliore, migliore_score = None, float("-inf")
    numeri = list(pesi)
    valori = [pesi[n] for n in numeri]
    soglia = 0 if quanti % 2 == 0 else -1
    for _ in range(tentativi):
        scelti = set()
        while len(scelti) < quanti:
            scelti.add(random.choices(numeri, weights=valori)[0])
        pari = sum(1 for n in scelti if n % 2 == 0)
        bassi = sum(1 for n in scelti if n <= 45)
        meta = quanti / 2
        score = -abs(pari - meta) - abs(bassi - meta)
        if score > migliore_score:
            migliore, migliore_score = scelti, score
        if migliore_score >= soglia:
            break
    return list(migliore)


# ------------------------------------------------------------------- verifica

SORTI = {1: "ambata", 2: "ambo", 3: "terno", 4: "quaterna", 5: "cinquina"}

ORA_CHIUSURA = "19:30"  # dopo quest'ora la giocata vale per l'estrazione successiva


def verifica_giocate(con):
    """Confronta ogni giocata aperta con la prima estrazione utile: quella del
    giorno stesso se creata entro le 19:30, altrimenti la successiva.
    Ritorna il numero di giocate verificate."""
    aggiornate = 0
    for g in con.execute("SELECT * FROM giocate WHERE esito IS NULL").fetchall():
        numeri = [int(n) for n in g["numeri"].split(",")]
        data_creazione = g["creata"][:10]
        ora_creazione = g["creata"][11:16]
        confronto = ">=" if ora_creazione < ORA_CHIUSURA else ">"

        if g["gioco"] == "lotto":
            righe = con.execute(
                "SELECT data, ruota, n1, n2, n3, n4, n5 FROM lotto "
                f"WHERE data {confronto} ? AND ruota = ? ORDER BY data LIMIT 1",
                (data_creazione, g["ruota"])).fetchall()
            if not righe:
                continue
            r = righe[0]
            estratti = [r["n1"], r["n2"], r["n3"], r["n4"], r["n5"]]
            indovinati = sorted(set(numeri) & set(estratti))
            esito = {
                "data": r["data"], "estratti": estratti, "indovinati": indovinati,
                "sorte": SORTI.get(len(indovinati), "nulla di fatto")
                if indovinati else "nessun numero",
            }
        else:
            r = con.execute(
                f"SELECT * FROM superenalotto WHERE data {confronto} ? "
                "ORDER BY data LIMIT 1",
                (data_creazione,)).fetchone()
            if not r:
                continue
            estratti = [r["n1"], r["n2"], r["n3"], r["n4"], r["n5"], r["n6"]]
            indovinati = sorted(set(numeri) & set(estratti))
            esito = {
                "data": r["data"], "concorso": r["concorso"], "estratti": estratti,
                "jolly": r["jolly"], "superstar": r["superstar"],
                "indovinati": indovinati,
                "sorte": f"{len(indovinati)} punti" if indovinati else "nessun numero",
            }

        con.execute("UPDATE giocate SET esito = ? WHERE id = ?",
                    (json.dumps(esito), g["id"]))
        aggiornate += 1
    con.commit()
    return aggiornate
