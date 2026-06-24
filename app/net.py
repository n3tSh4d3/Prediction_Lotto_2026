"""Analisi NET — framework modulare di algoritmi attivabili su richiesta.

Ogni algoritmo lavora su una base dati selezionata dall'utente: tutte le
estrazioni di una ruota del Lotto, oppure la lista unica del SuperEnalotto.

Per aggiungere un algoritmo basta una funzione registrata con il decoratore:

    @algoritmo("mia-chiave", "Titolo mostrato", "Descrizione del calcolo",
               parametri={"finestra": 50,          # numero  -> campo di testo
                          "opzione": True,         # bool    -> interruttore
                          "modo": ["a", "b"]})     # lista   -> menu a tendina
    def mia_analisi(estrazioni, par):
        # estrazioni: [(data, [numeri]), ...] in ordine CRONOLOGICO,
        #             tutta la base dati della ruota (o del SuperEnalotto)
        # par:        parametri (default sopra, modificabili dall'utente)
        # ritorna:    {"numeri": [probabili estratti], "dettagli": "spiegazione"}
        ...

L'algoritmo compare automaticamente nella tab Analisi NET con i suoi parametri.
"""
import inspect
from collections import Counter

import numpy as np

from . import db, stats

ALGORITMI = {}


def algoritmo(chiave, titolo, descrizione, parametri=None, in_meta=True):
    """Registra un algoritmo. Se la funzione accetta un terzo parametro,
    riceve un contesto {'con', 'gioco', 'ruota'} con accesso al database.
    in_meta=False esclude l'algoritmo dal backtest del meta-algoritmo."""
    def registra(fn):
        ALGORITMI[chiave] = {
            "chiave": chiave, "titolo": titolo, "descrizione": descrizione,
            "parametri": parametri or {}, "fn": fn, "in_meta": in_meta,
            "con_contesto": len(inspect.signature(fn).parameters) >= 3,
        }
        return fn
    return registra


def parametri_default(alg):
    """Valori di default dei parametri (per le liste: la prima opzione)."""
    return {k: (v[0] if isinstance(v, list) else v)
            for k, v in alg["parametri"].items()}


def base_dati(con, gioco, ruota=None):
    """Tutte le estrazioni della base scelta, in ordine cronologico."""
    return list(reversed(stats._estrazioni(con, gioco, ruota)))


def esegui(con, gioco, ruota, attivi, parametri_grezzi):
    """Esegue gli algoritmi attivi sulla base dati scelta.

    parametri_grezzi: dict tipo {"chiave__nomeparam": "valore"} (dalla query).
    Ritorna (risultati, convergenza)."""
    estrazioni = base_dati(con, gioco, ruota if gioco == "lotto" else None)
    risultati = []
    for chiave in attivi:
        alg = ALGORITMI.get(chiave)
        if not alg:
            continue
        par = {}
        for nome, default in alg["parametri"].items():
            grezzo = parametri_grezzi.get(f"{chiave}__{nome}", default)
            if isinstance(default, bool):
                par[nome] = grezzo in (True, "1", "true", "on")
                continue
            if isinstance(default, list):  # scelta fissa: menu a tendina
                par[nome] = grezzo if grezzo in default else default[0]
                continue
            try:
                par[nome] = type(default)(grezzo)
            except (TypeError, ValueError):
                par[nome] = default
        try:
            if alg["con_contesto"]:
                esito = alg["fn"](estrazioni, par,
                                  {"con": con, "gioco": gioco, "ruota": ruota})
            else:
                esito = alg["fn"](estrazioni, par)
            errore = None
        except Exception as exc:
            esito, errore = {"numeri": [], "dettagli": ""}, str(exc)
        risultati.append({
            "chiave": chiave, "titolo": alg["titolo"], "parametri": par,
            "numeri": sorted(esito.get("numeri", [])),
            "dettagli": esito.get("dettagli", ""), "errore": errore,
        })

    # convergenza: numeri proposti da più algoritmi
    conta = Counter()
    for r in risultati:
        if not r["errore"]:
            conta.update(set(r["numeri"]))
    convergenza = [(n, c) for n, c in conta.most_common() if c >= 2]
    return risultati, convergenza


# --------------------------------------------------------------------------
# Algoritmi definiti dall'operatore.

@algoritmo(
    "distanze", "Distanze tra i numeri estratti",
    "Per ogni estrazione della finestra calcola le distanze tra i numeri "
    "consecutivi (ordinati); le distanze più ricorrenti vengono applicate ai "
    "numeri dell'ultima estrazione per ottenere i probabili estratti. "
    "Con 'chiusura_circolare' include la distanza dall'ultimo al primo numero "
    "sul cerchio 1-90 (e i candidati oltre il 90 rientrano dal basso). "
    "Con 'gap_stesso_numero' cambia il calcolo: misura le distanze in "
    "estrazioni tra le uscite successive di ciascun numero e propone i numeri "
    "il cui ritardo attuale coincide con un proprio intervallo storico di "
    "riuscita ricorrente. Con 'distanza_fissa' > 0 (es. 15, 30, 45) non usa "
    "le distanze più frequenti ma applica quella distanza in entrambe le "
    "direzioni ai numeri dell'ultima estrazione.",
    parametri={"finestra": 300, "top_distanze": 5, "quanti": 5,
               "distanza_fissa": 0,
               "chiusura_circolare": True, "gap_stesso_numero": False},
)
def _distanze(estrazioni, par):
    if not estrazioni:
        raise ValueError("Base dati vuota")
    finestra = int(par["finestra"]) or len(estrazioni)
    recenti = estrazioni[-finestra:]
    quanti = max(1, min(int(par["quanti"]), 10))

    if par["gap_stesso_numero"]:
        return _distanze_gap_stesso_numero(recenti, quanti)

    conta = Counter()
    for _, nums in recenti:
        ordinati = sorted(nums)
        for a, b in zip(ordinati, ordinati[1:]):
            conta[b - a] += 1
        if par["chiusura_circolare"]:
            conta[ordinati[0] + 90 - ordinati[-1]] += 1  # chiusura del cerchio

    fissa = max(0, min(int(par["distanza_fissa"]), 45))
    if fissa:
        top = [(fissa, conta.get(fissa, 0)), (-fissa, conta.get(fissa, 0))]
    else:
        top = conta.most_common(max(1, min(int(par["top_distanze"]), 20)))

    data_ultima, ultimi = estrazioni[-1]
    punteggio, origine = Counter(), {}
    for distanza, freq in top:
        for n in ultimi:
            if par["chiusura_circolare"]:
                candidato = (n + distanza - 1) % 90 + 1
            else:
                candidato = n + distanza
                if not 1 <= candidato <= 90:
                    continue
            punteggio[candidato] += freq or 1
            origine.setdefault(candidato, f"{n}{distanza:+d}")

    scelti = [n for n, _ in punteggio.most_common(quanti)]
    dettagli = (
        f"Distanze più frequenti su {len(recenti)} estrazioni"
        f"{' (con chiusura circolare)' if par['chiusura_circolare'] else ''}: "
        + ", ".join(f"{d} (×{c})" for d, c in top)
        + f". Applicate all'ultima estrazione del {data_ultima} "
        + f"({', '.join(map(str, sorted(ultimi)))}) — generazione: "
        + ", ".join(f"{n} = {origine[n]}" for n in sorted(scelti)))
    return {"numeri": scelti, "dettagli": dettagli}


def _distanze_gap_stesso_numero(recenti, quanti):
    """Variante: distanze (in estrazioni) tra uscite successive dello stesso
    numero. Candidati i numeri il cui ritardo attuale, alla prossima
    estrazione, completerebbe un gap già ricorrente nella loro storia."""
    tot = len(recenti)
    punteggio, info = {}, {}
    for n in range(1, 91):
        posizioni = [i for i, (_, nums) in enumerate(recenti) if n in nums]
        if len(posizioni) < 2:
            continue
        gap_storici = Counter(b - a for a, b in zip(posizioni, posizioni[1:]))
        gap_prossimo = tot - posizioni[-1]  # gap se uscisse alla prossima
        ricorrenze = gap_storici.get(gap_prossimo, 0)
        if ricorrenze:
            punteggio[n] = ricorrenze
            info[n] = gap_prossimo
    scelti = sorted(punteggio, key=lambda n: (-punteggio[n], n))[:quanti]
    dettagli = (
        f"Gap dello stesso numero su {tot} estrazioni — alla prossima "
        "estrazione questi numeri completerebbero un proprio intervallo "
        "storico ricorrente: "
        + ", ".join(f"{n} (gap {info[n]}, già avvenuto ×{punteggio[n]})"
                    for n in sorted(scelti)))
    return {"numeri": scelti, "dettagli": dettagli}


@algoritmo(
    "media-differenziali", "Media e differenziali per posizione (ML)",
    "Per ogni estrazione calcola la media dei numeri estratti (parametro "
    "'media': aritmetica, ponderata o quadratica) e i differenziali di ogni "
    "numero, ordinato per posizione (1º, 2º, ...), rispetto alla media della "
    "propria estrazione. Un regressore ML prevede la media della prossima "
    "estrazione dall'andamento delle medie precedenti; i differenziali per "
    "posizione si prevedono con ML sulle rispettive serie storiche "
    "('ml_differenziali' attivo) oppure in modo deterministico (media mobile "
    "dei differenziali recenti). I probabili sono: media prevista + "
    "differenziale previsto di ogni posizione. Con 'pesa_recenti' le "
    "estrazioni recenti contano di più di quelle vecchie (peso esponenziale, "
    "dimezzato ogni 'mezza_vita' estrazioni) sia nell'addestramento ML sia "
    "nella media mobile deterministica. Con 'condiziona_somma' i "
    "differenziali sono CONDIZIONATI alla somma prevista: si usano quelli "
    "delle 'vicini_somma' estrazioni storiche con media più simile a quella "
    "prevista, perché somma e forma sono correlate (somma alta = cinquina "
    "compressa verso il 90, e viceversa) — la ricostruzione resta così "
    "sempre coerente con la propria somma.",
    parametri={"finestra": 1000, "lag": 10,
               "media": ["aritmetica", "ponderata", "quadratica"],
               "ml_differenziali": True,
               "pesa_recenti": False, "mezza_vita": 200,
               "condiziona_somma": False, "vicini_somma": 100},
)
def _media_differenziali(estrazioni, par):
    from sklearn.ensemble import HistGradientBoostingRegressor

    if not estrazioni:
        raise ValueError("Base dati vuota")
    finestra = int(par["finestra"]) or len(estrazioni)
    recenti = estrazioni[-finestra:]
    lag = max(2, min(int(par["lag"]), 30))
    if len(recenti) < lag + 50:
        raise ValueError(f"Servono almeno {lag + 50} estrazioni "
                         f"(finestra attuale: {len(recenti)})")

    k = len(recenti[-1][1])  # 5 numeri (lotto) o 6 (superenalotto)
    X = np.array([sorted(nums) for _, nums in recenti], dtype=float)

    tipo = str(par["media"]).strip().lower()
    if tipo.startswith("pond"):
        tipo = "ponderata"  # pesi crescenti con la posizione (1, 2, ..., k)
        pesi = np.arange(1, k + 1, dtype=float)
        medie = (X * pesi).sum(axis=1) / pesi.sum()
    elif tipo.startswith("quad"):
        tipo = "quadratica"
        medie = np.sqrt((X ** 2).mean(axis=1))
    else:
        tipo = "aritmetica"
        medie = X.mean(axis=1)

    diff = X - medie[:, None]  # differenziale di ogni posizione dalla media

    mezza_vita = max(1, min(int(par["mezza_vita"]), 5000))

    def pesi_eta(quanti_campioni):
        """Peso esponenziale: 1 per il campione più recente, dimezzato ogni
        mezza_vita campioni andando indietro."""
        eta = np.arange(quanti_campioni)[::-1]
        return 0.5 ** (eta / mezza_vita)

    def prevedi_serie(serie):
        """Prevede il prossimo valore della serie con regressione sui lag."""
        camp = np.array([serie[t - lag:t] for t in range(lag, len(serie))])
        target = serie[lag:]
        modello = HistGradientBoostingRegressor(random_state=0)
        modello.fit(camp, target,
                    sample_weight=pesi_eta(len(target))
                    if par["pesa_recenti"] else None)
        return float(modello.predict(serie[-lag:].reshape(1, -1))[0])

    media_prevista = prevedi_serie(medie)

    correlazioni_somma = None
    if par["condiziona_somma"]:
        # l'idea dell'operatore: somma e differenziali sono correlati dentro
        # la stessa estrazione, quindi i differenziali vanno presi dalle
        # estrazioni storiche con somma simile a quella prevista
        vicini = max(10, min(int(par["vicini_somma"]), len(medie)))
        ordine_vicini = np.argsort(np.abs(medie - media_prevista))[:vicini]
        if par["pesa_recenti"]:
            eta = (len(medie) - 1) - ordine_vicini
            pesi_vicini = 0.5 ** (eta / mezza_vita)
        else:
            pesi_vicini = np.ones(len(ordine_vicini))
        diff_previsti = [float(np.average(diff[ordine_vicini, i],
                                          weights=pesi_vicini))
                         for i in range(k)]
        correlazioni_somma = [
            round(float(np.corrcoef(medie, diff[:, i])[0, 1]), 2)
            for i in range(k)]
        modo = (f"condizionati alla somma: {vicini} estrazioni con media "
                f"{tipo} più vicina a {media_prevista:.1f}")
    elif par["ml_differenziali"]:
        diff_previsti = [prevedi_serie(diff[:, i]) for i in range(k)]
        modo = f"ML per posizione (lag {lag})"
    else:
        if par["pesa_recenti"]:
            pesi_mm = pesi_eta(lag)
            diff_previsti = [float(np.average(diff[-lag:, i], weights=pesi_mm))
                             for i in range(k)]
        else:
            diff_previsti = [float(diff[-lag:, i].mean()) for i in range(k)]
        modo = f"media mobile degli ultimi {lag} differenziali"
    if par["pesa_recenti"]:
        modo += f", estrazioni recenti pesate (mezza vita {mezza_vita})"

    numeri, usati = [], set()
    for valore in (media_prevista + d for d in diff_previsti):
        n = int(round(max(1, min(90, valore))))
        while n in usati and n < 90:
            n += 1
        while n in usati and n > 1:
            n -= 1
        usati.add(n)
        numeri.append(n)

    dettagli = (
        f"Media {tipo} dell'ultima estrazione: {medie[-1]:.2f}; media prevista "
        f"per la prossima (ML su {len(medie) - lag} campioni, lag {lag}): "
        f"{media_prevista:.2f}. Differenziali previsti per posizione [{modo}]: "
        + ", ".join(f"{i + 1}º {d:+.1f}" for i, d in enumerate(diff_previsti))
        + " → ricostruzione: "
        + ", ".join(f"{media_prevista:.1f}{d:+.1f}≈{n}"
                    for d, n in zip(diff_previsti, numeri)))
    if correlazioni_somma:
        dettagli += (
            " Correlazione misurata media↔differenziale per posizione: "
            + ", ".join(f"{i + 1}º {c:+.2f}"
                        for i, c in enumerate(correlazioni_somma)))
    return {"numeri": numeri, "dettagli": dettagli}


@algoritmo(
    "decine-cadenze", "Frequenze per decine e cadenze",
    "Misura, sulla finestra scelta, frequenza e ritardo di ogni decina "
    "(1-10, 11-20, ... 81-90) e di ogni cadenza (numeri con la stessa cifra "
    "finale: cadenza 0 = 10, 20...90). Seleziona le migliori decine e cadenze "
    "secondo il 'criterio': frequenza (gruppi più usciti), ritardo (gruppi "
    "assenti da più estrazioni) o sotto_media (gruppi usciti meno dell'atteso "
    "teorico). I probabili sono i numeri all'incrocio tra le decine e le "
    "cadenze selezionate, ordinati per punteggio (incrocio pieno prima) e, "
    "a parità, per il criterio a livello di singolo numero.",
    parametri={"finestra": 300, "top_decine": 3, "top_cadenze": 3,
               "criterio": ["frequenza", "ritardo", "sotto_media"],
               "quanti": 5},
)
def _decine_cadenze(estrazioni, par):
    if not estrazioni:
        raise ValueError("Base dati vuota")
    finestra = int(par["finestra"]) or len(estrazioni)
    recenti = estrazioni[-finestra:]
    tot = len(recenti)
    quanti = max(1, min(int(par["quanti"]), 10))
    top_d = max(1, min(int(par["top_decine"]), 9))
    top_c = max(1, min(int(par["top_cadenze"]), 10))

    decina = lambda n: (n - 1) // 10          # 0..8
    cadenza = lambda n: n % 10                # 0..9
    nome_decina = lambda d: f"{d * 10 + 1}-{d * 10 + 10}"

    freq_dec, freq_cad, freq_num = Counter(), Counter(), Counter()
    ult_dec, ult_cad, ult_num = {}, {}, {}
    estratti_tot = 0
    for i, (_, nums) in enumerate(recenti):
        estratti_tot += len(nums)
        for n in nums:
            freq_dec[decina(n)] += 1
            freq_cad[cadenza(n)] += 1
            freq_num[n] += 1
            ult_dec[decina(n)] = i
            ult_cad[cadenza(n)] = i
            ult_num[n] = i

    rit_dec = {d: tot - 1 - ult_dec.get(d, -1) for d in range(9)}
    rit_cad = {c: tot - 1 - ult_cad.get(c, -1) for c in range(10)}
    rit_num = {n: tot - 1 - ult_num.get(n, -1) for n in range(1, 91)}
    attesa_dec = estratti_tot / 9.0   # ogni decina ha 10 numeri su 90
    attesa_cad = estratti_tot / 10.0  # ogni cadenza ha 9 numeri su 90

    criterio = par["criterio"]
    if criterio == "ritardo":
        chiave_d = lambda d: (-rit_dec[d], -attesa_dec + freq_dec[d])
        chiave_c = lambda c: (-rit_cad[c], -attesa_cad + freq_cad[c])
        chiave_n = lambda n: -rit_num[n]
    elif criterio == "sotto_media":
        chiave_d = lambda d: freq_dec[d] - attesa_dec
        chiave_c = lambda c: freq_cad[c] - attesa_cad
        chiave_n = lambda n: -rit_num[n]
    else:  # frequenza
        chiave_d = lambda d: -freq_dec[d]
        chiave_c = lambda c: -freq_cad[c]
        chiave_n = lambda n: -freq_num[n]

    decine_top = sorted(range(9), key=chiave_d)[:top_d]
    cadenze_top = sorted(range(10), key=chiave_c)[:top_c]

    candidati = []
    for n in range(1, 91):
        punteggio = (decina(n) in decine_top) + (cadenza(n) in cadenze_top)
        if punteggio:
            candidati.append((n, punteggio))
    candidati.sort(key=lambda nc: (-nc[1], chiave_n(nc[0]), nc[0]))
    scelti = [n for n, _ in candidati[:quanti]]
    incroci = sum(1 for n, p in candidati[:quanti] if p == 2)

    dettagli = (
        f"Criterio '{criterio}' su {tot} estrazioni. Decine scelte: "
        + ", ".join(f"{nome_decina(d)} (×{freq_dec[d]}, atteso {attesa_dec:.0f}, "
                    f"rit. {rit_dec[d]})" for d in decine_top)
        + ". Cadenze scelte: "
        + ", ".join(f"{c} (×{freq_cad[c]}, atteso {attesa_cad:.0f}, "
                    f"rit. {rit_cad[c]})" for c in cadenze_top)
        + f". {incroci} dei {len(scelti)} probabili sono all'incrocio "
        "decina+cadenza.")
    return {"numeri": scelti, "dettagli": dettagli}


@algoritmo(
    "impopolare", "Giocata impopolare (massimizza il premio)",
    "Non cambia la probabilità di vincere ma quanto si vince: nel "
    "SuperEnalotto il montepremi si divide tra i vincitori, quindi giocare "
    "combinazioni che pochi scelgono aumenta il premio atteso a parità di "
    "probabilità. Propone numeri statisticamente poco giocati (niente date "
    "1-31, niente numeri 'famosi' della smorfia, niente cifre tonde) evitando "
    "consecutivi e concentrazioni di decina. Nota: vale per i premi a "
    "ripartizione (jackpot SuperEnalotto); il Lotto ha premi a quota fissa.",
    parametri={"quanti": 6, "max_per_decina": 2, "evita_consecutivi": True},
    in_meta=False,
)
def _impopolare(estrazioni, par):
    famosi = {7: 2, 13: 3, 17: 2, 21: 1, 27: 1, 33: 2, 48: 2,
              55: 1, 66: 1, 77: 2, 88: 1, 90: 3}  # smorfia e "fortunati"
    popolarita = {}
    for n in range(1, 91):
        p = 0.0
        if n <= 31:
            p += 3              # date di nascita
        if n <= 12:
            p += 2              # giorno e mese
        if n % 10 == 0:
            p += 1              # cifre tonde
        p += famosi.get(n, 0)
        popolarita[n] = p

    quanti = max(1, min(int(par["quanti"]), 10))
    max_dec = max(1, min(int(par["max_per_decina"]), 10))
    scelti = []
    for n in sorted(popolarita, key=lambda x: (popolarita[x], -x)):
        if len(scelti) == quanti:
            break
        if par["evita_consecutivi"] and any(abs(n - s) == 1 for s in scelti):
            continue
        if sum(1 for s in scelti if (s - 1) // 10 == (n - 1) // 10) >= max_dec:
            continue
        scelti.append(n)

    bassi = sum(1 for n in scelti if n <= 31)
    dettagli = (
        f"Combinazione a bassa popolarità: {bassi} numeri ≤31 (i più giocati "
        "perché legati alle date), nessun numero simbolico della tradizione, "
        "niente consecutivi né concentrazioni. Stessa probabilità di qualsiasi "
        "altra combinazione, ma in caso di vincita a ripartizione il premio si "
        "divide con meno giocatori.")
    return {"numeri": scelti, "dettagli": dettagli}


@algoritmo(
    "simpatie-ruote", "Simpatie tra ruote (solo Lotto)",
    "Cerca le ruote 'amiche' della ruota selezionata: quelle i cui numeri "
    "ricompaiono sulla ruota in esame nell'estrazione successiva più spesso "
    "dell'atteso teorico (lift > 1). Propone i numeri usciti sulle ruote più "
    "simpatiche nell'ultima estrazione, pesati per il lift della ruota di "
    "provenienza e per il track record del singolo numero.",
    parametri={"finestra": 1000, "top_ruote": 3, "quanti": 5},
    in_meta=False,
)
def _simpatie_ruote(estrazioni, par, contesto):
    if contesto["gioco"] != "lotto":
        raise ValueError("Analisi disponibile solo per il Lotto: "
                         "serve il confronto tra ruote")
    con, ruota = contesto["con"], contesto["ruota"]
    finestra = max(50, min(int(par["finestra"]) or 100000, 100000))
    top_r = max(1, min(int(par["top_ruote"]), 10))
    quanti = max(1, min(int(par["quanti"]), 10))

    date = [r["data"] for r in con.execute(
        "SELECT DISTINCT data FROM lotto ORDER BY data DESC LIMIT ?",
        (finestra,))][::-1]
    per_data = {}
    for r in con.execute(
            "SELECT data, ruota, n1, n2, n3, n4, n5 FROM lotto WHERE data >= ?",
            (date[0],)):
        per_data.setdefault(r["data"], {})[r["ruota"]] = {
            r["n1"], r["n2"], r["n3"], r["n4"], r["n5"]}

    lift, seguiti = {}, {}
    for w in db.RUOTE:
        if w == ruota:
            continue
        osservati, attesi = 0, 0.0
        segue = Counter()
        for d1, d2 in zip(date, date[1:]):
            su_w = per_data.get(d1, {}).get(w)
            su_r = per_data.get(d2, {}).get(ruota)
            if not su_w or not su_r:
                continue
            comuni = su_w & su_r
            osservati += len(comuni)
            attesi += len(su_w) * len(su_r) / 90.0
            for n in comuni:
                segue[n] += 1
        if attesi:
            lift[w] = osservati / attesi
            seguiti[w] = segue

    amiche = sorted(lift, key=lambda w: -lift[w])[:top_r]
    ultima = date[-1]
    punteggio, fonte = Counter(), {}
    for w in amiche:
        for n in per_data.get(ultima, {}).get(w, ()):
            punteggio[n] += lift[w] + 0.01 * seguiti[w][n]
            fonte.setdefault(n, w)
    scelti = [n for n, _ in punteggio.most_common(quanti)]

    dettagli = (
        f"Su {len(date)} estrazioni, ruote più simpatiche a "
        f"{db.NOMI_RUOTE[ruota]} (lift = ripetizioni osservate/attese alla "
        "estrazione successiva): "
        + ", ".join(f"{db.NOMI_RUOTE[w]} ({lift[w]:.2f})" for w in amiche)
        + f". Probabili dai numeri usciti il {ultima} sulle ruote amiche: "
        + ", ".join(f"{n} (da {db.NOMI_RUOTE[fonte[n]]}, già seguito "
                    f"×{seguiti[fonte[n]][n]})" for n in sorted(scelti)))
    return {"numeri": scelti, "dettagli": dettagli}


@algoritmo(
    "trasformazioni", "Vertibili e complementi",
    "Trasformazioni della tradizione applicate all'ultima estrazione: "
    "vertibile (cifre scambiate: 12-21, 9-90), complemento a 90 (17-73) e "
    "figura (numeri con lo stesso resto mod 9). Per ogni trasformazione "
    "misura sullo storico quanto spesso l'estrazione successiva contiene "
    "trasformati dei numeri appena usciti (lift = osservato/atteso), poi "
    "propone i trasformati dell'ultima estrazione pesati per il lift.",
    parametri={"finestra": 1000, "quanti": 5,
               "vertibile": True, "complemento": True, "figura": False},
)
def _trasformazioni(estrazioni, par):
    if not estrazioni:
        raise ValueError("Base dati vuota")

    def vertibile(n):
        d, u = divmod(n, 10)
        v = u * 10 + d
        return [v] if 1 <= v <= 90 and v != n else []

    def complemento(n):
        c = 90 - n
        return [c] if 1 <= c <= 90 and c != n else []

    def figura(n):
        f = n % 9 or 9
        return [m for m in range(f, 91, 9) if m != n]

    tipi = {}
    if par["vertibile"]:
        tipi["vertibile"] = vertibile
    if par["complemento"]:
        tipi["complemento"] = complemento
    if par["figura"]:
        tipi["figura"] = figura
    if not tipi:
        raise ValueError("Attiva almeno una trasformazione")

    finestra = int(par["finestra"]) or len(estrazioni)
    recenti = estrazioni[-finestra:]
    osservati = {t: 0 for t in tipi}
    attesi = {t: 0.0 for t in tipi}
    for (_, prima), (_, dopo) in zip(recenti, recenti[1:]):
        insieme_dopo = set(dopo)
        for t, trasforma in tipi.items():
            for n in prima:
                for m in trasforma(n):
                    osservati[t] += m in insieme_dopo
                    attesi[t] += len(dopo) / 90.0
    lift = {t: (osservati[t] / attesi[t] if attesi[t] else 0.0) for t in tipi}

    data_ultima, ultimi = recenti[-1]
    punteggio, origine = Counter(), {}
    for t, trasforma in tipi.items():
        for n in ultimi:
            for m in trasforma(n):
                punteggio[m] += lift[t]
                origine.setdefault(m, f"{t} di {n}")
    quanti = max(1, min(int(par["quanti"]), 10))
    scelti = [n for n, _ in punteggio.most_common(quanti)]

    dettagli = (
        f"Track record su {len(recenti)} estrazioni (lift osservato/atteso): "
        + ", ".join(f"{t} {lift[t]:.2f} ({osservati[t]} su {attesi[t]:.0f} attesi)"
                    for t in tipi)
        + f". Dall'estrazione del {data_ultima}: "
        + ", ".join(f"{n} = {origine[n]}" for n in sorted(scelti)))
    return {"numeri": scelti, "dettagli": dettagli}


@algoritmo(
    "meta-backtest", "Meta-algoritmo: l'analisi più in forma",
    "Invece di analizzare i numeri, analizza gli algoritmi: riesegue ogni "
    "analisi NET (con parametri di default) sulle ultime 'prove' estrazioni "
    "come se fossero da prevedere, conta i punti che ognuna avrebbe fatto e "
    "propone i numeri dell'algoritmo col miglior rendimento recente. Il "
    "dettaglio riporta la classifica completa e l'attesa del puro caso.",
    parametri={"prove": 20, "quanti": 5},
    in_meta=False,
)
def _meta_backtest(estrazioni, par):
    prove = max(5, min(int(par["prove"]), 200))
    quanti = max(1, min(int(par["quanti"]), 10))
    if len(estrazioni) < prove + 350:
        raise ValueError(f"Servono almeno {prove + 350} estrazioni")

    # configurazioni economiche per il backtest ripetuto
    forzati = {"media-differenziali": {"ml_differenziali": False,
                                       "finestra": 400}}
    candidati = {c: a for c, a in ALGORITMI.items()
                 if a["in_meta"] and not a["con_contesto"]}
    punti = {c: 0 for c in candidati}
    for t in range(len(estrazioni) - prove, len(estrazioni)):
        passato = estrazioni[:t]
        reale = set(estrazioni[t][1])
        for c, a in candidati.items():
            p = parametri_default(a)
            p.update(forzati.get(c, {}))
            try:
                numeri = a["fn"](passato, p).get("numeri", [])[:quanti]
            except Exception:
                continue
            punti[c] += len(set(numeri) & reale)

    classifica = sorted(punti.items(), key=lambda cp: -cp[1])
    migliore = classifica[0][0]
    p = parametri_default(ALGORITMI[migliore])
    p.update(forzati.get(migliore, {}))
    esito = ALGORITMI[migliore]["fn"](estrazioni, p)
    scelti = esito.get("numeri", [])[:quanti]

    k = len(estrazioni[-1][1])
    attesa = prove * quanti * k / 90.0
    dettagli = (
        f"Backtest sulle ultime {prove} estrazioni ({quanti} numeri a colpo, "
        f"attesa del caso {attesa:.1f} punti): "
        + ", ".join(f"{ALGORITMI[c]['titolo']} = {pt}" for c, pt in classifica)
        + f". Proposta dell'algoritmo migliore [{ALGORITMI[migliore]['titolo']}]: "
        + ", ".join(map(str, sorted(scelti))))
    return {"numeri": scelti, "dettagli": dettagli}


@algoritmo(
    "ciclometria", "Ciclometria: diametrali e terzine simmetriche",
    "Dispone i 90 numeri su un cerchio (distanza ciclometrica = min(diff, "
    "90-diff), max 45). Dall'ultima estrazione genera i completamenti delle "
    "figure: per ogni numero il suo diametrale (distanza esatta 45) e, per "
    "ogni coppia a distanza 30, il terzo vertice del triangolo equilatero "
    "(terzina simmetrica). Con 'quadrati' completa anche quadrati e croci "
    "ciclometriche (vertici a passo 22/23): dalle coppie diametrali propone "
    "i vertici perpendicolari, dalle coppie a distanza 22-23 i diametrali "
    "dei due numeri. Ogni famiglia di figure è pesata sul suo track record "
    "storico (lift = chiusure osservate/attese all'estrazione successiva).",
    parametri={"finestra": 1000, "quanti": 5,
               "diametrali": True, "terzine_simmetriche": True,
               "quadrati": False},
)
def _ciclometria(estrazioni, par):
    if not estrazioni:
        raise ValueError("Base dati vuota")
    if not (par["diametrali"] or par["terzine_simmetriche"] or par["quadrati"]):
        raise ValueError("Attiva almeno una figura ciclometrica")

    def diametrale(n):
        return (n - 1 + 45) % 90 + 1

    def completamenti(nums):
        """Candidati per figura generati da un'estrazione."""
        out = {}  # tipo -> set di numeri
        if par["diametrali"]:
            out["diametrale"] = {diametrale(n) for n in nums}
        if par["terzine_simmetriche"]:
            terzi = set()
            for i, a in enumerate(nums):
                for b in nums[i + 1:]:
                    a0, b0 = a - 1, b - 1
                    if (b0 - a0) % 90 == 30:
                        terzi.add((b0 + 30) % 90 + 1)
                    elif (a0 - b0) % 90 == 30:
                        terzi.add((a0 + 30) % 90 + 1)
            out["terzina"] = terzi
        if par["quadrati"]:
            vertici = set()
            for i, a in enumerate(nums):
                for b in nums[i + 1:]:
                    a0, b0 = a - 1, b - 1
                    diff = min((b0 - a0) % 90, (a0 - b0) % 90)
                    if diff == 45:          # diagonale: vertici perpendicolari
                        for scarto in (22, 23, 67, 68):
                            vertici.add((a0 + scarto) % 90 + 1)
                    elif diff in (22, 23):  # lato: vertici opposti
                        vertici.add((a0 + 45) % 90 + 1)
                        vertici.add((b0 + 45) % 90 + 1)
            out["quadrato"] = vertici
        return out

    finestra = int(par["finestra"]) or len(estrazioni)
    recenti = estrazioni[-finestra:]
    osservati, attesi = Counter(), Counter()
    for (_, prima), (_, dopo) in zip(recenti, recenti[1:]):
        insieme_dopo = set(dopo)
        for tipo, cand in completamenti(prima).items():
            osservati[tipo] += len(cand & insieme_dopo)
            attesi[tipo] += len(cand) * len(dopo) / 90.0
    lift = {t: (osservati[t] / attesi[t] if attesi[t] else 0.0)
            for t in osservati | attesi}

    data_ultima, ultimi = recenti[-1]
    punteggio, origine = Counter(), {}
    for tipo, cand in completamenti(ultimi).items():
        for n in cand:
            punteggio[n] += lift.get(tipo, 0) or 0.01
            origine.setdefault(n, tipo)
    quanti = max(1, min(int(par["quanti"]), 10))
    scelti = [n for n, _ in punteggio.most_common(quanti)]

    dettagli = (
        f"Track record su {len(recenti)} estrazioni: "
        + ", ".join(f"{t} lift {lift[t]:.2f} ({osservati[t]} chiusure su "
                    f"{attesi[t]:.0f} attese)" for t in lift)
        + f". Completamenti dell'estrazione del {data_ultima}: "
        + ", ".join(f"{n} ({origine[n]})" for n in sorted(scelti)))
    return {"numeri": scelti, "dettagli": dettagli}


@algoritmo(
    "markov", "Catena di Markov (spia generalizzata)",
    "Costruisce la matrice di transizione 90×90: quante volte il numero m in "
    "un'estrazione è seguito dal numero n in quella successiva, con "
    "smoothing di Laplace. Il punteggio di ogni candidato è la somma delle "
    "probabilità condizionate rispetto a tutti i numeri dell'ultima "
    "estrazione. Con 'doppia_spia' aggiunge il condizionamento sulle coppie "
    "(quali numeri seguono una data coppia uscita insieme).",
    parametri={"finestra": 2000, "quanti": 5, "doppia_spia": False},
)
def _markov(estrazioni, par):
    if len(estrazioni) < 100:
        raise ValueError("Servono almeno 100 estrazioni")
    finestra = int(par["finestra"]) or len(estrazioni)
    recenti = estrazioni[-finestra:]
    alpha = 1.0  # smoothing di Laplace

    seg_singolo = {m: Counter() for m in range(1, 91)}
    tot_singolo = Counter()
    seg_coppia, tot_coppia = {}, Counter()
    for (_, prima), (_, dopo) in zip(recenti, recenti[1:]):
        for m in prima:
            seg_singolo[m].update(dopo)
            tot_singolo[m] += len(dopo)
        if par["doppia_spia"]:
            ordinati = sorted(prima)
            for i, a in enumerate(ordinati):
                for b in ordinati[i + 1:]:
                    seg_coppia.setdefault((a, b), Counter()).update(dopo)
                    tot_coppia[(a, b)] += len(dopo)

    _, ultimi = recenti[-1]
    punteggio = Counter()
    for n in range(1, 91):
        s = sum((seg_singolo[m][n] + alpha) / (tot_singolo[m] + alpha * 90)
                for m in ultimi)
        punteggio[n] = s
    if par["doppia_spia"]:
        ordinati = sorted(ultimi)
        for i, a in enumerate(ordinati):
            for b in ordinati[i + 1:]:
                cnt, tot = seg_coppia.get((a, b)), tot_coppia[(a, b)]
                if cnt:
                    for n in range(1, 91):
                        punteggio[n] += (cnt[n] + alpha) / (tot + alpha * 90)

    quanti = max(1, min(int(par["quanti"]), 10))
    scelti = [n for n, _ in punteggio.most_common(quanti)]
    base = len(ultimi) * (len(ultimi) / 90.0)
    dettagli = (
        f"Matrice di transizione su {len(recenti)} estrazioni"
        f"{', con condizionamento sulle coppie' if par['doppia_spia'] else ''}"
        f" (smoothing {alpha}). Punteggi dei probabili (riferimento neutro "
        f"{base / len(ultimi):.4f} a numero): "
        + ", ".join(f"{n} ({punteggio[n]:.4f})" for n in sorted(scelti)))
    return {"numeri": scelti, "dettagli": dettagli}


@algoritmo(
    "somme", "Metodo delle somme",
    "Genera candidati dalle combinazioni sommative dell'ultima estrazione, "
    "con riporto 'fuori 90' (91 diventa 1): somma di tutti gli estratti, "
    "somma di ogni coppia, somma dei primi due numeri nell'ordine di "
    "estrazione. In più i generatori esotici: 'speculare' (complemento a 90 "
    "della somma di ogni coppia), 'spirale' (somme cumulative progressive "
    "n1, n1+n2, n1+n2+n3, ...), 'triangolare' (riduzione a triangolo: somme "
    "a coppie adiacenti ripetute fino al vertice) e 'progressione' "
    "(continuazione aritmetica di ogni coppia consecutiva ordinata: "
    "2b-a). Ogni generatore è pesato sul suo track record storico "
    "(uscite del candidato all'estrazione successiva, osservate vs attese).",
    parametri={"finestra": 1000, "quanti": 5,
               "somma_totale": True, "somma_coppie": True,
               "primi_due": True, "speculare": False, "spirale": False,
               "triangolare": False, "progressione": False},
)
def _somme(estrazioni, par):
    if not estrazioni:
        raise ValueError("Base dati vuota")

    def fuori90(x):
        return (x - 1) % 90 + 1

    def generatori(nums):
        out = {}
        if par["somma_totale"]:
            out["somma totale"] = {fuori90(sum(nums))}
        if par["somma_coppie"]:
            coppie = set()
            for i, a in enumerate(nums):
                for b in nums[i + 1:]:
                    coppie.add(fuori90(a + b))
            out["somma coppia"] = coppie
        if par["primi_due"]:
            out["primi due"] = {fuori90(nums[0] + nums[1])}
        if par["speculare"]:
            out["speculare"] = {fuori90(90 - (a + b))
                                for i, a in enumerate(nums)
                                for b in nums[i + 1:]}
        if par["spirale"]:
            parziali, acc = set(), 0
            for n in nums:
                acc += n
                parziali.add(fuori90(acc))
            out["spirale"] = parziali
        if par["triangolare"]:
            livello = list(nums)
            while len(livello) > 1:
                livello = [fuori90(a + b)
                           for a, b in zip(livello, livello[1:])]
            out["triangolare"] = {livello[0]}
        if par["progressione"]:
            ordinati = sorted(nums)
            out["progressione"] = {fuori90(2 * b - a)
                                   for a, b in zip(ordinati, ordinati[1:])}
        return out

    if not generatori(estrazioni[-1][1]):
        raise ValueError("Attiva almeno un generatore di somme")

    finestra = int(par["finestra"]) or len(estrazioni)
    recenti = estrazioni[-finestra:]
    osservati, attesi = Counter(), Counter()
    for (_, prima), (_, dopo) in zip(recenti, recenti[1:]):
        insieme_dopo = set(dopo)
        for tipo, cand in generatori(prima).items():
            osservati[tipo] += len(cand & insieme_dopo)
            attesi[tipo] += len(cand) * len(dopo) / 90.0
    lift = {t: (osservati[t] / attesi[t] if attesi[t] else 0.0)
            for t in osservati | attesi}

    data_ultima, ultimi = recenti[-1]
    punteggio, origine = Counter(), {}
    for tipo, cand in generatori(ultimi).items():
        for n in cand:
            punteggio[n] += lift.get(tipo, 0) or 0.01
            origine.setdefault(n, tipo)
    quanti = max(1, min(int(par["quanti"]), 10))
    scelti = [n for n, _ in punteggio.most_common(quanti)]

    dettagli = (
        f"Track record su {len(recenti)} estrazioni: "
        + ", ".join(f"{t} lift {lift[t]:.2f}" for t in lift)
        + f". Dall'estrazione del {data_ultima} "
        f"({', '.join(map(str, ultimi))}): "
        + ", ".join(f"{n} ({origine[n]})" for n in sorted(scelti)))
    return {"numeri": scelti, "dettagli": dettagli}


@algoritmo(
    "indice-convenienza", "Indice di convenienza",
    "Ritardatari 'veri': per ogni numero rapporta il ritardo attuale al "
    "proprio ritardo medio storico (un numero a ritardo 100 con media 18 è "
    "più anomalo di uno a 110 con media 25) e aggiunge il rapporto col "
    "massimo storico. IC = ritardo/media + ritardo/massimo. Esclude i numeri "
    "con meno di 'min_uscite' presenze (statistica inaffidabile).",
    parametri={"finestra": 0, "quanti": 5, "min_uscite": 10},
)
def _indice_convenienza(estrazioni, par):
    if not estrazioni:
        raise ValueError("Base dati vuota")
    finestra = int(par["finestra"]) or len(estrazioni)
    recenti = estrazioni[-finestra:]
    tot = len(recenti)
    min_uscite = max(2, int(par["min_uscite"]))

    indici, info = {}, {}
    for n in range(1, 91):
        posizioni = [i for i, (_, nums) in enumerate(recenti) if n in nums]
        if len(posizioni) < min_uscite:
            continue
        gaps = [b - a for a, b in zip(posizioni, posizioni[1:])]
        media = sum(gaps) / len(gaps) if gaps else tot
        massimo = max(gaps) if gaps else tot
        attuale = tot - 1 - posizioni[-1]
        indici[n] = attuale / media + attuale / massimo
        info[n] = (attuale, media, massimo)

    quanti = max(1, min(int(par["quanti"]), 10))
    scelti = sorted(indici, key=lambda n: -indici[n])[:quanti]
    dettagli = (
        f"Su {tot} estrazioni (IC = ritardo/media + ritardo/massimo): "
        + ", ".join(
            f"{n} (IC {indici[n]:.2f}: rit. {info[n][0]}, media "
            f"{info[n][1]:.1f}, max {info[n][2]})" for n in sorted(scelti)))
    return {"numeri": scelti, "dettagli": dettagli}


@algoritmo(
    "isotopi", "Isotopi (distanza 9 a parità di posizione)",
    "Cerca nelle estrazioni consecutive le coppie di numeri usciti NELLA "
    "STESSA POSIZIONE con distanza ciclometrica pari a 'distanza' (classico: "
    "9). Quando trova un isotopo propone la continuazione della progressione "
    "nella stessa direzione (es. 12 al 3º posto, poi 21 al 3º posto → "
    "propone 30). Scansiona gli ultimi 'colpi_ricerca' passaggi e pesa ogni "
    "proposta sul track record storico delle continuazioni.",
    parametri={"finestra": 2000, "distanza": 9, "colpi_ricerca": 10,
               "quanti": 5},
    in_meta=False,
)
def _isotopi(estrazioni, par):
    if len(estrazioni) < 50:
        raise ValueError("Servono almeno 50 estrazioni")
    dist = max(1, min(int(par["distanza"]), 45))
    finestra = int(par["finestra"]) or len(estrazioni)
    recenti = estrazioni[-finestra:]
    colpi = max(1, min(int(par["colpi_ricerca"]), 100))
    quanti = max(1, min(int(par["quanti"]), 10))

    def fuori90(x):
        return (x - 1) % 90 + 1

    def isotopi_della_coppia(prima, dopo):
        """[(posizione, n_prev, n_succ, continuazione), ...]"""
        trovati = []
        for i in range(min(len(prima), len(dopo))):
            a, b = prima[i], dopo[i]
            passo = (b - a) % 90
            if passo == dist:
                trovati.append((i, a, b, fuori90(b + dist)))
            elif passo == 90 - dist:
                trovati.append((i, a, b, fuori90(b - dist)))
        return trovati

    # track record: la continuazione esce all'estrazione successiva?
    osservati, attesi = 0, 0.0
    for t in range(1, len(recenti) - 1):
        eventi = isotopi_della_coppia(recenti[t - 1][1], recenti[t][1])
        if not eventi:
            continue
        dopo = set(recenti[t + 1][1])
        for _, _, _, cont in eventi:
            osservati += cont in dopo
            attesi += len(recenti[t + 1][1]) / 90.0
    lift = osservati / attesi if attesi else 0.0

    # proposta: isotopi negli ultimi 'colpi' passaggi consecutivi
    punteggio, origine = Counter(), {}
    inizio = max(1, len(recenti) - colpi)
    for t in range(len(recenti) - 1, inizio - 1, -1):
        for pos, a, b, cont in isotopi_della_coppia(recenti[t - 1][1],
                                                    recenti[t][1]):
            recenza = 1.0 / (len(recenti) - t + 1)  # eventi recenti pesano di più
            punteggio[cont] += (lift or 0.01) * recenza
            origine.setdefault(
                cont, f"{a}→{b} al {pos + 1}º posto ({recenti[t][0]})")
    scelti = [n for n, _ in punteggio.most_common(quanti)]

    if not scelti:
        dettagli = (f"Nessun isotopo a distanza {dist} negli ultimi {colpi} "
                    f"passaggi. Track record storico delle continuazioni: "
                    f"lift {lift:.2f} ({osservati} su {attesi:.0f} attese).")
    else:
        dettagli = (
            f"Track record continuazioni su {len(recenti)} estrazioni: lift "
            f"{lift:.2f} ({osservati} su {attesi:.0f} attese). Isotopi "
            f"recenti: " + "; ".join(f"{n} da {origine[n]}"
                                     for n in sorted(scelti)))
    return {"numeri": scelti, "dettagli": dettagli}


@algoritmo(
    "montecarlo", "Simulazione Monte Carlo",
    "Simula migliaia di estrazioni future campionando senza ripetizione con "
    "pesi derivati dalla finestra recente — 'pesi': frequenza (i numeri più "
    "usciti pesano di più), ritardo (i più ritardatari pesano di più) o "
    "misto. Classifica i numeri per presenza negli scenari simulati. Il "
    "seme è derivato dall'ultima estrazione: a parità di dati il risultato "
    "è riproducibile.",
    parametri={"finestra": 200, "simulazioni": 5000, "quanti": 5,
               "pesi": ["frequenza", "ritardo", "misto"]},
)
def _montecarlo(estrazioni, par):
    if len(estrazioni) < 50:
        raise ValueError("Servono almeno 50 estrazioni")
    finestra = int(par["finestra"]) or len(estrazioni)
    recenti = estrazioni[-finestra:]
    k = len(recenti[-1][1])
    quanti = max(1, min(int(par["quanti"]), 10))
    nsim = max(100, min(int(par["simulazioni"]), 100000))

    freq = Counter()
    ultimo = {}
    for i, (_, nums) in enumerate(recenti):
        for n in nums:
            freq[n] += 1
            ultimo[n] = i
    ritardo = {n: len(recenti) - 1 - ultimo.get(n, -1) for n in range(1, 91)}

    f = np.array([freq.get(n, 0) + 1.0 for n in range(1, 91)])
    r = np.array([ritardo[n] + 1.0 for n in range(1, 91)])
    if par["pesi"] == "ritardo":
        w = r
    elif par["pesi"] == "misto":
        w = f / f.sum() + r / r.sum()
    else:
        w = f
    p = w / w.sum()

    seme = abs(hash(recenti[-1][0])) % (2 ** 32)
    rng = np.random.default_rng(seme)
    presenze = np.zeros(90)
    for _ in range(nsim):
        estratti = rng.choice(90, size=k, replace=False, p=p)
        presenze[estratti] += 1

    ordine = np.argsort(-presenze)[:quanti]
    scelti = [int(i + 1) for i in ordine]
    dettagli = (
        f"{nsim} estrazioni simulate (pesi: {par['pesi']}, finestra "
        f"{len(recenti)}, seme {seme}). Presenza negli scenari: "
        + ", ".join(f"{n} ({100 * presenze[n - 1] / nsim:.1f}%)"
                    for n in sorted(scelti))
        + f" — riferimento neutro {100 * k / 90:.1f}%.")
    return {"numeri": scelti, "dettagli": dettagli}


def _fuori90(x):
    return (x - 1) % 90 + 1


def _dist_ciclometrica(a, b):
    d = abs(a - b)
    return min(d, 90 - d)


@algoritmo(
    "spie-avanzate", "Varianti di spia",
    "Evoluzioni del numero spia, selezionabili con 'tipo': RITARDATA (la "
    "spia agisce con ritardo: i probabili sono i numeri storicamente usciti "
    "'ritardo_spia' colpi dopo i numeri dell'estrazione corrispondente), "
    "SEQUENZA (seguenti entro i prossimi 'colpi' invece che al colpo "
    "subito dopo), GEMELLARE (spie sono i gemelli 11, 22...88: seguenti "
    "del gemello uscito più di recente), RECIPROCA (coppie che si seguono "
    "a vicenda sopra l'atteso, in entrambe le direzioni), INVERTITA (i "
    "seguenti del vertibile di ogni numero dell'ultima estrazione).",
    parametri={"tipo": ["ritardata", "sequenza", "gemellare",
                        "reciproca", "invertita"],
               "finestra": 2000, "ritardo_spia": 3, "colpi": 3, "quanti": 5},
)
def _spie_avanzate(estrazioni, par):
    if len(estrazioni) < 100:
        raise ValueError("Servono almeno 100 estrazioni")
    finestra = int(par["finestra"]) or len(estrazioni)
    recenti = estrazioni[-finestra:]
    quanti = max(1, min(int(par["quanti"]), 10))
    tipo = par["tipo"]

    def seguiti_con_lag(lag):
        """seg[m] = Counter dei numeri usciti lag colpi dopo m; occ[m] = casi."""
        seg = {m: Counter() for m in range(1, 91)}
        occ = Counter()
        for t in range(lag, len(recenti)):
            for m in recenti[t - lag][1]:
                seg[m].update(recenti[t][1])
                occ[m] += 1
        return seg, occ

    def vertibile(n):
        d, u = divmod(n, 10)
        v = u * 10 + d
        return v if 1 <= v <= 90 else n

    punteggio = Counter()
    spiegazione = ""

    if tipo == "ritardata":
        lag = max(1, min(int(par["ritardo_spia"]), 50))
        seg, occ = seguiti_con_lag(lag)
        spie = recenti[-lag][1]  # i numeri che 'agiscono' sulla prossima
        for m in spie:
            if occ[m]:
                for n, c in seg[m].items():
                    punteggio[n] += c / occ[m]
        spiegazione = (f"Spie con ritardo {lag}: estrazione del "
                       f"{recenti[-lag][0]} ({', '.join(map(str, spie))})")

    elif tipo == "sequenza":
        colpi = max(1, min(int(par["colpi"]), 20))
        seg = {m: Counter() for m in range(1, 91)}
        occ = Counter()
        for t in range(len(recenti) - 1):
            for m in recenti[t][1]:
                occ[m] += 1
                for dt in range(1, min(colpi, len(recenti) - 1 - t) + 1):
                    seg[m].update(recenti[t + dt][1])
        for m in recenti[-1][1]:
            if occ[m]:
                for n, c in seg[m].items():
                    punteggio[n] += c / occ[m]
        spiegazione = (f"Seguenti entro {colpi} colpi dei numeri "
                       f"dell'ultima estrazione")

    elif tipo == "gemellare":
        gemelli = {11, 22, 33, 44, 55, 66, 77, 88}
        seg, occ = seguiti_con_lag(1)
        spia = None
        for t in range(len(recenti) - 1, -1, -1):
            usciti = gemelli & set(recenti[t][1])
            if usciti:
                spia = (sorted(usciti)[0], recenti[t][0])
                break
        if not spia:
            return {"numeri": [], "dettagli": "Nessun gemello in archivio."}
        m = spia[0]
        if occ[m]:
            for n, c in seg[m].items():
                punteggio[n] += c / occ[m]
        spiegazione = f"Spia gemellare: {m} (uscito il {spia[1]})"

    elif tipo == "reciproca":
        seg, occ = seguiti_con_lag(1)
        k_medio = sum(len(e[1]) for e in recenti) / len(recenti)
        for m in recenti[-1][1]:
            for n in range(1, 91):
                if n == m or not occ[m] or not occ[n]:
                    continue
                if seg[m][n] < 3 or seg[n][m] < 3:
                    continue
                lift_mn = (seg[m][n] / occ[m]) / (k_medio / 90)
                lift_nm = (seg[n][m] / occ[n]) / (k_medio / 90)
                reciproco = min(lift_mn, lift_nm)
                if reciproco > punteggio[n]:
                    punteggio[n] = reciproco
        spiegazione = ("Coppie reciproche (lift minimo nelle due direzioni, "
                       "almeno 3 casi per direzione)")

    elif tipo == "invertita":
        seg, occ = seguiti_con_lag(1)
        for m in recenti[-1][1]:
            v = vertibile(m)
            if occ[v]:
                for n, c in seg[v].items():
                    punteggio[n] += c / occ[v]
        spiegazione = ("Seguenti dei vertibili dell'ultima estrazione: "
                       + ", ".join(f"{m}→{vertibile(m)}"
                                   for m in recenti[-1][1]))

    scelti = [n for n, _ in punteggio.most_common(quanti)]
    k = len(recenti[-1][1])
    dettagli = (f"Tipo '{tipo}' su {len(recenti)} estrazioni. {spiegazione}. "
                f"Probabili (punteggio; riferimento neutro "
                f"{k * k / 90:.2f}): "
                + ", ".join(f"{n} ({punteggio[n]:.2f})"
                            for n in sorted(scelti)))
    return {"numeri": scelti, "dettagli": dettagli}


@algoritmo(
    "fase-multiruota", "Numeri in fase e ritardi sincronici (solo Lotto)",
    "Analisi trasversale a tutte le ruote. Tipo IN_FASE: numeri usciti su "
    "almeno 'min_ruote' ruote diverse negli ultimi 'colpi' concorsi (numeri "
    "'caldi ovunque'). Tipo RITARDI_SINCRONICI: numeri con ritardo di almeno "
    "'soglia_ritardo' estrazioni contemporaneamente su almeno 'min_ruote' "
    "ruote. I probabili sono proposti per la ruota selezionata.",
    parametri={"tipo": ["in_fase", "ritardi_sincronici"],
               "finestra": 500, "colpi": 5, "soglia_ritardo": 30,
               "min_ruote": 4, "quanti": 5},
    in_meta=False,
)
def _fase_multiruota(estrazioni, par, contesto):
    if contesto["gioco"] != "lotto":
        raise ValueError("Analisi disponibile solo per il Lotto: "
                         "serve il confronto tra ruote")
    con = contesto["con"]
    finestra = max(50, min(int(par["finestra"]) or 100000, 100000))
    colpi = max(1, min(int(par["colpi"]), 100))
    soglia = max(1, int(par["soglia_ritardo"]))
    min_ruote = max(2, min(int(par["min_ruote"]), 11))
    quanti = max(1, min(int(par["quanti"]), 10))

    date = [r["data"] for r in con.execute(
        "SELECT DISTINCT data FROM lotto ORDER BY data DESC LIMIT ?",
        (finestra,))][::-1]
    per_ruota = {w: [] for w in db.RUOTE}  # liste cronologiche di insiemi
    righe = con.execute(
        "SELECT data, ruota, n1, n2, n3, n4, n5 FROM lotto "
        "WHERE data >= ? ORDER BY data", (date[0],))
    per_data_ruota = {}
    for r in righe:
        per_data_ruota.setdefault(r["data"], {})[r["ruota"]] = {
            r["n1"], r["n2"], r["n3"], r["n4"], r["n5"]}
    for d in date:
        for w in db.RUOTE:
            estratto = per_data_ruota.get(d, {}).get(w)
            if estratto:
                per_ruota[w].append(estratto)

    punteggio, dettagli_num = Counter(), {}
    if par["tipo"] == "in_fase":
        for n in range(1, 91):
            ruote_calde = [w for w in db.RUOTE
                           if any(n in e for e in per_ruota[w][-colpi:])]
            if len(ruote_calde) >= min_ruote:
                punteggio[n] = len(ruote_calde)
                dettagli_num[n] = f"{len(ruote_calde)} ruote"
        descr = (f"Numeri usciti su almeno {min_ruote} ruote negli ultimi "
                 f"{colpi} concorsi")
    else:
        for n in range(1, 91):
            ritardi_alti = []
            for w in db.RUOTE:
                serie = per_ruota[w]
                rit = next((i for i, e in enumerate(reversed(serie))
                            if n in e), len(serie))
                if rit >= soglia:
                    ritardi_alti.append((w, rit))
            if len(ritardi_alti) >= min_ruote:
                punteggio[n] = sum(r for _, r in ritardi_alti)
                dettagli_num[n] = (f"{len(ritardi_alti)} ruote, ritardi "
                                   + "/".join(str(r) for _, r in ritardi_alti))
        descr = (f"Numeri con ritardo ≥{soglia} contemporaneamente su almeno "
                 f"{min_ruote} ruote")

    scelti = [n for n, _ in punteggio.most_common(quanti)]
    if not scelti:
        return {"numeri": [], "dettagli": descr + ": nessun numero trovato "
                "con questi parametri (prova ad allentare le soglie)."}
    dettagli = (descr + f" (su {len(date)} concorsi): "
                + ", ".join(f"{n} ({dettagli_num[n]})" for n in sorted(scelti)))
    return {"numeri": scelti, "dettagli": dettagli}


@algoritmo(
    "indicatori-multipli", "Indicatori multipli (punteggio composito)",
    "Classifica unica che fonde più indicatori, ognuno col suo peso "
    "regolabile (0 = escluso): ritardo attuale, frequenza nella finestra, "
    "trend (frequenza recente vs storica) e indice di convenienza "
    "(ritardo/media storica). Ogni indicatore è convertito in percentile "
    "0-1 sui 90 numeri, poi combinato: punteggio = somma pesata / somma "
    "dei pesi.",
    parametri={"finestra": 300, "peso_ritardo": 1.0, "peso_frequenza": 1.0,
               "peso_trend": 1.0, "peso_ic": 1.0, "quanti": 5},
)
def _indicatori_multipli(estrazioni, par):
    if len(estrazioni) < 100:
        raise ValueError("Servono almeno 100 estrazioni")
    finestra = int(par["finestra"]) or len(estrazioni)
    recenti = estrazioni[-finestra:]
    tot = len(recenti)
    pesi = {k: max(0.0, float(par[f"peso_{k}"]))
            for k in ("ritardo", "frequenza", "trend", "ic")}
    if not any(pesi.values()):
        raise ValueError("Imposta almeno un peso > 0")
    quanti = max(1, min(int(par["quanti"]), 10))

    breve = max(10, tot // 10)
    indicatori = {k: {} for k in pesi}
    for n in range(1, 91):
        posizioni = [i for i, (_, nums) in enumerate(recenti) if n in nums]
        freq = len(posizioni)
        rit = tot - 1 - (posizioni[-1] if posizioni else -1)
        freq_breve = sum(1 for i in posizioni if i >= tot - breve)
        gaps = [b - a for a, b in zip(posizioni, posizioni[1:])]
        media_gap = sum(gaps) / len(gaps) if gaps else tot
        indicatori["ritardo"][n] = rit
        indicatori["frequenza"][n] = freq
        indicatori["trend"][n] = freq_breve / breve - freq / tot
        indicatori["ic"][n] = rit / media_gap

    def percentili(valori):
        ordine = sorted(valori, key=valori.get)
        return {n: i / 89.0 for i, n in enumerate(ordine)}

    perc = {k: percentili(v) for k, v in indicatori.items()}
    somma_pesi = sum(pesi.values())
    punteggio = {n: sum(pesi[k] * perc[k][n] for k in pesi) / somma_pesi
                 for n in range(1, 91)}
    scelti = sorted(punteggio, key=lambda n: -punteggio[n])[:quanti]

    dettagli = (
        f"Pesi: " + ", ".join(f"{k}={pesi[k]:g}" for k in pesi)
        + f" su {tot} estrazioni. Probabili (punteggio composito 0-1): "
        + ", ".join(
            f"{n} ({punteggio[n]:.2f}: rit.{indicatori['ritardo'][n]}, "
            f"freq.{indicatori['frequenza'][n]}, IC {indicatori['ic'][n]:.1f})"
            for n in sorted(scelti)))
    return {"numeri": scelti, "dettagli": dettagli}


@algoritmo(
    "gemelli", "Gemelli e cadenze gemelle",
    "I gemelli (11, 22, ... 88) come famiglia: classifica per 'criterio' "
    "(frequenza, ritardo o sotto_media rispetto all'atteso teorico) e "
    "misura il 'richiamo' storico (quanto spesso un gemello è seguito da "
    "un altro gemello entro 'colpi_richiamo' estrazioni, rispetto "
    "all'atteso). Con 'cadenze_gemelle' aggiunge i numeri più ritardatari "
    "della coppia di cadenze storicamente più correlata (le cadenze che si "
    "muovono insieme).",
    parametri={"criterio": ["ritardo", "frequenza", "sotto_media"],
               "finestra": 1000, "colpi_richiamo": 5,
               "cadenze_gemelle": False, "quanti": 4},
    in_meta=False,
)
def _gemelli(estrazioni, par):
    if len(estrazioni) < 100:
        raise ValueError("Servono almeno 100 estrazioni")
    finestra = int(par["finestra"]) or len(estrazioni)
    recenti = estrazioni[-finestra:]
    tot = len(recenti)
    quanti = max(1, min(int(par["quanti"]), 10))
    gemelli = [11, 22, 33, 44, 55, 66, 77, 88]

    freq, ultimo = Counter(), {}
    estratti_tot = 0
    for i, (_, nums) in enumerate(recenti):
        estratti_tot += len(nums)
        for n in nums:
            freq[n] += 1
            ultimo[n] = i
    rit = {n: tot - 1 - ultimo.get(n, -1) for n in gemelli}
    attesa = estratti_tot / 90.0

    criterio = par["criterio"]
    if criterio == "frequenza":
        ordinati = sorted(gemelli, key=lambda n: -freq[n])
    elif criterio == "sotto_media":
        ordinati = sorted(gemelli, key=lambda n: freq[n] - attesa)
    else:
        ordinati = sorted(gemelli, key=lambda n: -rit[n])
    scelti = ordinati[:quanti]

    # richiamo: gemello seguito da gemello entro K colpi
    colpi = max(1, min(int(par["colpi_richiamo"]), 30))
    osservati, attesi_r = 0, 0.0
    p_gemello = 8 / 90.0
    for t in range(len(recenti) - 1):
        if not set(recenti[t][1]) & set(gemelli):
            continue
        seguono = set()
        for dt in range(1, min(colpi, len(recenti) - 1 - t) + 1):
            seguono |= set(recenti[t + dt][1])
        osservati += bool(seguono & set(gemelli))
        attesi_r += 1 - (1 - p_gemello) ** (len(seguono))
    lift = osservati / attesi_r if attesi_r else 0.0

    dettagli = (
        f"Gemelli per '{criterio}' su {tot} estrazioni: "
        + ", ".join(f"{n} (×{freq[n]}, atteso {attesa:.0f}, rit. {rit[n]})"
                    for n in scelti)
        + f". Richiamo gemello→gemello entro {colpi} colpi: lift {lift:.2f} "
        f"({osservati} casi su {attesi_r:.0f} attesi).")

    if par["cadenze_gemelle"]:
        serie = np.zeros((tot, 10))
        for i, (_, nums) in enumerate(recenti):
            for n in nums:
                serie[i][n % 10] += 1
        corr_max, coppia = -2.0, (0, 1)
        for a in range(10):
            for b in range(a + 1, 10):
                c = np.corrcoef(serie[:, a], serie[:, b])[0, 1]
                if c > corr_max:
                    corr_max, coppia = c, (a, b)
        rit_cad = {}
        for cad in coppia:
            numeri_cad = [n for n in range(1, 91) if n % 10 == cad]
            rit_cad.update({n: tot - 1 - ultimo.get(n, -1)
                            for n in numeri_cad})
        extra = sorted(rit_cad, key=lambda n: -rit_cad[n])[:2]
        scelti = scelti + [n for n in extra if n not in scelti]
        dettagli += (
            f" Cadenze gemelle: {coppia[0]} e {coppia[1]} (correlazione "
            f"{corr_max:.2f}) → aggiunti i loro ritardatari: "
            + ", ".join(f"{n} (rit. {rit_cad[n]})" for n in extra))

    return {"numeri": scelti, "dettagli": dettagli}


@algoritmo(
    "archi-ciclometrici", "Archi ciclometrici",
    "Divide il cerchio 1-90 in archi di 'ampiezza' numeri consecutivi "
    "(circolari, sfalsati di 'passo': con passo < ampiezza gli archi si "
    "sovrappongono). Per ogni arco misura frequenza, atteso teorico e "
    "ritardo, seleziona i migliori 'top_archi' secondo il 'criterio' e "
    "propone i numeri interni agli archi scelti, ordinati per il criterio "
    "a livello di singolo numero. Le 'cerniere' (numeri in più archi "
    "selezionati) hanno precedenza.",
    parametri={"finestra": 300, "ampiezza": 10, "passo": 5, "top_archi": 3,
               "criterio": ["sotto_media", "frequenza", "ritardo"],
               "quanti": 5},
)
def _archi_ciclometrici(estrazioni, par):
    if not estrazioni:
        raise ValueError("Base dati vuota")
    finestra = int(par["finestra"]) or len(estrazioni)
    recenti = estrazioni[-finestra:]
    tot = len(recenti)
    ampiezza = max(2, min(int(par["ampiezza"]), 45))
    passo = max(1, min(int(par["passo"]), ampiezza))
    top_a = max(1, min(int(par["top_archi"]), 90 // passo))
    quanti = max(1, min(int(par["quanti"]), 10))

    archi = [tuple(_fuori90(inizio + j) for j in range(ampiezza))
             for inizio in range(1, 91, passo)]

    freq, ultimo = Counter(), {}
    estratti_tot = 0
    for i, (_, nums) in enumerate(recenti):
        estratti_tot += len(nums)
        for n in nums:
            freq[n] += 1
            ultimo[n] = i
    rit = {n: tot - 1 - ultimo.get(n, -1) for n in range(1, 91)}

    freq_arco = {a: sum(freq[n] for n in a) for a in archi}
    rit_arco = {a: min(rit[n] for n in a) for a in archi}
    attesa = estratti_tot * ampiezza / 90.0

    criterio = par["criterio"]
    if criterio == "frequenza":
        chiave_a = lambda a: -freq_arco[a]
        chiave_n = lambda n: -freq[n]
    elif criterio == "ritardo":
        chiave_a = lambda a: -rit_arco[a]
        chiave_n = lambda n: -rit[n]
    else:
        chiave_a = lambda a: freq_arco[a] - attesa
        chiave_n = lambda n: -rit[n]

    migliori = sorted(archi, key=chiave_a)[:top_a]
    conteggio_archi = Counter()
    for a in migliori:
        conteggio_archi.update(a)
    candidati = sorted(conteggio_archi,
                       key=lambda n: (-conteggio_archi[n], chiave_n(n), n))
    scelti = candidati[:quanti]
    cerniere = [n for n in scelti if conteggio_archi[n] > 1]

    def nome(a):
        return f"{a[0]}-{a[-1]}"

    dettagli = (
        f"Criterio '{criterio}', archi di {ampiezza} sfalsati di {passo} su "
        f"{tot} estrazioni. Archi scelti: "
        + ", ".join(f"{nome(a)} (×{freq_arco[a]}, atteso {attesa:.0f}, "
                    f"rit. {rit_arco[a]})" for a in migliori)
        + (f". Cerniere (in più archi): "
           + ", ".join(map(str, sorted(cerniere))) if cerniere else ""))
    return {"numeri": scelti, "dettagli": dettagli}


@algoritmo(
    "nodi-numerici", "Nodi numerici",
    "Individua i numeri al centro di più relazioni ciclometriche con "
    "l'ultima estrazione: per ogni candidato conta quanti numeri appena "
    "usciti gli sono legati da una delle distanze in 'distanze_nodo' "
    "(elenco separato da virgole). Servono almeno 'min_relazioni' legami. "
    "Il track record storico misura quanto spesso i nodi così definiti "
    "escono davvero al colpo successivo.",
    parametri={"finestra": 1000, "distanze_nodo": "45,30,15",
               "min_relazioni": 2, "quanti": 5},
)
def _nodi_numerici(estrazioni, par):
    if not estrazioni:
        raise ValueError("Base dati vuota")
    try:
        distanze = sorted({int(x) for x in
                           str(par["distanze_nodo"]).replace(" ", "").split(",")
                           if x and 1 <= int(x) <= 45})
    except ValueError:
        distanze = [45, 30, 15]
    if not distanze:
        distanze = [45, 30, 15]
    min_rel = max(1, min(int(par["min_relazioni"]), 10))
    finestra = int(par["finestra"]) or len(estrazioni)
    recenti = estrazioni[-finestra:]
    quanti = max(1, min(int(par["quanti"]), 10))

    def nodi(nums):
        relazioni = Counter()
        for n in range(1, 91):
            for m in nums:
                if n != m and _dist_ciclometrica(n, m) in distanze:
                    relazioni[n] += 1
        return {n: r for n, r in relazioni.items() if r >= min_rel}

    osservati, attesi = 0, 0.0
    for (_, prima), (_, dopo) in zip(recenti, recenti[1:]):
        candidati = nodi(prima)
        if not candidati:
            continue
        insieme_dopo = set(dopo)
        osservati += sum(1 for n in candidati if n in insieme_dopo)
        attesi += len(candidati) * len(dopo) / 90.0
    lift = osservati / attesi if attesi else 0.0

    data_ultima, ultimi = recenti[-1]
    candidati = nodi(ultimi)
    if not candidati:
        return {"numeri": [], "dettagli":
                f"Nessun numero con almeno {min_rel} relazioni "
                f"(distanze {distanze}) con l'estrazione del {data_ultima}."}
    scelti = sorted(candidati, key=lambda n: (-candidati[n], n))[:quanti]
    dettagli = (
        f"Distanze di relazione {distanze}, almeno {min_rel} legami. Track "
        f"record su {len(recenti)} estrazioni: lift {lift:.2f} ({osservati} "
        f"uscite su {attesi:.0f} attese). Nodi dell'estrazione del "
        f"{data_ultima}: "
        + ", ".join(f"{n} ({candidati[n]} relazioni)" for n in sorted(scelti)))
    return {"numeri": scelti, "dettagli": dettagli}


@algoritmo(
    "diagonale-estrattiva", "Diagonale estrattiva",
    "Legge il quadro estrazionale come matrice (righe = estrazioni in "
    "ordine cronologico, colonne = posizioni di estrazione) e continua le "
    "diagonali: per ogni diagonale discendente (verso destra e verso "
    "sinistra) che termina sull'ultima riga, il candidato è la "
    "continuazione aritmetica 2b-a (fuori 90) dei suoi ultimi due numeri. "
    "Il track record storico misura quanto spesso le continuazioni "
    "diagonali escono davvero al colpo successivo.",
    parametri={"finestra": 1000, "quanti": 5},
)
def _diagonale_estrattiva(estrazioni, par):
    if len(estrazioni) < 10:
        raise ValueError("Servono almeno 10 estrazioni")
    finestra = int(par["finestra"]) or len(estrazioni)
    recenti = estrazioni[-finestra:]
    k = len(recenti[-1][1])
    quanti = max(1, min(int(par["quanti"]), 10))

    def continuazioni(penultima, ultima):
        """{candidato: descrizione} dalle diagonali che finiscono in ultima."""
        out = {}
        for j in range(k):
            for direzione, nome in ((1, "↘"), (-1, "↙")):
                j_prev = j - direzione
                if 0 <= j_prev < k and j_prev < len(penultima) and j < len(ultima):
                    a, b = penultima[j_prev], ultima[j]
                    cand = _fuori90(2 * b - a)
                    out.setdefault(cand, f"{nome} {a}→{b}")
        return out

    osservati, attesi = 0, 0.0
    for t in range(1, len(recenti) - 1):
        cand = continuazioni(recenti[t - 1][1], recenti[t][1])
        if not cand:
            continue
        insieme_dopo = set(recenti[t + 1][1])
        osservati += sum(1 for n in cand if n in insieme_dopo)
        attesi += len(cand) * len(recenti[t + 1][1]) / 90.0
    lift = osservati / attesi if attesi else 0.0

    cand = continuazioni(recenti[-2][1], recenti[-1][1])
    scelti = sorted(cand)[:quanti]
    dettagli = (
        f"Track record su {len(recenti)} estrazioni: lift {lift:.2f} "
        f"({osservati} uscite su {attesi:.0f} attese). Continuazioni delle "
        f"diagonali del quadro ({recenti[-2][0]} → {recenti[-1][0]}): "
        + ", ".join(f"{n} ({cand[n]})" for n in scelti))
    return {"numeri": scelti, "dettagli": dettagli}


@algoritmo(
    "genetico", "Algoritmo genetico (2 modalità)",
    "MODALITÀ 'indicatori' (uso sensato del GA): evolve direttamente una "
    "combinazione di numeri massimizzando una fitness composta dai nostri "
    "indicatori, ognuno col suo peso regolabile: frequenza, ritardo, "
    "coesione di coppia (coppie storicamente frequenti) ed equilibrio "
    "(pari/dispari, bassi/alti, decine distinte). MODALITÀ 'mt19937' "
    "(porting onesto di gnh1201/lotto_genetic_algorithm): il GA cerca i 4 "
    "parametri di un generatore Mersenne Twister che 'rigenera' le "
    "estrazioni di addestramento — ma a differenza dell'originale il "
    "rendimento è misurato su estrazioni di validazione MAI viste dal GA, "
    "e riportato nel dettaglio accanto all'attesa del caso. Selezione a "
    "torneo, crossover, mutazione, elitismo; seme derivato dall'ultima "
    "estrazione (risultato riproducibile).",
    parametri={"modalita": ["indicatori", "mt19937"], "quanti": 5,
               "popolazione": 40, "generazioni": 30, "finestra": 300,
               "peso_frequenza": 1.0, "peso_ritardo": 1.0,
               "peso_coppie": 1.0, "peso_equilibrio": 1.0,
               "addestramento": 60, "validazione": 20},
    in_meta=False,
)
def _genetico(estrazioni, par):
    import random as pyrandom

    if len(estrazioni) < 150:
        raise ValueError("Servono almeno 150 estrazioni")
    quanti = max(1, min(int(par["quanti"]), 10))
    pop_n = max(10, min(int(par["popolazione"]), 300))
    gen_n = max(5, min(int(par["generazioni"]), 300))
    rng = pyrandom.Random(f"ga-{estrazioni[-1][0]}")

    def evolvi(popolazione, fitness, incrocia, muta):
        """GA generico: torneo a 3, elitismo 2, ritorna il migliore."""
        valutati = sorted(((fitness(i), i) for i in popolazione),
                          key=lambda fi: -fi[0])
        for _ in range(gen_n):
            nuova = [valutati[0][1], valutati[1][1]]  # elitismo
            while len(nuova) < pop_n:
                genitori = []
                for _ in range(2):
                    torneo = rng.sample(valutati, min(3, len(valutati)))
                    genitori.append(max(torneo, key=lambda fi: fi[0])[1])
                figlio = incrocia(*genitori)
                if rng.random() < 0.4:
                    figlio = muta(figlio)
                nuova.append(figlio)
            valutati = sorted(((fitness(i), i) for i in nuova),
                              key=lambda fi: -fi[0])
        return valutati[0]

    if par["modalita"] == "mt19937":
        return _genetico_mt19937(estrazioni, par, quanti, pop_n, rng, evolvi,
                                 pyrandom)

    # ---------------- modalità 'indicatori': il GA evolve la combinazione
    finestra = int(par["finestra"]) or len(estrazioni)
    recenti = estrazioni[-finestra:]
    tot = len(recenti)
    pesi = {k: max(0.0, float(par[f"peso_{k}"]))
            for k in ("frequenza", "ritardo", "coppie", "equilibrio")}
    if not any(pesi.values()):
        raise ValueError("Imposta almeno un peso > 0")

    freq, ultimo, coppie = Counter(), {}, Counter()
    for i, (_, nums) in enumerate(recenti):
        ordinati = sorted(nums)
        for j, n in enumerate(ordinati):
            freq[n] += 1
            ultimo[n] = i
            for m in ordinati[j + 1:]:
                coppie[(n, m)] += 1
    rit = {n: tot - 1 - ultimo.get(n, -1) for n in range(1, 91)}
    max_coppia = max(coppie.values()) if coppie else 1

    def percentili(valori):
        ordine = sorted(range(1, 91), key=lambda n: valori.get(n, 0))
        return {n: i / 89.0 for i, n in enumerate(ordine)}
    perc_freq, perc_rit = percentili(freq), percentili(rit)

    def fitness(combo):
        s = pesi["frequenza"] * sum(perc_freq[n] for n in combo) / quanti
        s += pesi["ritardo"] * sum(perc_rit[n] for n in combo) / quanti
        if pesi["coppie"] and quanti > 1:
            cp = [coppie.get((a, b), 0) for i, a in enumerate(sorted(combo))
                  for b in sorted(combo)[i + 1:]]
            s += pesi["coppie"] * (sum(cp) / len(cp)) / max_coppia
        if pesi["equilibrio"]:
            pari = sum(1 for n in combo if n % 2 == 0)
            bassi = sum(1 for n in combo if n <= 45)
            decine = len({(n - 1) // 10 for n in combo})
            eq = (1 - (abs(pari - quanti / 2) + abs(bassi - quanti / 2))
                  / quanti) * 0.7 + (decine / quanti) * 0.3
            s += pesi["equilibrio"] * eq
        return s / sum(pesi.values())

    def incrocia(p1, p2):
        pool = list(set(p1) | set(p2))
        return tuple(sorted(rng.sample(pool, quanti)))

    def muta(combo):
        resto = [n for n in range(1, 91) if n not in combo]
        nuovo = list(combo)
        nuovo[rng.randrange(quanti)] = rng.choice(resto)
        return tuple(sorted(set(nuovo)))[:quanti] if len(set(nuovo)) == quanti \
            else tuple(sorted(rng.sample(range(1, 91), quanti)))

    popolazione = [tuple(sorted(rng.sample(range(1, 91), quanti)))
                   for _ in range(pop_n)]
    migliore_fit, migliore = evolvi(popolazione, fitness, incrocia, muta)
    scelti = list(migliore)

    dettagli = (
        f"GA su combinazioni ({pop_n} individui × {gen_n} generazioni, "
        f"finestra {tot}). Pesi: "
        + ", ".join(f"{k}={v:g}" for k, v in pesi.items())
        + f". Fitness della combinazione vincente: {migliore_fit:.3f} "
        f"(media pesata dei percentili 0-1). Numeri: "
        + ", ".join(f"{n} (freq.{freq[n]}, rit.{rit[n]})"
                    for n in sorted(scelti)))
    return {"numeri": scelti, "dettagli": dettagli}


def _genetico_mt19937(estrazioni, par, quanti, pop_n, rng, evolvi, pyrandom):
    """Porting di gnh1201/lotto_genetic_algorithm con validazione onesta."""
    k = len(estrazioni[-1][1])
    addestr = max(20, min(int(par["addestramento"]), 500))
    valida = max(5, min(int(par["validazione"]), 200))
    if len(estrazioni) < addestr + valida + 10:
        raise ValueError(f"Servono almeno {addestr + valida + 10} estrazioni")

    base = len(estrazioni) - addestr - valida
    train = [(base + i, set(nums))
             for i, (_, nums) in enumerate(estrazioni[base:base + addestr])]
    val = [(base + addestr + i, set(nums))
           for i, (_, nums) in enumerate(estrazioni[-valida:])]

    def genera(x, parametri, n_numeri):
        a, b, c, e = parametri
        numeri, prev, riserva = [], 0, n_numeri + 3
        for pos in range(1, n_numeri + 1):
            num = 1 + int(90 * pyrandom.Random(
                (x * a + pos * b + c * prev + e) & 0x7FFFFFFFFFFF).random())
            while num in numeri:
                num = 1 + int(90 * pyrandom.Random(
                    (x * a + riserva * b + c * prev + e)
                    & 0x7FFFFFFFFFFF).random())
                riserva += 1
            numeri.append(num)
            prev = num
        return numeri

    def fitness(parametri):
        return sum(len(set(genera(x, parametri, k)) & reale)
                   for x, reale in train)

    def incrocia(p1, p2):
        return tuple(p1[i] if rng.random() < 0.5 else p2[i] for i in range(4))

    def muta(parametri):
        nuovo = list(parametri)
        nuovo[rng.randrange(4)] = rng.randrange(10001)
        return tuple(nuovo)

    popolazione = [tuple(rng.randrange(10001) for _ in range(4))
                   for _ in range(pop_n)]
    fit_train, vincente = evolvi(popolazione, fitness, incrocia, muta)

    punti_val = sum(len(set(genera(x, vincente, k)) & reale)
                    for x, reale in val)
    attesa_train = addestr * k * k / 90.0
    attesa_val = valida * k * k / 90.0
    scelti = genera(len(estrazioni), vincente, quanti)

    dettagli = (
        f"GA sui parametri del generatore MT19937 (a,b,c,e = {vincente}). "
        f"Punti in ADDESTRAMENTO ({addestr} estrazioni): {fit_train} contro "
        f"{attesa_train:.0f} attesi dal caso — qui il GA può solo "
        f"sovradattarsi. Punti in VALIDAZIONE su {valida} estrazioni mai "
        f"viste: {punti_val} contro {attesa_val:.0f} attesi — questo è il "
        f"rendimento reale del metodo. Proposta per la prossima estrazione: "
        + ", ".join(map(str, sorted(scelti))))
    return {"numeri": scelti, "dettagli": dettagli}


@algoritmo(
    "caso-vero", "Caso vero (Random.org)",
    "La scelta filosoficamente più pulita: numeri da casualità FISICA "
    "(rumore atmosferico, via random.org), non da un algoritmo. Ha la "
    "stessa identica probabilità di qualsiasi analisi, ma è priva di ogni "
    "bias umano e di ogni pattern: il complemento perfetto della giocata "
    "impopolare. Se random.org non è raggiungibile, ripiega "
    "sull'entropia del sistema operativo (SystemRandom) dichiarandolo.",
    parametri={"quanti": 6},
    in_meta=False,
)
def _caso_vero(estrazioni, par):
    import urllib.request

    quanti = max(1, min(int(par["quanti"]), 10))
    fonte = "random.org (rumore atmosferico)"
    try:
        req = urllib.request.Request(
            "https://www.random.org/sequences/?min=1&max=90&col=1"
            "&format=plain&rnd=new",
            headers={"User-Agent": "lotto-manager (uso personale)"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            permutazione = [int(r) for r in
                            resp.read().decode().split() if r.strip()]
        if len(permutazione) != 90 or set(permutazione) != set(range(1, 91)):
            raise ValueError("risposta inattesa")
        scelti = permutazione[:quanti]
    except Exception:
        import random as pyrandom
        sr = pyrandom.SystemRandom()
        scelti = sr.sample(range(1, 91), quanti)
        fonte = ("entropia del sistema operativo (SystemRandom) — "
                 "random.org non raggiungibile")

    dettagli = (
        f"Permutazione casuale dei 90 numeri, fonte: {fonte}. Nessuna "
        "analisi, nessun pattern, nessun bias: probabilità identica a "
        "qualsiasi altra combinazione. Ogni esecuzione dà numeri diversi.")
    return {"numeri": scelti, "dettagli": dettagli}


@algoritmo(
    "eco-storica", "L'eco della storia (analoghi)",
    "Il metodo degli analoghi, preso in prestito dalla meteorologia di "
    "Lorenz: la storia non si ripete, ma a volte fa rima. Cerca nello "
    "storico i momenti più simili alle ultime 'finestra_ricordo' estrazioni "
    "(somiglianza = numeri in comune, con mezzo punto per quelli a distanza "
    "ciclometrica entro 'tolleranza'), prende le 'analoghi' rime migliori e "
    "propone ciò che uscì subito dopo, pesato per la somiglianza. Il track "
    "record sulle ultime 'prove' estrazioni misura onestamente se l'eco "
    "esiste davvero.",
    parametri={"finestra_ricordo": 3, "analoghi": 20, "tolleranza": 1,
               "prove": 30, "quanti": 5},
)
def _eco_storica(estrazioni, par):
    if len(estrazioni) < 200:
        raise ValueError("Servono almeno 200 estrazioni")
    w = max(1, min(int(par["finestra_ricordo"]), 10))
    K = max(1, min(int(par["analoghi"]), 200))
    toll = max(0, min(int(par["tolleranza"]), 5))
    prove = max(0, min(int(par["prove"]), 200))
    quanti = max(1, min(int(par["quanti"]), 10))

    T = len(estrazioni)
    M = np.zeros((T, 90))
    for t, (_, nums) in enumerate(estrazioni):
        for n in nums:
            M[t, n - 1] = 1.0
    # presenza allargata alla tolleranza ciclometrica (cerchio 1-90)
    Mv = M.copy()
    for d in range(1, toll + 1):
        Mv += np.roll(M, d, axis=1) + np.roll(M, -d, axis=1)
    Mv = np.minimum(Mv, 1.0)

    def previsione(q):
        """Analoghi del 'presente' che termina all'indice q (solo passato)."""
        S = np.zeros(T)
        for o in range(w):
            esatti = M @ M[q - o]
            vicini = np.clip(Mv @ M[q - o] - esatti, 0.0, None)
            contrib = esatti + 0.5 * vicini
            S[w - 1:T] += contrib[w - 1 - o:T - o]
        valide = np.zeros(T, dtype=bool)
        valide[w - 1:q] = True          # finestre interamente nel passato
        S = np.where(valide, S, -1.0)
        migliori = np.argsort(-S)[:K]
        punteggio = np.zeros(90)
        for t in migliori:
            punteggio += S[t] * M[t + 1]  # ciò che uscì subito dopo la rima
        ordine = np.argsort(-punteggio)[:quanti]
        return [int(i + 1) for i in ordine], migliori, S

    # track record onesto sulle ultime 'prove' estrazioni
    k = len(estrazioni[-1][1])
    punti, attesa = 0, 0.0
    for q in range(max(w, T - 1 - prove), T - 1):
        previsti, _, _ = previsione(q)
        punti += len(set(previsti) & set(estrazioni[q + 1][1]))
        attesa += quanti * k / 90.0

    scelti, migliori, S = previsione(T - 1)
    rima = int(migliori[0])
    dettagli = (
        f"{K} analoghi su {T} estrazioni (finestra di ricordo {w}, "
        f"tolleranza ±{toll}). La rima migliore col presente: il "
        f"{estrazioni[rima][0]} (somiglianza {S[rima]:.1f}) — l'estrazione "
        f"successiva fu {', '.join(map(str, sorted(estrazioni[rima + 1][1])))}."
        + (f" Track record su {prove} estrazioni: {punti} punti contro "
           f"{attesa:.1f} attesi dal caso." if prove else ""))
    return {"numeri": scelti, "dettagli": dettagli}
