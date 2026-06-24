"""Analisi Generative — ricostruzione sequenziale dello storico.

La teoria dell'operatore: un generatore casuale estrae cinquine finché non
trova quella successiva dello storico, registrando QUANTI tentativi sono
serviti per ogni transizione. Se la serie dei tentativi formasse una forma
d'onda riconoscibile, si potrebbe prevedere il conteggio successivo e quindi
la prossima estrazione. Questo modulo esegue l'esperimento per davvero
(GPU o CPU), espone l'avanzamento in tempo reale e analizza la serie:
distribuzione contro la geometrica teorica, autocorrelazione, spettro.
"""
import json
import math
import time
from datetime import datetime

import numpy as np

from . import db, stats

LOTTO_GPU = 2_000_000     # cinquine generate per lotto su GPU
LOTTO_CPU = 200_000       # su CPU (numpy)


def _dispositivo(scelta):
    if scelta != "cpu":
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda", f"GPU ({torch.cuda.get_device_name(0)})"
        except ImportError:
            pass
    return "cpu", "CPU (numpy)"


def ricostruisci(con, gioco, ruota, parametri, stato, stop):
    """Esegue la ricostruzione sequenziale. `stato` è il dict condiviso per
    il monitor live; `stop` un threading.Event per l'interruzione."""
    transizioni = max(2, min(int(parametri.get("transizioni", 60)), 500))
    seme = int(parametri.get("seme", 1234))
    disp, nome_disp = _dispositivo(str(parametri.get("dispositivo", "auto")))
    batch = LOTTO_GPU if disp == "cuda" else LOTTO_CPU

    estr = list(reversed(stats._estrazioni(con, gioco, ruota)))
    if len(estr) < transizioni + 1:
        raise ValueError(f"Servono almeno {transizioni + 1} estrazioni")
    finestra = estr[-(transizioni + 1):]
    k = len(finestra[-1][1])
    teorica = math.comb(90, k)
    chiave = f"generativa_{gioco}_{ruota or 'SE'}"
    avviata = datetime.now().strftime("%Y-%m-%d %H:%M")

    def interrotto():
        return stop is not None and stop.is_set()

    if disp == "cuda":
        import torch
        dev = torch.device("cuda")

        def conta_tentativi(numeri, seme_locale):
            gen = torch.Generator(device="cuda").manual_seed(seme_locale)
            bersaglio = torch.tensor(sorted(numeri), device=dev) - 1
            fatti = 0
            while True:
                if interrotto():
                    raise InterruptedError("fermata dall'utente")
                r = torch.rand(batch, 90, device=dev, generator=gen)
                c = r.topk(k, dim=1).indices.sort(dim=1).values
                colpi = (c == bersaglio).all(dim=1).nonzero()
                if len(colpi):
                    return fatti + colpi[0].item() + 1
                fatti += batch
                stato["tentativi_correnti"] = fatti

        def cinquina_al_tentativo(n_tentativo, seme_locale):
            gen = torch.Generator(device="cuda").manual_seed(seme_locale)
            fatti = 0
            while True:
                r = torch.rand(batch, 90, device=dev, generator=gen)
                if fatti + batch >= n_tentativo:
                    c = r[n_tentativo - fatti - 1].topk(k).indices.sort().values
                    return sorted(int(x + 1) for x in c)
                fatti += batch
    else:
        def conta_tentativi(numeri, seme_locale):
            rng = np.random.default_rng(seme_locale)
            bersaglio = np.array(sorted(numeri)) - 1
            fatti = 0
            while True:
                if interrotto():
                    raise InterruptedError("fermata dall'utente")
                r = rng.random((batch, 90))
                idx = np.sort(np.argpartition(r, 90 - k, axis=1)[:, 90 - k:],
                              axis=1)
                colpi = np.nonzero((idx == bersaglio).all(axis=1))[0]
                if len(colpi):
                    return fatti + int(colpi[0]) + 1
                fatti += batch
                stato["tentativi_correnti"] = fatti

        def cinquina_al_tentativo(n_tentativo, seme_locale):
            rng = np.random.default_rng(seme_locale)
            fatti = 0
            while True:
                r = rng.random((batch, 90))
                if fatti + batch >= n_tentativo:
                    riga = r[n_tentativo - fatti - 1]
                    idx = np.sort(np.argpartition(riga, 90 - k)[90 - k:])
                    return sorted(int(x + 1) for x in idx)
                fatti += batch

    def salva(stato_lavoro, analisi=None, previsione=None):
        dati = {
            "gioco": gioco, "ruota": ruota, "avviata": avviata,
            "aggiornata": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "stato": stato_lavoro, "dispositivo": nome_disp,
            "parametri": {"transizioni": transizioni, "seme": seme},
            "teorica": teorica, "k": k,
            "partenza": {"data": finestra[0][0],
                         "numeri": sorted(finestra[0][1])},
            "righe": stato["righe"],
            "analisi": analisi, "previsione": previsione,
        }
        db.set_meta(con, chiave, json.dumps(dati))
        con.commit()
        return dati

    stato.update(fase="preparazione", transizione=0, totale=transizioni,
                 tentativi_correnti=0, bersaglio="", righe=[])
    serie = []
    for i in range(transizioni):
        data_t, numeri_t = finestra[i + 1]
        stato.update(transizione=i + 1, tentativi_correnti=0,
                     fase="ricostruzione",
                     bersaglio=f"{data_t}: "
                               f"{', '.join(map(str, sorted(numeri_t)))}")
        t0 = time.time()
        try:
            tentativi = conta_tentativi(numeri_t, seme + i)
        except InterruptedError:
            salva("fermata")
            raise
        serie.append(tentativi)
        stato["righe"].append({
            "n": i + 1, "data": data_t, "numeri": sorted(numeri_t),
            "tentativi": tentativi, "secondi": round(time.time() - t0, 1)})
        salva("in corso")

    stato["fase"] = "analisi della forma d'onda"
    analisi = _analisi_onda(np.array(serie, dtype=float), teorica)

    stato["fase"] = "previsione dal modello d'onda"
    previsione = None
    try:
        atteso = analisi["prossimo_previsto"]
        numeri_previsti = cinquina_al_tentativo(atteso, seme + transizioni)
        previsione = {
            "tentativo_previsto": atteso,
            "numeri": numeri_previsti,
            "nota": ("Cinquina che il generatore produce esattamente al "
                     "tentativo previsto dal modello d'onda (AR1). Se il "
                     "verdetto dice che l'onda non esiste, questa proposta "
                     "equivale a una scelta casuale uniforme."),
        }
    except Exception:
        pass

    return salva("completata", analisi, previsione)


def _analisi_onda(serie, teorica):
    """Tutte le analisi del caso sulla serie dei tentativi."""
    n = len(serie)
    media, mediana, dev = float(serie.mean()), float(np.median(serie)), \
        float(serie.std())
    out = {
        "campioni": n,
        "media": round(media), "mediana": round(mediana),
        "dev_std": round(dev), "teorica": int(teorica),
        "rapporto_media": round(media / teorica, 2),
        "rapporto_dev": round(dev / teorica, 2),
    }

    # autocorrelazione (una vera onda la mostra; il rumore no)
    z = (serie - media) / (dev or 1.0)
    max_lag = max(1, min(8, n // 4))
    acf = [round(float(np.corrcoef(z[:-l], z[l:])[0, 1]), 3)
           for l in range(1, max_lag + 1)]
    out["acf"] = acf
    out["soglia_acf"] = round(2.0 / math.sqrt(n), 3)

    # spettro (piattezza ~1 = rumore bianco; un'onda concentra la potenza)
    spettro = np.abs(np.fft.rfft(z)) ** 2
    spettro = spettro[1:]  # senza la componente continua
    if len(spettro):
        piatto = float(np.exp(np.mean(np.log(spettro + 1e-12)))
                       / (spettro.mean() + 1e-12))
        out["piattezza"] = round(piatto, 2)
        out["spettro"] = [round(float(v), 2) for v in
                          (spettro / spettro.max())]
    else:
        out["piattezza"], out["spettro"] = 1.0, []

    # distribuzione: i tentativi/teorica dovrebbero seguire una geometrica
    # (≈ esponenziale di media 1)
    rapporti = serie / teorica
    bordi = [0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 100.0]
    etichette = ["0-0.25", "0.25-0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", ">3"]
    osservati = [int(((rapporti >= a) & (rapporti < b)).sum())
                 for a, b in zip(bordi, bordi[1:])]
    attesi = [round(n * (math.exp(-a) - math.exp(-b)), 1)
              for a, b in zip(bordi, bordi[1:])]
    out["istogramma"] = {"etichette": etichette, "osservati": osservati,
                         "attesi": attesi}

    # verdetto onesto
    significativi = sum(1 for a in acf if abs(a) > out["soglia_acf"])
    onda = significativi >= 2 or out["piattezza"] < 0.45
    out["verdetto"] = (
        "POSSIBILE STRUTTURA: più autocorrelazioni oltre la soglia o spettro "
        "concentrato. Prima di crederci: ripeti con un seme diverso — se la "
        "struttura resta identica è dei dati, se cambia era del generatore."
        if onda else
        "NESSUNA FORMA D'ONDA: autocorrelazioni dentro la banda del rumore "
        "(±{s}) e spettro piatto ({p}). La serie è compatibile con una "
        "geometrica senza memoria: ogni transizione riparte da zero, come "
        "previsto dalla teoria dei processi senza memoria."
        .format(s=out["soglia_acf"], p=out["piattezza"]))
    out["onda"] = bool(onda)

    # modello d'onda AR(1) per la previsione del prossimo conteggio
    acf1 = acf[0] if acf else 0.0
    prossimo = media + acf1 * (serie[-1] - media)
    out["prossimo_previsto"] = max(1, int(round(prossimo)))
    out["acf1"] = acf1
    return out


def salvata(con, gioco, ruota=None):
    raw = db.get_meta(con, f"generativa_{gioco}_{ruota or 'SE'}")
    return json.loads(raw) if raw else None
