"""Interfaccia web — FastAPI + Jinja2."""
import json
import threading
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from . import db, fetch, generative, ml, net, predict, sistemi, stats

app = FastAPI(title="Lotto & SuperEnalotto")
templates = Jinja2Templates(directory=Path(__file__).resolve().parent / "templates")
templates.env.globals.update(
    RUOTE=db.RUOTE, NOMI_RUOTE=db.NOMI_RUOTE, STRATEGIE=predict.STRATEGIE,
    NOMI_MODELLI=ml.NOMI_MODELLI)


def render(request, nome, **ctx):
    return templates.TemplateResponse(request, nome, ctx)


@app.get("/")
def home(request: Request):
    con = db.connect()
    try:
        riepilogo = stats.riepilogo(con)
        data_lotto, ruote = stats.ultima_estrazione_lotto(con)
        se = stats.ultima_estrazione_superenalotto(con)
        ultimo_agg = db.get_meta(con, "ultimo_aggiornamento")
        esito_agg = db.get_meta(con, "esito_aggiornamento")
        return render(request, "index.html", riepilogo=riepilogo,
                      data_lotto=data_lotto, ruote=ruote, se=se,
                      ultimo_agg=ultimo_agg,
                      esito_agg=json.loads(esito_agg) if esito_agg else None)
    finally:
        con.close()


@app.post("/aggiorna")
def aggiorna():
    con = db.connect()
    try:
        esito = fetch.aggiorna_tutto(con)
        db.set_meta(con, "esito_aggiornamento", json.dumps(esito))
        con.commit()
    finally:
        con.close()
    return RedirectResponse("/", status_code=303)


@app.get("/lotto")
def pagina_lotto(request: Request, ruota: str = "RM", finestra: int = 0,
                 spia: int = 0):
    con = db.connect()
    try:
        fin = finestra or None
        freq = stats.frequenze(con, "lotto", ruota, fin)
        rit = stats.ritardi(con, "lotto", ruota)
        coppie = stats.combinazioni_frequenti(con, "lotto", ruota, fin, k=2)
        terzine = stats.combinazioni_frequenti(con, "lotto", ruota, fin, k=3)
        distr = stats.distribuzione(con, "lotto", ruota, fin)
        dati_spia = stats.numeri_spia(con, ruota, spia) if spia else None
        return render(request, "lotto.html", ruota=ruota, finestra=finestra,
                      freq=freq, rit=rit, coppie=coppie, terzine=terzine,
                      distr=distr, spia=spia, dati_spia=dati_spia,
                      top_freq=sorted(freq, key=lambda n: -freq[n])[:15],
                      top_rit=sorted(rit, key=lambda n: -rit[n]["attuale"])[:15])
    finally:
        con.close()


@app.get("/superenalotto")
def pagina_superenalotto(request: Request, finestra: int = 0):
    con = db.connect()
    try:
        fin = finestra or None
        freq = stats.frequenze(con, "superenalotto", None, fin)
        rit = stats.ritardi(con, "superenalotto")
        coppie = stats.combinazioni_frequenti(con, "superenalotto", None, fin, k=2)
        distr = stats.distribuzione(con, "superenalotto", None, fin)
        return render(request, "superenalotto.html", finestra=finestra,
                      freq=freq, rit=rit, coppie=coppie, distr=distr,
                      top_freq=sorted(freq, key=lambda n: -freq[n])[:15],
                      top_rit=sorted(rit, key=lambda n: -rit[n]["attuale"])[:15])
    finally:
        con.close()


@app.get("/previsioni")
def previsioni(request: Request, gioco: str = "lotto", strategia: str = "ritardatari",
               ruota: str = "RM", quanti: int = 5, genera: int = 0):
    proposta = errore = None
    if genera:
        con = db.connect()
        try:
            proposta = predict.genera(con, gioco, strategia, ruota, quanti)
        except ValueError as exc:
            errore = str(exc)
        finally:
            con.close()
    return render(request, "previsioni.html", gioco=gioco, strategia=strategia,
                  ruota=ruota, quanti=quanti, proposta=proposta, errore=errore)


@app.post("/giocate")
def salva_giocata(request: Request, gioco: str = Form(...), ruota: str = Form(""),
                  numeri: str = Form(...), strategia: str = Form(""),
                  note: str = Form("")):
    con = db.connect()
    try:
        cur = con.execute(
            "INSERT INTO giocate (creata, gioco, ruota, numeri, strategia, note) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M"), gioco,
             ruota or None, numeri, strategia, note))
        con.commit()
        nuovo_id = cur.lastrowid
    finally:
        con.close()
    if "application/json" in request.headers.get("accept", ""):
        return {"ok": True, "id": nuovo_id}
    return RedirectResponse("/giocate", status_code=303)


@app.get("/giocate")
def lista_giocate(request: Request):
    con = db.connect()
    try:
        righe = con.execute("SELECT * FROM giocate ORDER BY id DESC").fetchall()
        giocate = [dict(r, esito=json.loads(r["esito"]) if r["esito"] else None)
                   for r in righe]
        return render(request, "giocate.html", giocate=giocate)
    finally:
        con.close()


@app.post("/giocate/verifica")
def verifica():
    con = db.connect()
    try:
        predict.verifica_giocate(con)
    finally:
        con.close()
    return RedirectResponse("/giocate", status_code=303)


@app.get("/ml")
def pagina_ml(request: Request, gioco: str = "lotto", ruota: str = "RM",
              errore: str = ""):
    con = db.connect()
    try:
        r = ruota if gioco == "lotto" else None
        risultati = ml.risultati_salvati(con, gioco, r)
        generatore = ml.generatore_salvato(con, gioco, r)
        bias = ml.bias_salvato(con, gioco, r)
        cfg = ml.carica_config(con)
        master = ml.master_salvato(con, gioco, r)
        return render(request, "ml.html", gioco=gioco, ruota=ruota,
                      risultati=risultati, generatore=generatore,
                      bias=bias, cfg=cfg, errore=errore,
                      gen_stato=GENERAZIONE, master=master,
                      master_stato=MASTER, gpu=ml.stato_gpu())
    finally:
        con.close()


@app.post("/ml/bias")
def calcola_bias(gioco: str = Form("lotto"), ruota: str = Form("RM")):
    con = db.connect()
    errore = ""
    try:
        ml.bias_e_salva(con, gioco, ruota if gioco == "lotto" else None)
    except ValueError as exc:
        errore = f"&errore={exc}"
    finally:
        con.close()
    return RedirectResponse(f"/ml?gioco={gioco}&ruota={ruota}{errore}",
                            status_code=303)


# stato condiviso dei lavori neurali (uno alla volta: la GPU è una)
GENERAZIONE = {"attivo": False, "testo": "", "esito": "",
               "gioco": "", "ruota": "", "stop": threading.Event()}
MASTER = {"attivo": False, "testo": "", "esito": "",
          "gioco": "", "ruota": "", "stop": threading.Event()}
GENERATIVA = {"attivo": False, "fase": "", "esito": "", "gioco": "",
              "ruota": "", "transizione": 0, "totale": 0,
              "tentativi_correnti": 0, "bersaglio": "", "righe": [],
              "stop": threading.Event()}


def _lavoro_neurale_attivo():
    return GENERAZIONE["attivo"] or MASTER["attivo"] or GENERATIVA["attivo"]


@app.post("/ml/genera")
def genera_neurale(gioco: str = Form("lotto"), ruota: str = Form("RM")):
    if _lavoro_neurale_attivo():
        return RedirectResponse(
            f"/ml?gioco={gioco}&ruota={ruota}&errore=È già in corso un "
            "lavoro neurale: fermalo o attendi che finisca", status_code=303)
    GENERAZIONE.update(attivo=True, testo="avvio...", esito="",
                       gioco=gioco, ruota=ruota)
    GENERAZIONE["stop"].clear()

    def lavoro():
        con = db.connect()
        try:
            ml.generatore_e_salva(
                con, gioco, ruota if gioco == "lotto" else None,
                progresso=lambda msg: GENERAZIONE.__setitem__("testo", msg),
                stop=GENERAZIONE["stop"])
            GENERAZIONE["esito"] = "completata"
        except InterruptedError:
            GENERAZIONE["esito"] = "fermata dall'utente (nessun salvataggio)"
        except Exception as exc:
            GENERAZIONE["esito"] = f"errore: {exc}"
            import traceback
            traceback.print_exc()  # visibile nel log del server
        finally:
            con.close()
            GENERAZIONE["attivo"] = False

    threading.Thread(target=lavoro, daemon=True).start()
    return RedirectResponse(f"/ml?gioco={gioco}&ruota={ruota}",
                            status_code=303)


@app.post("/ml/genera/ferma")
def ferma_generazione(gioco: str = Form("lotto"), ruota: str = Form("RM")):
    GENERAZIONE["stop"].set()
    return RedirectResponse(f"/ml?gioco={gioco}&ruota={ruota}",
                            status_code=303)


@app.get("/api/gpu")
def api_gpu():
    return ml.stato_gpu() or {}


@app.post("/ml/master")
def avvia_master(gioco: str = Form("lotto"), ruota: str = Form("RM")):
    if _lavoro_neurale_attivo():
        return RedirectResponse(
            f"/ml?gioco={gioco}&ruota={ruota}&errore=È già in corso un "
            "lavoro neurale: fermalo o attendi che finisca", status_code=303)
    MASTER.update(attivo=True, testo="avvio...", esito="",
                  gioco=gioco, ruota=ruota)
    MASTER["stop"].clear()

    def lavoro():
        con = db.connect()
        try:
            ml.master_neurale(
                con, gioco, ruota if gioco == "lotto" else None,
                progresso=lambda msg: MASTER.__setitem__("testo", msg),
                stop=MASTER["stop"])
            MASTER["esito"] = "completato"
        except InterruptedError:
            MASTER["esito"] = ("fermato dall'utente — la matrice parziale "
                               "è salvata")
        except Exception as exc:
            MASTER["esito"] = f"errore: {exc}"
            import traceback
            traceback.print_exc()
        finally:
            con.close()
            MASTER["attivo"] = False

    threading.Thread(target=lavoro, daemon=True).start()
    return RedirectResponse(f"/ml?gioco={gioco}&ruota={ruota}",
                            status_code=303)


@app.post("/ml/master/ferma")
def ferma_master(gioco: str = Form("lotto"), ruota: str = Form("RM")):
    MASTER["stop"].set()
    return RedirectResponse(f"/ml?gioco={gioco}&ruota={ruota}",
                            status_code=303)


@app.post("/ml/calcola")
def calcola_ml(gioco: str = Form("lotto"), ruota: str = Form("RM")):
    con = db.connect()
    errore = ""
    try:
        ml.analizza_e_salva(con, gioco, ruota if gioco == "lotto" else None)
    except ValueError as exc:
        errore = f"&errore={exc}"
    finally:
        con.close()
    return RedirectResponse(f"/ml?gioco={gioco}&ruota={ruota}{errore}",
                            status_code=303)


@app.get("/net")
def pagina_net(request: Request, gioco: str = "lotto", ruota: str = "RM"):
    attivi = request.query_params.getlist("attivi")
    risultati = convergenza = None
    if attivi:
        con = db.connect()
        try:
            risultati, convergenza = net.esegui(
                con, gioco, ruota, attivi, dict(request.query_params))
        finally:
            con.close()
    return render(request, "net.html", gioco=gioco, ruota=ruota,
                  algoritmi=net.ALGORITMI, attivi=attivi,
                  parametri=dict(request.query_params),
                  risultati=risultati, convergenza=convergenza)


@app.get("/generative")
def pagina_generative(request: Request, gioco: str = "lotto",
                      ruota: str = "RM", errore: str = ""):
    con = db.connect()
    try:
        risultati = generative.salvata(con, gioco,
                                       ruota if gioco == "lotto" else None)
        return render(request, "generative.html", gioco=gioco, ruota=ruota,
                      risultati=risultati, errore=errore,
                      stato=GENERATIVA, gpu=ml.stato_gpu())
    finally:
        con.close()


@app.post("/generative/avvia")
def avvia_generativa(gioco: str = Form("lotto"), ruota: str = Form("RM"),
                     transizioni: int = Form(60), seme: int = Form(1234),
                     dispositivo: str = Form("auto")):
    if _lavoro_neurale_attivo():
        return RedirectResponse(
            f"/generative?gioco={gioco}&ruota={ruota}&errore=È già in corso "
            "un lavoro su GPU/CPU: fermalo o attendi", status_code=303)
    GENERATIVA.update(attivo=True, fase="avvio...", esito="", gioco=gioco,
                      ruota=ruota, transizione=0, totale=transizioni,
                      tentativi_correnti=0, bersaglio="", righe=[])
    GENERATIVA["stop"].clear()
    parametri = {"transizioni": transizioni, "seme": seme,
                 "dispositivo": dispositivo}

    def lavoro():
        con = db.connect()
        try:
            generative.ricostruisci(
                con, gioco, ruota if gioco == "lotto" else None,
                parametri, GENERATIVA, GENERATIVA["stop"])
            GENERATIVA["esito"] = "completata"
        except InterruptedError:
            GENERATIVA["esito"] = ("fermata dall'utente — le transizioni "
                                   "completate sono salvate")
        except Exception as exc:
            GENERATIVA["esito"] = f"errore: {exc}"
            import traceback
            traceback.print_exc()
        finally:
            con.close()
            GENERATIVA["attivo"] = False

    threading.Thread(target=lavoro, daemon=True).start()
    return RedirectResponse(f"/generative?gioco={gioco}&ruota={ruota}",
                            status_code=303)


@app.post("/generative/ferma")
def ferma_generativa(gioco: str = Form("lotto"), ruota: str = Form("RM")):
    GENERATIVA["stop"].set()
    return RedirectResponse(f"/generative?gioco={gioco}&ruota={ruota}",
                            status_code=303)


@app.get("/api/generativa")
def api_generativa():
    return {
        "attivo": GENERATIVA["attivo"], "fase": GENERATIVA["fase"],
        "esito": GENERATIVA["esito"], "gioco": GENERATIVA["gioco"],
        "ruota": GENERATIVA["ruota"],
        "transizione": GENERATIVA["transizione"],
        "totale": GENERATIVA["totale"],
        "tentativi_correnti": GENERATIVA["tentativi_correnti"],
        "bersaglio": GENERATIVA["bersaglio"],
        "righe": GENERATIVA["righe"],
        "gpu": ml.stato_gpu(),
    }


@app.get("/sistemi")
def pagina_sistemi(request: Request, gioco: str = "lotto", ruota: str = "RM",
                   numeri: str = "", colonna: int = 5, garanzia: int = 3):
    sistema = errore = None
    if numeri.strip():
        try:
            lista = [int(x) for x in numeri.replace(" ", "").split(",") if x]
            sistema = sistemi.genera_sistema(lista, colonna, garanzia)
        except ValueError as exc:
            errore = str(exc)
    return render(request, "sistemi.html", gioco=gioco, ruota=ruota,
                  numeri=numeri, colonna=colonna, garanzia=garanzia,
                  sistema=sistema, errore=errore, SORTI=sistemi.SORTI)


@app.get("/setup")
def pagina_setup(request: Request, salvata: int = 0):
    con = db.connect()
    try:
        cfg = ml.carica_config(con)
        return render(request, "setup.html", cfg=cfg, salvata=salvata)
    finally:
        con.close()


@app.post("/setup")
def salva_setup(modello: str = Form(...), finestra_addestramento: int = Form(...),
                backtest_estrazioni: int = Form(...),
                finestre_frequenza: str = Form(...),
                numeri_giocata: int = Form(...), cluster_k: int = Form(...),
                gen_lag: int = Form(10), gen_reti: int = Form(5),
                gen_simulazioni: int = Form(1000),
                gen_neuroni: str = Form("64,32"),
                gen_rumore: float = Form(0.1),
                gen_dispositivo: str = Form("auto"),
                gen_epoche: int = Form(300),
                gen_pazienza: int = Form(15),
                master_pausa: int = Form(10)):
    finestre = sorted({int(f) for f in finestre_frequenza.replace(" ", "").split(",")
                       if f.strip().isdigit() and int(f) > 0}) or \
        ml.DEFAULT_CONFIG["finestre_frequenza"]
    cfg = {
        "modello": modello if modello in ml.NOMI_MODELLI else "logistic",
        "finestra_addestramento": max(300, min(finestra_addestramento, 6000)),
        "backtest_estrazioni": max(30, min(backtest_estrazioni, 1000)),
        "finestre_frequenza": finestre[:6],
        "numeri_giocata": max(1, min(numeri_giocata, 10)),
        "cluster_k": max(2, min(cluster_k, 15)),
        "gen_lag": max(2, min(gen_lag, 200)),
        "gen_reti": max(1, min(gen_reti, 100)),
        "gen_simulazioni": max(100, min(gen_simulazioni, 200000)),
        "gen_neuroni": gen_neuroni.strip() or "64,32",
        "gen_rumore": max(0.0, min(gen_rumore, 2.0)),
        "gen_dispositivo": gen_dispositivo
        if gen_dispositivo in ("auto", "gpu", "cpu") else "auto",
        "gen_epoche": max(10, min(gen_epoche, 5000)),
        "gen_pazienza": max(0, min(gen_pazienza, 500)),
        "master_pausa": max(0, min(master_pausa, 600)),
    }
    con = db.connect()
    try:
        ml.salva_config(con, cfg)
    finally:
        con.close()
    return RedirectResponse("/setup?salvata=1", status_code=303)


@app.post("/giocate/{giocata_id}/elimina")
def elimina_giocata(giocata_id: int):
    con = db.connect()
    try:
        con.execute("DELETE FROM giocate WHERE id = ?", (giocata_id,))
        con.commit()
    finally:
        con.close()
    return RedirectResponse("/giocate", status_code=303)
