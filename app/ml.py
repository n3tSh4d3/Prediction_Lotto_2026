"""Analisi con machine learning.

Il modello impara, per ogni numero, la relazione tra il suo "stato" prima di
un'estrazione (ritardo, frequenze recenti, presenza nell'ultima estrazione...)
e la sua uscita. Il backtest confronta il modello con le strategie classiche e
con l'attesa del puro caso: per un gioco equo l'AUC atteso è ~0.5 e nessuna
strategia dovrebbe battere il caso in modo sistematico — il confronto serve
proprio a misurarlo.
"""
import json
from collections import Counter
from datetime import datetime

import numpy as np
from scipy.stats import chisquare
from sklearn.cluster import KMeans
from sklearn.ensemble import (ExtraTreesClassifier,
                              HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import db, stats

DEFAULT_CONFIG = {
    "modello": "logistic",
    "finestra_addestramento": 1500,   # estrazioni usate per l'addestramento
    "backtest_estrazioni": 100,       # estrazioni finali riservate al backtest
    "finestre_frequenza": [10, 30, 100, 300],
    "numeri_giocata": 5,              # numeri puntati per estrazione nel backtest
    "cluster_k": 6,                   # gruppi nel clustering di co-occorrenza
    # generatore neurale
    "gen_lag": 10,                    # estrazioni precedenti in ingresso alla rete
    "gen_reti": 5,                    # reti nel comitato (semi diversi)
    "gen_simulazioni": 1000,          # estrazioni generate in totale
    "gen_neuroni": "64,32",           # neuroni dei livelli nascosti
    "gen_rumore": 0.1,                # rumore sugli ingressi (frazione della dev. std)
    "gen_dispositivo": "auto",        # auto | gpu (CUDA) | cpu (scikit-learn)
    "gen_epoche": 300,                # epoche massime di addestramento
    "gen_pazienza": 15,               # early stopping (0 = sempre tutte le epoche)
    "master_pausa": 10,               # secondi di riposo GPU tra le generazioni
}

NOMI_MODELLI = {
    "logistic": "Regressione logistica",
    "random_forest": "Random forest",
    "gradient_boosting": "Gradient boosting",
    "extra_trees": "Extra trees",
    "naive_bayes": "Naive Bayes gaussiano",
    "knn": "K-nearest neighbors",
    "rete_neurale": "Rete neurale (MLP 32×16)",
    "rete_neurale_profonda": "Rete neurale profonda (256×128×64, GPU)",
    "ensemble_media": "Ensemble parallelo (media di 4 modelli)",
    "stacking": "Stacking sequenziale (meta-modello sui modelli base)",
}

BASI_COMPOSITE = ["logistic", "gradient_boosting", "naive_bayes", "extra_trees"]


# -------------------------------------------------------------- configurazione

def carica_config(con):
    cfg = dict(DEFAULT_CONFIG)
    salvata = db.get_meta(con, "config_ml")
    if salvata:
        cfg.update(json.loads(salvata))
    return cfg


def salva_config(con, cfg):
    db.set_meta(con, "config_ml", json.dumps(cfg))
    con.commit()


# ------------------------------------------------------------------- dataset

def _matrice_presenze(con, gioco, ruota):
    """Matrice booleana (estrazioni in ordine cronologico × 90 numeri)."""
    estr = list(reversed(stats._estrazioni(con, gioco, ruota)))
    date = [d for d, _ in estr]
    M = np.zeros((len(estr), 90), dtype=np.float64)
    for t, (_, nums) in enumerate(estr):
        for n in nums:
            M[t, n - 1] = 1.0
    return date, M


def _features(M, finestre):
    """Per ogni riga t (0..T compreso): feature calcolate sulle estrazioni < t.
    La riga T descrive lo stato attuale, cioè la prossima estrazione."""
    T = M.shape[0]
    righe = np.arange(T + 1)
    C = np.zeros((T + 1, 90))
    C[1:] = np.cumsum(M, axis=0)

    colonne, nomi = [], []
    freq = {}
    for w in sorted(finestre):
        idx = np.maximum(righe - w, 0)
        freq[w] = (C[righe] - C[idx]) / w
        colonne.append(freq[w])
        nomi.append(f"freq_{w}")

    if len(finestre) >= 2:
        w_min, w_max = min(finestre), max(finestre)
        colonne.append(freq[w_min] - freq[w_max])
        nomi.append(f"trend_{w_min}_vs_{w_max}")

    # ritardo relativo: estrazioni dall'ultima uscita / gap medio del gioco
    R = np.zeros((T + 1, 90))
    indici = np.arange(T)
    for n in range(90):
        ultime = np.where(M[:, n] > 0, indici, -1)
        ultima_prima = np.concatenate(([-1], np.maximum.accumulate(ultime)))
        R[:, n] = righe - ultima_prima - 1
    gap_medio = 90.0 / max(M.sum(axis=1).mean(), 1e-9)
    colonne.append(R / gap_medio)
    nomi.append("ritardo_rel")

    # presenza nell'ultima estrazione
    P = np.zeros((T + 1, 90))
    P[1:] = M
    colonne.append(P)
    nomi.append("in_ultima")

    X = np.stack(colonne, axis=-1)  # (T+1, 90, n_feature)
    return X, nomi, R


def _dispositivo_scelto(cfg):
    """Risoluzione del selettore dispositivo: ('cuda'|'cpu', descrizione)."""
    scelta = str(cfg.get("gen_dispositivo", "auto")).lower()
    if scelta != "cpu":
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda", f"GPU ({torch.cuda.get_device_name(0)})"
        except ImportError:
            pass
        if scelta == "gpu":
            return "cpu", ("CPU (scikit-learn) — GPU richiesta ma CUDA "
                           "non disponibile")
    return "cpu", "CPU (scikit-learn)"


class _ClassificatoreTorch:
    """Classificatore MLP PyTorch su GPU con interfaccia scikit-learn
    (fit/predict_proba). Addestramento a minibatch con early stopping
    sulla coda cronologica del train."""

    def __init__(self, neuroni=(32, 16), dispositivo="cuda", seme=0,
                 epoche=40, lotto=8192):
        self.neuroni, self.dispositivo, self.seme = neuroni, dispositivo, seme
        self.epoche, self.lotto = epoche, lotto

    def fit(self, X, y):
        import torch
        torch.manual_seed(self.seme)
        self.media_ = X.mean(axis=0)
        self.scala_ = X.std(axis=0) + 1e-9
        dev = torch.device(self.dispositivo)
        Xt = torch.tensor((X - self.media_) / self.scala_,
                          dtype=torch.float32, device=dev)
        yt = torch.tensor(y, dtype=torch.float32, device=dev)

        strati, ingresso = [], X.shape[1]
        for h in self.neuroni:
            strati += [torch.nn.Linear(ingresso, h), torch.nn.ReLU()]
            ingresso = h
        strati.append(torch.nn.Linear(ingresso, 1))
        self.rete_ = torch.nn.Sequential(*strati).to(dev)

        ottimizza = torch.optim.Adam(self.rete_.parameters(), lr=1e-3)
        bce = torch.nn.BCEWithLogitsLoss()
        n_val = max(500, len(yt) // 10)
        Xa, ya, Xv, yv = Xt[:-n_val], yt[:-n_val], Xt[-n_val:], yt[-n_val:]
        migliore, pazienza, stato = float("inf"), 0, None
        for _ in range(self.epoche):
            self.rete_.train()
            ordine = torch.randperm(len(ya), device=dev)
            for i in range(0, len(ya), self.lotto):
                blocco = ordine[i:i + self.lotto]
                ottimizza.zero_grad()
                errore = bce(self.rete_(Xa[blocco]).squeeze(1), ya[blocco])
                errore.backward()
                ottimizza.step()
            self.rete_.eval()
            with torch.no_grad():
                val = bce(self.rete_(Xv).squeeze(1), yv).item()
            if val < migliore - 1e-5:
                migliore, pazienza = val, 0
                stato = {k: v.detach().clone()
                         for k, v in self.rete_.state_dict().items()}
            else:
                pazienza += 1
                if pazienza >= 5:
                    break
        if stato:
            self.rete_.load_state_dict(stato)
        self.rete_.eval()
        return self

    def predict_proba(self, X):
        import torch
        Xs = torch.tensor((X - self.media_) / self.scala_,
                          dtype=torch.float32,
                          device=torch.device(self.dispositivo))
        with torch.no_grad():
            p = torch.sigmoid(self.rete_(Xs).squeeze(1)).cpu().numpy()
        return np.column_stack([1.0 - p, p])


def _nuovo_modello(nome, dispositivo="cpu"):
    if nome == "rete_neurale" and dispositivo == "cuda":
        return _ClassificatoreTorch((32, 16), dispositivo)
    if nome == "rete_neurale_profonda":
        if dispositivo == "cuda":
            return _ClassificatoreTorch((256, 128, 64), dispositivo)
        return make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(256, 128, 64), max_iter=100,
            early_stopping=True, random_state=0))
    if nome == "random_forest":
        return RandomForestClassifier(
            n_estimators=80, max_depth=8, n_jobs=-1, random_state=0)
    if nome == "gradient_boosting":
        return HistGradientBoostingClassifier(random_state=0)
    if nome == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=80, max_depth=10, n_jobs=-1, random_state=0)
    if nome == "naive_bayes":
        return GaussianNB()
    if nome == "knn":
        return KNeighborsClassifier(n_neighbors=64, n_jobs=-1)
    if nome == "rete_neurale":
        return make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(32, 16), max_iter=150,
            early_stopping=True, random_state=0))
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=1000))


def _modello_composito(tipo, Xtr, ytr, Xbt, ybt, Xnext):
    """Ensemble parallelo o stacking sequenziale sui modelli base.

    Ritorna (prob_backtest, prob_next, incroci, importanze)."""
    prob_bt_basi, prob_next_basi, auc_basi = {}, {}, {}

    if tipo == "stacking":
        # sequenziale: basi addestrate sul primo 70%, il meta-modello impara
        # a combinare le loro previsioni sul restante 30% (mai visto dalle basi)
        taglio = int(len(ytr) * 0.7)
        meta_feature = []
        for nome in BASI_COMPOSITE:
            m = _nuovo_modello(nome)
            m.fit(Xtr[:taglio], ytr[:taglio])
            meta_feature.append(m.predict_proba(Xtr[taglio:])[:, 1])
            prob_bt_basi[nome] = m.predict_proba(Xbt)[:, 1]
            prob_next_basi[nome] = m.predict_proba(Xnext)[:, 1]
        meta = LogisticRegression(max_iter=1000)
        meta.fit(np.column_stack(meta_feature), ytr[taglio:])
        prob_bt = meta.predict_proba(np.column_stack(
            [prob_bt_basi[n] for n in BASI_COMPOSITE]))[:, 1]
        prob_next = meta.predict_proba(np.column_stack(
            [prob_next_basi[n] for n in BASI_COMPOSITE]))[:, 1]
        coef = np.abs(meta.coef_[0])
        importanze = sorted(
            ((NOMI_MODELLI[n], round(float(c / (coef.sum() or 1)), 3))
             for n, c in zip(BASI_COMPOSITE, coef)), key=lambda x: -x[1])
    else:
        # parallelo: ogni base addestrata su tutto, media delle probabilità
        for nome in BASI_COMPOSITE:
            m = _nuovo_modello(nome)
            m.fit(Xtr, ytr)
            prob_bt_basi[nome] = m.predict_proba(Xbt)[:, 1]
            prob_next_basi[nome] = m.predict_proba(Xnext)[:, 1]
        prob_bt = np.mean([prob_bt_basi[n] for n in BASI_COMPOSITE], axis=0)
        prob_next = np.mean([prob_next_basi[n] for n in BASI_COMPOSITE],
                            axis=0)
        importanze = None

    # controlli incrociati: AUC di ogni base, correlazione tra i punteggi,
    # accordo sulle top-15 proposte
    for nome in BASI_COMPOSITE:
        auc_basi[nome] = round(float(roc_auc_score(ybt, prob_bt_basi[nome])), 4)
    nomi = list(BASI_COMPOSITE)
    matrice_prob = np.array([prob_bt_basi[n] for n in nomi])
    correlazioni = np.corrcoef(matrice_prob)
    top15 = {n: set(np.argsort(-prob_next_basi[n])[:15]) for n in nomi}
    accordo = [[len(top15[a] & top15[b]) for b in nomi] for a in nomi]
    incroci = {
        "basi": [{"nome": NOMI_MODELLI[n], "auc": auc_basi[n]} for n in nomi],
        "nomi": [NOMI_MODELLI[n] for n in nomi],
        "correlazioni": [[round(float(c), 2) for c in riga]
                         for riga in correlazioni],
        "accordo_top15": accordo,
    }
    return prob_bt, prob_next, incroci, importanze


def _importanze(modello, nomi):
    if hasattr(modello, "named_steps"):
        if "logisticregression" not in modello.named_steps:
            return None  # rete neurale: nessuna importanza nativa
        coef = modello.named_steps["logisticregression"].coef_[0]
        valori = np.abs(coef)
    elif hasattr(modello, "feature_importances_"):
        valori = modello.feature_importances_
    else:  # HistGradientBoosting non espone importanze native
        return None
    tot = valori.sum() or 1.0
    return sorted(((nomi[i], round(float(v / tot), 3)) for i, v in enumerate(valori)),
                  key=lambda x: -x[1])


# -------------------------------------------------------------------- analisi

def analizza(con, gioco, ruota, cfg):
    """Esegue l'intera analisi ML e ritorna un dict JSON-serializzabile."""
    date, M = _matrice_presenze(con, gioco, ruota)
    T = M.shape[0]
    if T < 400:
        raise ValueError("Storico insufficiente per l'analisi (servono ≥400 estrazioni)")

    X, nomi_feature, R = _features(M, cfg["finestre_frequenza"])
    n_feat = X.shape[-1]

    B = min(int(cfg["backtest_estrazioni"]), T // 4)
    fine_train = T - B
    inizio_train = max(max(cfg["finestre_frequenza"]),
                       fine_train - int(cfg["finestra_addestramento"]))

    Xtr = X[inizio_train:fine_train].reshape(-1, n_feat)
    ytr = M[inizio_train:fine_train].ravel()
    Xbt = X[fine_train:T].reshape(-1, n_feat)
    ybt = M[fine_train:T].ravel()
    Xnext = X[T].reshape(90, n_feat)

    incroci = None
    dispositivo_usato = "CPU (scikit-learn)"
    if cfg["modello"] in ("ensemble_media", "stacking"):
        prob_bt, prob_next, incroci, importanze = _modello_composito(
            cfg["modello"], Xtr, ytr, Xbt, ybt, Xnext)
        prob = prob_bt.reshape(B, 90)
    else:
        if cfg["modello"] in ("rete_neurale", "rete_neurale_profonda"):
            disp, dispositivo_usato = _dispositivo_scelto(cfg)
        else:
            disp = "cpu"
        modello = _nuovo_modello(cfg["modello"], disp)
        modello.fit(Xtr, ytr)
        prob = modello.predict_proba(Xbt)[:, 1].reshape(B, 90)
        prob_next = modello.predict_proba(Xnext)[:, 1]
        importanze = _importanze(modello, nomi_feature)
    auc = float(roc_auc_score(ybt, prob.ravel()))

    k = max(1, min(int(cfg["numeri_giocata"]), 10))
    estratti_per_estr = int(round(M.sum(axis=1).mean()))

    def conta_punti(punteggi):
        scelte = np.argsort(-punteggi, axis=1)[:, :k]
        return int(sum(M[fine_train + t, scelte[t]].sum() for t in range(B)))

    freq_recente = X[fine_train:T, :, nomi_feature.index(f"freq_{max(cfg['finestre_frequenza'])}")]
    backtest = {
        "estrazioni": B,
        "numeri_puntati": k,
        "attesa_caso": round(B * k * estratti_per_estr / 90.0, 1),
        "punti": {
            "modello_ml": conta_punti(prob),
            "ritardatari": conta_punti(R[fine_train:T]),
            "frequenti": conta_punti(freq_recente),
        },
    }

    # ---- punteggi per la prossima estrazione (riga T = stato attuale)
    ordine = np.argsort(-prob_next)
    punteggi = [{
        "numero": int(i + 1),
        "punteggio": round(float(prob_next[i]), 4),
        "ritardo": int(R[T, i]),
        "freq_recente": int(round(X[T, i, nomi_feature.index(f'freq_{max(cfg.get("finestre_frequenza"))}')]
                                  * max(cfg["finestre_frequenza"]))),
    } for i in ordine[:20]]

    # ---- test di uniformità (chi-quadro su tutto lo storico)
    conteggi = M.sum(axis=0)
    chi2, p_value = chisquare(conteggi)
    uniforme = p_value > 0.05

    # ---- clustering di co-occorrenza
    co = M.T @ M
    np.fill_diagonal(co, 0.0)
    norme = co.sum(axis=1, keepdims=True)
    co_norm = co / np.where(norme == 0, 1, norme)
    km = KMeans(n_clusters=int(cfg["cluster_k"]), n_init=10, random_state=0)
    etichette = km.fit_predict(co_norm)
    clusters = [sorted(int(n + 1) for n in np.where(etichette == c)[0])
                for c in range(int(cfg["cluster_k"]))]

    return {
        "gioco": gioco,
        "ruota": ruota,
        "calcolata": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "config": cfg,
        "periodo": {"dal": date[0], "al": date[-1], "estrazioni": T,
                    "train": [date[inizio_train], date[fine_train - 1]],
                    "backtest": [date[fine_train], date[-1]]},
        "auc": round(auc, 4),
        "dispositivo": dispositivo_usato,
        "backtest": backtest,
        "punteggi": punteggi,
        "importanze": importanze,
        "incroci": incroci,
        "uniformita": {"chi2": round(float(chi2), 1),
                       "p_value": round(float(p_value), 4),
                       "uniforme": bool(uniforme)},
        "clusters": clusters,
    }


# ---------------------------------------------------------------------- cache

def _chiave(gioco, ruota):
    return f"ml_risultati_{gioco}_{ruota or 'SE'}"


def analizza_e_salva(con, gioco, ruota=None):
    cfg = carica_config(con)
    risultati = analizza(con, gioco, ruota, cfg)
    db.set_meta(con, _chiave(gioco, ruota), json.dumps(risultati))
    con.commit()
    return risultati


def risultati_salvati(con, gioco, ruota=None):
    raw = db.get_meta(con, _chiave(gioco, ruota))
    return json.loads(raw) if raw else None


# ----------------------------------------------------------- generatore neurale

def generatore_neurale(con, gioco, ruota, cfg, progresso=None, stop=None):
    """Rete neurale che impara a mappare le ultime `gen_lag` estrazioni nella
    successiva (numeri ordinati per posizione). Addestrata su tutto lo storico
    tranne l'ultimo 10%, tenuto come test. Un comitato di `gen_reti` reti con
    semi diversi genera `gen_simulazioni` estrazioni perturbando gli ingressi
    con rumore gaussiano; la proposta è la media delle generazioni.

    progresso: callback opzionale (stringa di stato).
    stop: threading.Event opzionale — se impostato, l'addestramento si
    interrompe con InterruptedError alla prima occasione utile."""
    def segnala(msg):
        if progresso:
            progresso(msg)

    def controlla():
        if stop is not None and stop.is_set():
            raise InterruptedError("Generazione fermata dall'utente")

    estr = list(reversed(stats._estrazioni(con, gioco, ruota)))
    finestra_storico = int(cfg.get("gen_finestra", 0) or 0)
    if finestra_storico:
        estr = estr[-finestra_storico:]
    if len(estr) < 400:
        raise ValueError("Storico insufficiente (servono ≥400 estrazioni)")
    date = [d for d, _ in estr]
    X = np.array([sorted(nums) for _, nums in estr], dtype=float)
    T, k = X.shape

    lag = max(2, min(int(cfg["gen_lag"]), 200))
    reti = max(1, min(int(cfg["gen_reti"]), 100))
    simulazioni = max(100, min(int(cfg["gen_simulazioni"]), 200000))
    rumore = max(0.0, min(float(cfg["gen_rumore"]), 2.0))
    epoche = max(10, min(int(cfg.get("gen_epoche", 300)), 5000))
    pazienza = max(0, min(int(cfg.get("gen_pazienza", 15)), 500))
    try:
        neuroni = tuple(int(x) for x in
                        str(cfg["gen_neuroni"]).replace(" ", "").split(",")
                        if x and 0 < int(x) <= 8192)
    except ValueError:
        neuroni = (64, 32)
    neuroni = neuroni or (64, 32)

    ingressi = np.array([X[t - lag:t].ravel() for t in range(lag, T)])
    uscite = X[lag:]
    n_test = max(30, len(uscite) // 10)  # ultimo 10% come test
    Xtr, ytr = ingressi[:-n_test], uscite[:-n_test]
    Xte, yte = ingressi[-n_test:], uscite[-n_test:]

    scala = StandardScaler().fit(Xtr)
    base_mae = float(np.abs(yte - ytr.mean(axis=0)).mean())

    ultimo_ingresso = X[-lag:].ravel()
    dev_std = Xtr.std(axis=0)
    rng = np.random.default_rng(20260101)
    per_rete = max(1, simulazioni // reti)

    scelta = str(cfg.get("gen_dispositivo", "auto")).lower()
    usa_gpu, gpu_mancante = False, False
    if scelta in ("auto", "gpu"):
        try:
            import torch
            usa_gpu = torch.cuda.is_available()
        except ImportError:
            usa_gpu = False
        gpu_mancante = scelta == "gpu" and not usa_gpu

    mae_reti, generazioni = [], []
    if usa_gpu:
        dispositivo = f"GPU ({torch.cuda.get_device_name(0)})"
        dev = torch.device("cuda")
        Xtr_t = torch.tensor(scala.transform(Xtr), dtype=torch.float32,
                             device=dev)
        ytr_t = torch.tensor(ytr, dtype=torch.float32, device=dev)
        Xte_t = torch.tensor(scala.transform(Xte), dtype=torch.float32,
                             device=dev)
        n_val = max(20, len(Xtr) // 10)  # coda del train per l'early stopping
        for seme in range(reti):
            controlla()
            torch.manual_seed(seme)
            strati, ingresso = [], Xtr_t.shape[1]
            for h in neuroni:
                strati += [torch.nn.Linear(ingresso, h), torch.nn.ReLU()]
                ingresso = h
            strati.append(torch.nn.Linear(ingresso, k))
            rete = torch.nn.Sequential(*strati).to(dev)
            with torch.no_grad():  # parte dalla media delle posizioni:
                rete[-1].bias.copy_(ytr_t.mean(dim=0))
            ottimizza = torch.optim.Adam(rete.parameters(), lr=1e-3)
            mse = torch.nn.MSELoss()
            migliore, senza_migliorie, stato = float("inf"), 0, None
            for epoca in range(epoche):
                if epoca % 10 == 0:
                    controlla()
                    segnala(f"GPU — rete {seme + 1}/{reti}, "
                            f"epoca {epoca + 1}/{epoche}")
                rete.train()
                ottimizza.zero_grad()
                errore = mse(rete(Xtr_t[:-n_val]), ytr_t[:-n_val])
                errore.backward()
                ottimizza.step()
                if not pazienza:        # 0 = niente early stopping
                    continue
                rete.eval()
                with torch.no_grad():
                    val = mse(rete(Xtr_t[-n_val:]), ytr_t[-n_val:]).item()
                if val < migliore - 1e-4:
                    migliore, senza_migliorie = val, 0
                    stato = {kk: v.detach().clone()
                             for kk, v in rete.state_dict().items()}
                else:
                    senza_migliorie += 1
                    if senza_migliorie >= pazienza:
                        break
            if pazienza and stato:
                rete.load_state_dict(stato)
            rete.eval()
            with torch.no_grad():
                previsto_test = rete(Xte_t).cpu().numpy()
                mae_reti.append(float(np.abs(yte - previsto_test).mean()))
                disturbo = rng.normal(0.0, rumore * dev_std,
                                      size=(per_rete, len(ultimo_ingresso)))
                ingressi = torch.tensor(
                    scala.transform(ultimo_ingresso + disturbo),
                    dtype=torch.float32, device=dev)
                generazioni.append(rete(ingressi).cpu().numpy())
    else:
        dispositivo = ("CPU (scikit-learn) — GPU richiesta ma CUDA non "
                       "disponibile" if gpu_mancante else
                       "CPU (scikit-learn)")
        for seme in range(reti):
            controlla()
            segnala(f"CPU — rete {seme + 1}/{reti}")
            rete = MLPRegressor(hidden_layer_sizes=neuroni, max_iter=epoche,
                                early_stopping=pazienza > 0,
                                n_iter_no_change=pazienza or 10,
                                random_state=seme)
            rete.fit(scala.transform(Xtr), ytr)
            previsto_test = rete.predict(scala.transform(Xte))
            mae_reti.append(float(np.abs(yte - previsto_test).mean()))
            disturbo = rng.normal(0.0, rumore * dev_std,
                                  size=(per_rete, len(ultimo_ingresso)))
            lotti = rete.predict(scala.transform(ultimo_ingresso + disturbo))
            generazioni.append(lotti)
    generato = np.vstack(generazioni)          # (≈simulazioni, k)

    media_pos = generato.mean(axis=0)
    std_pos = generato.std(axis=0)

    numeri, usati = [], set()
    for valore in media_pos:
        n = int(round(max(1, min(90, valore))))
        while n in usati and n < 90:
            n += 1
        while n in usati and n > 1:
            n -= 1
        usati.add(n)
        numeri.append(n)

    return {
        "gioco": gioco, "ruota": ruota,
        "calcolata": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "config": {"gen_lag": lag, "gen_reti": reti,
                   "gen_simulazioni": simulazioni,
                   "gen_neuroni": cfg["gen_neuroni"],
                   "gen_rumore": rumore, "gen_epoche": epoche,
                   "gen_pazienza": pazienza},
        "periodo": {"dal": date[0], "al": date[-1],
                    "campioni_train": len(ytr), "campioni_test": n_test},
        "neuroni": list(neuroni),
        "dispositivo": dispositivo,
        "generate": int(generato.shape[0]),
        "numeri": sorted(numeri),
        "posizioni": [{"posizione": i + 1,
                       "media": round(float(media_pos[i]), 1),
                       "std": round(float(std_pos[i]), 1),
                       "numero": numeri[i]} for i in range(k)],
        "mae_reti": [round(m, 2) for m in mae_reti],
        "mae_medio": round(float(np.mean(mae_reti)), 2),
        "mae_baseline": round(base_mae, 2),
    }


def generatore_e_salva(con, gioco, ruota=None, progresso=None, stop=None):
    cfg = carica_config(con)
    risultati = generatore_neurale(con, gioco, ruota, cfg,
                                   progresso=progresso, stop=stop)
    db.set_meta(con, f"ml_generatore_{gioco}_{ruota or 'SE'}",
                json.dumps(risultati))
    con.commit()
    return risultati


def generatore_salvato(con, gioco, ruota=None):
    raw = db.get_meta(con, f"ml_generatore_{gioco}_{ruota or 'SE'}")
    return json.loads(raw) if raw else None


# ------------------------------------------------------------ scanner di bias

def scanner_bias(con, gioco, ruota):
    """Caccia ai difetti del meccanismo di estrazione, stile Draft Lottery
    1970: oltre all'uniformità globale, verifica che la distribuzione dei
    numeri non dipenda dalla posizione di estrazione, dal mese o dal giorno
    della settimana. Test chi-quadro di indipendenza sulle tabelle di
    contingenza (gruppo × decina); soglia con correzione di Bonferroni."""
    from scipy.stats import chi2_contingency

    if gioco == "lotto":
        righe = con.execute(
            "SELECT data, n1, n2, n3, n4, n5 FROM lotto WHERE ruota = ? "
            "ORDER BY data", (ruota,)).fetchall()
        k = 5
    else:
        righe = con.execute(
            "SELECT data, n1, n2, n3, n4, n5, n6 FROM superenalotto "
            "ORDER BY data").fetchall()
        k = 6
    if len(righe) < 300:
        raise ValueError("Servono almeno 300 estrazioni")

    decina = lambda n: (n - 1) // 10
    GIORNI = ["lunedì", "martedì", "mercoledì", "giovedì",
              "venerdì", "sabato", "domenica"]

    conteggi_num = np.zeros(90)
    per_posizione = np.zeros((k, 9))
    per_mese = np.zeros((12, 9))
    per_giorno = np.zeros((7, 9))
    for r in righe:
        quando = datetime.strptime(r["data"], "%Y-%m-%d")
        for pos in range(k):
            n = r[f"n{pos + 1}"]
            conteggi_num[n - 1] += 1
            per_posizione[pos][decina(n)] += 1
            per_mese[quando.month - 1][decina(n)] += 1
            per_giorno[quando.weekday()][decina(n)] += 1

    test = []

    chi2, p = chisquare(conteggi_num)
    test.append({
        "nome": "Uniformità globale dei 90 numeri",
        "descrizione": "Ogni numero esce con la stessa frequenza?",
        "chi2": round(float(chi2), 1), "gdl": 89, "p_value": round(float(p), 4)})

    if gioco == "lotto":  # il SuperEnalotto è archiviato in ordine crescente:
        # l'ordine reale di estrazione non è disponibile e il test non ha senso
        chi2, p, gdl, _ = chi2_contingency(per_posizione)
        test.append({
            "nome": "Indipendenza dalla posizione di estrazione",
            "descrizione": f"La distribuzione delle decine è la stessa per il "
                           f"1º, 2º ... {k}º numero estratto? (in un'urna ben "
                           "mescolata, sì)",
            "chi2": round(float(chi2), 1), "gdl": int(gdl),
            "p_value": round(float(p), 4)})

    chi2, p, gdl, _ = chi2_contingency(per_mese)
    test.append({
        "nome": "Indipendenza dal mese",
        "descrizione": "I numeri estratti hanno la stessa distribuzione in "
                       "tutti i mesi dell'anno? (il difetto del Draft "
                       "Lottery 1970: capsule mescolate male per mese)",
        "chi2": round(float(chi2), 1), "gdl": int(gdl),
        "p_value": round(float(p), 4)})

    giorni_attivi = per_giorno[per_giorno.sum(axis=1) > 0]
    nomi_giorni = [GIORNI[i] for i in range(7) if per_giorno[i].sum() > 0]
    if len(giorni_attivi) >= 2:
        chi2, p, gdl, _ = chi2_contingency(giorni_attivi)
        test.append({
            "nome": "Indipendenza dal giorno della settimana",
            "descrizione": "Stessa distribuzione nei giorni di estrazione "
                           f"({', '.join(nomi_giorni)})?",
            "chi2": round(float(chi2), 1), "gdl": int(gdl),
            "p_value": round(float(p), 4)})

    soglia = round(0.05 / len(test), 4)  # Bonferroni
    anomalie = 0
    for t in test:
        t["anomalo"] = t["p_value"] < soglia
        anomalie += t["anomalo"]

    return {
        "gioco": gioco, "ruota": ruota,
        "calcolata": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "estrazioni": len(righe),
        "dal": righe[0]["data"], "al": righe[-1]["data"],
        "test": test, "soglia": soglia, "anomalie": int(anomalie),
    }


def bias_e_salva(con, gioco, ruota=None):
    risultati = scanner_bias(con, gioco, ruota)
    db.set_meta(con, f"ml_bias_{gioco}_{ruota or 'SE'}", json.dumps(risultati))
    con.commit()
    return risultati


def bias_salvato(con, gioco, ruota=None):
    raw = db.get_meta(con, f"ml_bias_{gioco}_{ruota or 'SE'}")
    return json.loads(raw) if raw else None


# ------------------------------------------------------------ algoritmo MASTER

GRIGLIA_MASTER = {
    # tipologie e grandezze diverse: da reti minime a profonde/larghe
    "architetture": ["32", "64,32", "128,64,32", "256,128", "512,256,128",
                     "64,64,64,64", "1024,512", "2048,1024,512"],
    "lag": [5, 10, 20, 50],                  # estrazioni in ingresso
    "addestramenti": [(100, 15), (300, 0)],  # (epoche, pazienza early stop)
    "reti": [3, 7],                          # reti nel comitato
    "finestre": [500, 2000, 0],              # estrazioni di storico (0 = tutte)
}


def stato_gpu():
    """Telemetria della GPU via nvidia-smi (None se non disponibile)."""
    import subprocess
    try:
        riga = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,temperature.gpu,utilization.gpu,"
             "memory.used,memory.total,power.draw,power.limit",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip()
        nome, temp, util, vram_u, vram_t, pw, pw_max = \
            [c.strip() for c in riga.split(",")]
        return {"nome": nome, "temperatura": int(float(temp)),
                "utilizzo": int(float(util)),
                "vram_usata": round(float(vram_u) / 1024, 1),
                "vram_totale": round(float(vram_t) / 1024, 1),
                "potenza": round(float(pw)),
                "potenza_max": round(float(pw_max))}
    except Exception:
        return None


def master_neurale(con, gioco, ruota, progresso=None, stop=None,
                   pausa=None, griglia=None):
    """Grid search sul generatore neurale: prova ogni combinazione della
    griglia, registra ogni previsione nella matrice (riconoscibile dalle
    impostazioni della rete che l'ha generata) e sintetizza i numeri finali
    con algoritmi deterministici. Salvataggio incrementale dopo ogni
    configurazione; pausa di `pausa` secondi tra una generazione e la
    successiva."""
    import time as _time
    from itertools import product

    g = griglia or GRIGLIA_MASTER
    combinazioni = list(product(g["architetture"], g["lag"],
                                g["addestramenti"], g["reti"], g["finestre"]))
    cfg_base = carica_config(con)
    if pausa is None:
        pausa = max(0, min(int(cfg_base.get("master_pausa", 10)), 600))
    chiave = f"ml_master_{gioco}_{ruota or 'SE'}"
    avviata = datetime.now().strftime("%Y-%m-%d %H:%M")
    righe = []

    def salva(stato):
        dati = {"gioco": gioco, "ruota": ruota, "avviata": avviata,
                "aggiornata": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "stato": stato, "totale": len(combinazioni),
                "completate": len(righe), "pausa": pausa,
                "righe": righe, "sintesi": _sintesi_master(righe)}
        db.set_meta(con, chiave, json.dumps(dati))
        con.commit()
        return dati

    def fermato():
        return stop is not None and stop.is_set()

    for i, (arch, lag, (epoche, pazienza), reti, fin) in enumerate(combinazioni):
        if fermato():
            salva("fermata")
            raise InterruptedError("MASTER fermato dall'utente")
        etichetta = (f"[{arch}] lag {lag}, {epoche} epoche (paz. {pazienza}), "
                     f"{reti} reti, storico {fin or 'tutto'}")
        if progresso:
            progresso(f"configurazione {i + 1}/{len(combinazioni)}: {etichetta}")

        cfg = dict(cfg_base)
        cfg.update({"gen_neuroni": arch, "gen_lag": lag, "gen_epoche": epoche,
                    "gen_pazienza": pazienza, "gen_reti": reti,
                    "gen_finestra": fin, "gen_simulazioni": 600})
        riga = {"n": i + 1, "architettura": arch, "lag": lag,
                "epoche": epoche, "pazienza": pazienza, "reti": reti,
                "finestra": fin or 0}
        t0 = _time.time()
        try:
            esito = generatore_neurale(con, gioco, ruota, cfg, stop=stop)
            riga.update(numeri=esito["numeri"], mae=esito["mae_medio"],
                        baseline=esito["mae_baseline"],
                        durata=round(_time.time() - t0, 1))
        except InterruptedError:
            salva("fermata")
            raise
        except Exception as exc:
            riga["errore"] = str(exc)
        righe.append(riga)
        salva("in corso")

        if i + 1 < len(combinazioni):       # pausa tra le generazioni
            for _ in range(int(pausa)):
                if fermato():
                    salva("fermata")
                    raise InterruptedError("MASTER fermato dall'utente")
                _time.sleep(1)

    return salva("completata")


def _sintesi_master(righe):
    """Algoritmi deterministici sulla matrice dei risultati."""
    valide = [r for r in righe if not r.get("errore")]
    if not valide:
        return None
    k = len(valide[0]["numeri"])

    # 1) voto semplice: i numeri proposti dal maggior numero di configurazioni
    conta = Counter(n for r in valide for n in r["numeri"])
    classifica = sorted(conta.items(), key=lambda nc: (-nc[1], nc[0]))
    voto_semplice = [n for n, _ in classifica[:k]]

    # 2) voto pesato sulla qualità: peso = 1/MAE sul test (reti che sbagliano
    #    meno contano di più)
    pesato = Counter()
    for r in valide:
        peso = 1.0 / max(float(r["mae"]), 0.1)
        for n in r["numeri"]:
            pesato[n] += peso
    voto_pesato = [n for n, _ in sorted(pesato.items(),
                                        key=lambda nc: (-nc[1], nc[0]))[:k]]

    # 3) media per posizione tra tutte le configurazioni
    medie = np.mean([r["numeri"] for r in valide], axis=0)
    media_posizioni, usati = [], set()
    for v in medie:
        n = int(round(max(1, min(90, v))))
        while n in usati and n < 90:
            n += 1
        while n in usati and n > 1:
            n -= 1
        usati.add(n)
        media_posizioni.append(n)

    migliore = min(valide, key=lambda r: float(r["mae"]))
    return {
        "configurazioni": len(valide),
        "voto_semplice": voto_semplice,
        "voto_pesato": voto_pesato,
        "media_posizioni": sorted(media_posizioni),
        "frequenze": classifica[:15],
        "migliore": {kk: migliore[kk] for kk in
                     ("n", "architettura", "lag", "epoche", "pazienza",
                      "reti", "finestra", "numeri", "mae", "baseline")},
    }


def master_salvato(con, gioco, ruota=None):
    raw = db.get_meta(con, f"ml_master_{gioco}_{ruota or 'SE'}")
    return json.loads(raw) if raw else None
