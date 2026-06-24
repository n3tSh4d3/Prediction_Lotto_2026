# Lotto Manager

Sistema di gestione delle previsioni per **Lotto** e **SuperEnalotto**: archivio
storico completo, analisi statistiche, generazione di proposte di giocata e
verifica degli esiti.

> Le estrazioni sono eventi casuali e indipendenti: nessuna strategia aumenta la
> probabilità di vincita. Il sistema serve ad analizzare lo storico e a costruire
> giocate secondo criteri statistici espliciti e ripetibili.

## Avvio

```bash
pip install -r requirements.txt   # solo la prima volta
./run.sh
```

Poi apri **http://localhost:8010** nel browser.

## Funzionalità

- **Dashboard** — copertura archivio, ultima estrazione Lotto (tutte le ruote)
  e ultimo concorso SuperEnalotto. Il pulsante *Aggiorna estrazioni* scarica gli
  archivi aggiornati dalle fonti online.
- **Lotto / SuperEnalotto** — statistiche filtrabili per ruota e periodo:
  numeri più frequenti, ritardatari (ritardo attuale e record storico), grafico
  delle frequenze, distribuzione per decine, pari/dispari, bassi/alti, somma,
  coppie e terzine più frequenti, analisi dei *numeri spia* (solo Lotto).
- **Analisi ML** — per ogni gioco/ruota: 7 modelli di machine learning
  (regressione logistica, random forest, gradient boosting, extra trees,
  naive Bayes, k-nearest neighbors, rete neurale MLP) più 2 modalità
  composite — ensemble parallelo con controlli incrociati (AUC per base,
  correlazione dei punteggi, accordo sulle top-15) e stacking sequenziale
  (meta-modello logistico sulle previsioni delle basi) — addestrati sulle
  feature di ogni numero (ritardo relativo, frequenze su più finestre, trend,
  presenza nell'ultima estrazione) con punteggi per la prossima estrazione;
  **generatore neurale**: comitato di reti MLP addestrato su
  tutto lo storico (ultimo 10% come test), genera N estrazioni perturbando
  gli ingressi e propone la media, con MAE confrontato alla baseline;
  **backtest walk-forward** su estrazioni mai viste in addestramento, con AUC e
  confronto punti tra modello, ritardatari, frequenti e attesa del puro caso;
  test di uniformità chi-quadro (l'estrazione è equa?); clustering k-means dei
  numeri per co-occorrenza. I risultati sono salvati in cache nel database.
- **Analisi NET** — framework modulare di 21 algoritmi attivabili a scelta
  su una base dati (ruota del Lotto o lista unica SuperEnalotto), ognuno con
  parametri regolabili: distanze, media-differenziali ML, decine-cadenze,
  giocata impopolare, simpatie tra ruote, vertibili/complementi, meta-backtest,
  ciclometria (diametrali/terzine/quadrati), catena di Markov, somme (7
  generatori), indice di convenienza, isotopi, Monte Carlo, varianti di spia,
  fase multi-ruota, indicatori multipli, gemelli, archi ciclometrici, nodi
  numerici, diagonale estrattiva, algoritmo genetico (2 modalità). Pannello di
  convergenza tra le analisi e salvataggio in giocate senza ricaricare la
  pagina. Per aggiungere un algoritmo: una funzione decorata in `app/net.py`.
- **Analisi Generative** — esperimento di ricostruzione sequenziale dello
  storico (GPU/CPU): conta i tentativi casuali necessari per ri-estrarre ogni
  estrazione dopo la precedente, con monitor live, matrice delle transizioni,
  grafico della "forma d'onda" dei conteggi e analisi complete
  (autocorrelazione, spettro, distribuzione vs geometrica) più la previsione
  derivata dal modello d'onda.
- **Previsioni** — genera proposte di giocata con 6 strategie: `ritardatari`,
  `frequenti`, `mista`, `bilanciata`, `spia`, `ml` (usa i punteggi del modello).
  Ogni proposta è accompagnata dalla motivazione statistica e può essere salvata.
- **Giocate** — elenco delle giocate salvate, da presentare all'operatore.
  Dopo l'estrazione, *Verifica esiti* confronta ogni giocata con la prima
  estrazione successiva al salvataggio (ambata/ambo/terno/… per il Lotto,
  punti per il SuperEnalotto).
- **Setup** — parametri delle analisi ML impostabili dall'operatore: tipo di
  modello, estrazioni di addestramento e di backtest, finestre di frequenza,
  numeri puntati nel backtest, numero di cluster.

## Fonti dati (aggiornate quotidianamente)

- Lotto, dal 7/1/1939: [robyzarra72/lotto-data](https://github.com/robyzarra72/lotto-data) (`storico.zip`)
- SuperEnalotto, dal 3/12/1997: [Lottopyrhon/Estrazioni_Superenalotto](https://github.com/Lottopyrhon/Estrazioni_Superenalotto) (`superenalotto.txt`)

L'aggiornamento reimporta l'intero archivio (idempotente, `INSERT OR REPLACE`),
quindi corregge anche eventuali rettifiche retroattive delle fonti.

## Struttura

```
app/
  db.py        # schema SQLite (lotto, superenalotto, giocate, meta)
  fetch.py     # download e import dalle fonti online
  stats.py     # frequenze, ritardi, combinazioni, distribuzioni, numeri spia
  ml.py        # modelli ML, backtest, chi-quadro, clustering (scikit-learn)
  net.py       # framework Analisi NET: 21 algoritmi modulari
  predict.py   # strategie di generazione giocate + verifica esiti
  main.py      # web app FastAPI
  templates/   # pagine Jinja2 (Bootstrap + Chart.js)
lotto.db       # database (creato al primo avvio)
```

## Aggiornamento da riga di comando

```bash
python3 -c "from app import db, fetch; print(fetch.aggiorna_tutto(db.connect()))"
```

Utile per un cron giornaliero (le fonti si aggiornano la sera dopo le estrazioni).

## Licenza

Distribuito sotto i termini della **GNU General Public License v3.0**.
Vedi il file [LICENSE](LICENSE) per il testo completo.

Copyright (C) 2026 Adriano Condro
