"""Aggiornamento dello storico da fonti online.

Fonti (archivi completi, aggiornati quotidianamente):
- Lotto:         github.com/robyzarra72/lotto-data        (storico.zip, dal 1939)
- SuperEnalotto: github.com/Lottopyrhon/Estrazioni_Superenalotto (superenalotto.txt, dal 1997)
"""
import io
import urllib.request
import zipfile
from datetime import datetime, timezone

from . import db

URL_LOTTO = "https://raw.githubusercontent.com/robyzarra72/lotto-data/main/storico.zip"
URL_SUPERENALOTTO = (
    "https://raw.githubusercontent.com/Lottopyrhon/Estrazioni_Superenalotto/main/superenalotto.txt"
)
TIMEOUT = 120


def _scarica(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return resp.read()


def aggiorna_lotto(con):
    """Scarica e importa l'intero storico del Lotto. Ritorna il n. di righe importate."""
    raw = _scarica(URL_LOTTO)
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        nome = next(n for n in zf.namelist() if n.endswith(".txt"))
        testo = zf.read(nome).decode("utf-8", errors="replace")

    righe = []
    for line in testo.splitlines():
        campi = line.split("\t")
        if len(campi) != 7:
            continue
        data, ruota, *numeri = campi
        try:
            nums = [int(n) for n in numeri]
        except ValueError:
            continue
        righe.append((data.replace("/", "-"), ruota.strip().upper(), *nums))

    if not righe:
        raise RuntimeError("Archivio Lotto vuoto o in formato inatteso")

    con.executemany(
        "INSERT OR REPLACE INTO lotto (data, ruota, n1, n2, n3, n4, n5) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        righe,
    )
    return len(righe)


def aggiorna_superenalotto(con):
    """Scarica e importa l'intero storico del SuperEnalotto. Ritorna il n. di concorsi."""
    testo = _scarica(URL_SUPERENALOTTO).decode("utf-8", errors="replace")

    righe = []
    for line in testo.splitlines():
        campi = line.strip().split(",")
        # formato: concorso,GG/MM/AAAA,n1..n6,jolly,superstar (00 = assente)
        if len(campi) != 10:
            continue
        try:
            concorso = int(campi[0])
            data = datetime.strptime(campi[1], "%d/%m/%Y").strftime("%Y-%m-%d")
            nums = [int(n) for n in campi[2:8]]
            jolly = int(campi[8])
            superstar = int(campi[9]) or None
        except ValueError:
            continue
        righe.append((data, concorso, *nums, jolly, superstar))

    if not righe:
        raise RuntimeError("Archivio SuperEnalotto vuoto o in formato inatteso")

    con.executemany(
        "INSERT OR REPLACE INTO superenalotto "
        "(data, concorso, n1, n2, n3, n4, n5, n6, jolly, superstar) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        righe,
    )
    return len(righe)


def aggiorna_tutto(con):
    """Aggiorna entrambi gli archivi e registra l'esito. Ritorna un dict riepilogo."""
    esito = {}
    for nome, fn in (("lotto", aggiorna_lotto), ("superenalotto", aggiorna_superenalotto)):
        try:
            esito[nome] = {"righe": fn(con), "errore": None}
        except Exception as exc:  # rete giù, formato cambiato, ...
            esito[nome] = {"righe": 0, "errore": str(exc)}
    con.commit()
    db.set_meta(con, "ultimo_aggiornamento",
                datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    con.commit()
    return esito
