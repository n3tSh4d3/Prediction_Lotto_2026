"""Database SQLite per lo storico delle estrazioni e le giocate."""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "lotto.db"

RUOTE = ["BA", "CA", "FI", "GE", "MI", "NA", "PA", "RM", "RN", "TO", "VE"]
NOMI_RUOTE = {
    "BA": "Bari", "CA": "Cagliari", "FI": "Firenze", "GE": "Genova",
    "MI": "Milano", "NA": "Napoli", "PA": "Palermo", "RM": "Roma",
    "RN": "Nazionale", "TO": "Torino", "VE": "Venezia",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS lotto (
    data  TEXT NOT NULL,
    ruota TEXT NOT NULL,
    n1 INTEGER, n2 INTEGER, n3 INTEGER, n4 INTEGER, n5 INTEGER,
    PRIMARY KEY (data, ruota)
);
CREATE INDEX IF NOT EXISTS idx_lotto_ruota ON lotto (ruota, data);

CREATE TABLE IF NOT EXISTS superenalotto (
    data TEXT PRIMARY KEY,
    concorso INTEGER,
    n1 INTEGER, n2 INTEGER, n3 INTEGER, n4 INTEGER, n5 INTEGER, n6 INTEGER,
    jolly INTEGER,
    superstar INTEGER
);

CREATE TABLE IF NOT EXISTS giocate (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    creata TEXT NOT NULL,
    gioco TEXT NOT NULL,            -- 'lotto' | 'superenalotto'
    ruota TEXT,                     -- solo per il lotto
    numeri TEXT NOT NULL,           -- es. '5,23,47,88'
    strategia TEXT,
    note TEXT,
    esito TEXT                      -- compilato dalla verifica
);

CREATE TABLE IF NOT EXISTS meta (
    chiave TEXT PRIMARY KEY,
    valore TEXT
);
"""


def connect():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def set_meta(con, chiave, valore):
    con.execute(
        "INSERT INTO meta (chiave, valore) VALUES (?, ?) "
        "ON CONFLICT(chiave) DO UPDATE SET valore = excluded.valore",
        (chiave, str(valore)),
    )


def get_meta(con, chiave, default=None):
    row = con.execute("SELECT valore FROM meta WHERE chiave = ?", (chiave,)).fetchone()
    return row["valore"] if row else default
