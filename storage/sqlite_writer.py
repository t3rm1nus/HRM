import sqlite3
import os
import json
import logging

logger = logging.getLogger("HRM")

DB_PATH = os.path.join("data", "historico.db")

def _init_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS estados (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            mercado TEXT,
            estrategia TEXT,
            portfolio TEXT,
            universo TEXT,
            exposicion TEXT,
            senales TEXT,
            ordenes TEXT,
            riesgo TEXT,
            deriva BOOLEAN
        )
    """)
    conn.commit()
    conn.close()

def guardar_estado_sqlite(state: dict):
    """
    Guarda el estado global en SQLite.
    Serializa campos complejos como JSON.
    """
    _init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO estados
        (mercado, estrategia, portfolio, universo, exposicion, senales, ordenes, riesgo, deriva)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        json.dumps(state.get("mercado", {})),
        state.get("estrategia", ""),
        json.dumps(state.get("portfolio", {})),
        json.dumps(state.get("universo", [])),
        json.dumps(state.get("exposicion", {})),
        json.dumps(state.get("senales", {})),
        json.dumps(state.get("ordenes", [])),
        json.dumps(state.get("riesgo", {})),
        int(state.get("deriva", False)),
    ))
    conn.commit()
    conn.close()

    logger.info(f"Estado guardado en SQLite → {DB_PATH}")
