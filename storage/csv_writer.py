import csv
import os
import logging

logger = logging.getLogger("HRM")

FILE_PATH = os.path.join("data", "historico.csv")

def guardar_estado_csv(state: dict):
    """
    Guarda el estado global en un CSV.
    Si el archivo no existe, escribe encabezados primero.
    """
    os.makedirs("data", exist_ok=True)

    row = {k: str(v) for k, v in state.items()}  # aplanar diccionario

    file_exists = os.path.isfile(FILE_PATH)
    with open(FILE_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    logger.info(f"Estado guardado en CSV → {FILE_PATH}")
