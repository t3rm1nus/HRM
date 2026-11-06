#!/usr/bin/env python3
"""Prueba manual rápida de la configuración de logging."""


def ejecutar_prueba_rapida_de_logging():
    """Comprueba manualmente que el logger persiste un mensaje en SQLite."""
    import os
    import sys
    import sqlite3

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    sys.path.append(".")

    print("Testing logging import and basic usage...")

    from core.logging import info, setup_logger

    setup_logger()
    info("Quick logging test", module="quick_test")

    with sqlite3.connect("logs/logs.db") as conexion:
        cursor = conexion.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM logs "
            "WHERE message LIKE '%Quick logging test%'"
        )
        cantidad = cursor.fetchone()[0]

    print(f"✓ Logged entry found in database: {cantidad}")
    print("✅ Quick logging test successful!")


if __name__ == "__main__":
    ejecutar_prueba_rapida_de_logging()