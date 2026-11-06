#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SYSTEM CLEANUP SCRIPT
Limpia todos los logs y archivos temporales para iniciar un proceso nuevo
"""

import os
import shutil
import json
import glob
from datetime import datetime
import logging

# Configurar logging para el cleanup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - CLEANUP - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SystemCleanup:
    """Utilidad para limpiar archivos temporales y de estado del sistema HRM"""

    def __init__(self, base_path="."):
        self.base_path = os.path.abspath(base_path)
        self.deleted_files = []
        self.deleted_dirs = []

    def safe_delete_file(self, filepath, description=""):
        """Elimina un archivo de forma segura"""
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                self.deleted_files.append(filepath)
                logger.info(f"🗑️ Eliminado: {description} {filepath}")
                return True
            else:
                logger.debug(f"ℹ️ No existe: {filepath}")
                return False
        except Exception as e:
            logger.warning(f"⚠️ Error eliminando {filepath}: {e}")
            return False

    def safe_delete_directory(self, dirpath, description=""):
        """Elimina un directorio de forma segura"""
        try:
            if os.path.exists(dirpath):
                shutil.rmtree(dirpath)
                self.deleted_dirs.append(dirpath)
                logger.info(f"🗂️ Directorio eliminado: {description} {dirpath}")
                return True
            else:
                logger.debug(f"ℹ️ Directorio no existe: {dirpath}")
                return False
        except Exception as e:
            logger.warning(f"⚠️ Error eliminando directorio {dirpath}: {e}")
            return False

    def clean_logs(self):
        """Limpia todos los directorios de logs"""
        logger.info("🧹 Limpiando directorios de logs...")

        # Directorios principales de logs
        log_dirs = [
            "logs",
            "test_logs",
            "backtesting/logs"
        ]

        for log_dir in log_dirs:
            full_path = os.path.join(self.base_path, log_dir)
            self.safe_delete_directory(full_path, "logs:")

    def clean_hacienda(self):
        """Limpia archivos del sistema de hacienda (taxes)"""
        logger.info("🧹 Limpiando archivos de hacienda (tax system)...")

        hacienda_files = [
            "hacienda/posiciones_fifo.json",
            "portfolio_state_live.json"
        ]

        for file_path in hacienda_files:
            full_path = os.path.join(self.base_path, file_path)
            self.safe_delete_file(full_path, "hacienda:")

    def clean_portfolio_data(self):
        """Limpia datos de portfolio"""
        logger.info("🧹 Limpiando datos de portfolio...")

        portfolio_patterns = [
            "portfolio_*.json",
            "*portfolio*.json"
        ]

        for pattern in portfolio_patterns:
            full_pattern = os.path.join(self.base_path, pattern)
            for file_path in glob.glob(full_pattern):
                self.safe_delete_file(file_path, "portfolio:")

    def clean_inference_data(self):
        """Limpia datos de inferencia y sentiment"""
        logger.info("🧹 Limpiando datos de inferencia y sentiment...")

        # Directorio principal de datos
        data_dir = os.path.join(self.base_path, "data")

        if os.path.exists(data_dir):
            try:
                # Eliminar archivos específicos que contienen datos
                inference_files = [
                    "sentiment_bert_cache.json",
                    "sentiment_cache_timestamp.json",
                    "sentiment_l2_*.json",
                    "sentiment_inference_*.csv",
                    "sentiment_summary_*.json"
                ]

                for pattern in inference_files:
                    full_pattern = os.path.join(data_dir, pattern)
                    for file_path in glob.glob(full_pattern):
                        self.safe_delete_file(file_path, "inference:")

                # Eliminar subdirectorios de datos de inferencia
                subdirs_to_clean = [
                    "datos_inferencia"
                ]

                for subdir in subdirs_to_clean:
                    full_subdir = os.path.join(data_dir, subdir)
                    self.safe_delete_directory(full_subdir, "inference data:")

            except Exception as e:
                logger.warning(f"⚠️ Error limpiando datos de inferencia: {e}")

    def clean_analysis_data(self):
        """Limpia archivos de análisis de portfolio e histórico de trading"""
        logger.info("🧹 Limpiando archivos de análisis de portfolio e histórico...")

        # Archivos de análisis principales
        analysis_files = [
            "historico.csv",  # Historial principal del sistema
            "historico.db",   # Base de datos del historial
        ]

        for file_path in analysis_files:
            full_path = os.path.join(self.base_path, "data", file_path)
            self.safe_delete_file(full_path, "analysis:history")

        # Archivos de portfolio
        portfolio_dir = os.path.join(self.base_path, "data", "portfolio")
        if os.path.exists(portfolio_dir):
            try:
                # Eliminar archivos de historial de portfolio
                portfolio_patterns = [
                    "portfolio_history*.csv",  # Todos los archivos de historial
                    "portfolio_state_*.json",  # Estados de portfolio por tipo
                ]

                for pattern in portfolio_patterns:
                    for file_path in glob.glob(os.path.join(portfolio_dir, pattern)):
                        self.safe_delete_file(file_path, "analysis:portfolio")

            except Exception as e:
                logger.warning(f"⚠️ Error limpiando archivos de portfolio: {e}")

        # Archivos de trading history en logs
        logs_dir = os.path.join(self.base_path, "data", "logs")
        if os.path.exists(logs_dir):
            trading_history_files = [
                "trades_history.csv",  # Historial de operaciones
            ]

            for file_path in trading_history_files:
                full_path = os.path.join(logs_dir, file_path)
                self.safe_delete_file(full_path, "analysis:trades")

    def clean_results_data(self):
        """Limpia archivos de resultados"""
        logger.info("🧹 Limpiando archivos de resultados...")

        results_dir = os.path.join(self.base_path, "results")
        self.safe_delete_directory(results_dir, "results:")

        # Also clean result files in root
        result_patterns = [
            "allocation_tiers_results.json",
            "risk_adjusted_sizing_results.json",
            "regimen_analysis.py"
        ]

        for file_path in result_patterns:
            full_path = os.path.join(self.base_path, file_path)
            self.safe_delete_file(full_path, "results:")

    def clean_temporary_files(self):
        """Limpia archivos temporales del sistema"""
        logger.info("🧹 Limpiando archivos temporales...")

        # Cache de Python
        pycache_dirs = []
        for root, dirs, files in os.walk(self.base_path):
            if "__pycache__" in dirs:
                pycache_dirs.append(os.path.join(root, "__pycache__"))

        for pycache_dir in pycache_dirs:
            rel_path = os.path.relpath(pycache_dir, self.base_path)
            self.safe_delete_directory(pycache_dir, f"cache Python: {rel_path}")

        # Archivos temporales específicos
        temp_patterns = [
            "*.pyc",
            "*.pyo",
            "*.tmp",
            ".pytest_cache",
            "kk.py"  # Archivo de debug que parece temporal
        ]

        for pattern in temp_patterns:
            if "*" in pattern:
                for file_path in glob.glob(os.path.join(self.base_path, pattern)):
                    self.safe_delete_file(file_path, "temp:")
            else:
                full_path = os.path.join(self.base_path, pattern)
                self.safe_delete_file(full_path, "temp:")

    def clean_ml_models_cache(self):
        """Limpia cache de modelos ML temporales"""
        logger.info("🧹 Limpiando cache de modelos ML...")

        # No eliminar los modelos entrenados, solo cache temporal
        pass  # Los modelos deben quedar

    def create_fresh_directories(self):
        """Crea directorios frescos necesarios"""
        logger.info("📁 Creando directorios frescos...")

        dirs_to_create = [
            "logs",
            "data/datos_inferencia",
            "results"
        ]

        for dir_path in dirs_to_create:
            full_path = os.path.join(self.base_path, dir_path)
            try:
                os.makedirs(full_path, exist_ok=True)
                logger.info(f"✅ Directorio creado: {dir_path}")
            except Exception as e:
                logger.warning(f"⚠️ Error creando directorio {dir_path}: {e}")

    def perform_full_cleanup(self):
        """Realiza una limpieza completa de todo el sistema"""
        logger.info("=" * 80)
        logger.info("🧽 INICIANDO LIMPIEZA COMPLETA DEL SISTEMA HRM")
        logger.info("=" * 80)

        start_time = datetime.now()

        # Ejecutar todas las limpiezas
        self.clean_logs()
        self.clean_hacienda()
        self.clean_portfolio_data()
        self.clean_inference_data()
        self.clean_analysis_data()  # NUEVO: Limpia archivos de análisis de portfolio
        self.clean_results_data()
        self.clean_temporary_files()
        self.clean_ml_models_cache()
        self.create_fresh_directories()

        end_time = datetime.now()
        duration = end_time - start_time

        # Reporte final
        logger.info("=" * 80)
        logger.info("✅ LIMPIEZA COMPLETADA")
        logger.info(f"⏱️ Duración: {duration.total_seconds():.2f} segundos")
        logger.info(f"🗑️ Archivos eliminados: {len(self.deleted_files)}")
        logger.info(f"📁 Directorios eliminados: {len(self.deleted_dirs)}")

        if self.deleted_files:
            logger.info("📋 Archivos eliminados:")
            for f in self.deleted_files[:10]:  # Mostrar primeros 10
                logger.info(f"   - {f}")
            if len(self.deleted_files) > 10:
                logger.info(f"   ... y {len(self.deleted_files) - 10} más")

        logger.info("\n🚀 Sistema listo para nuevo proceso")
        logger.info("=" * 80)

        return {
            "deleted_files": len(self.deleted_files),
            "deleted_dirs": len(self.deleted_dirs),
            "duration": duration.total_seconds(),
            "success": True
        }

def main():
    """Función principal para ejecutar desde línea de comandos"""
    import argparse

    parser = argparse.ArgumentParser(description="Limpieza completa del sistema HRM")
    parser.add_argument("--dry-run", action="store_true", help="Solo mostrar qué se eliminaría, sin eliminar")
    parser.add_argument("--path", default=".", help="Directorio base del proyecto")

    args = parser.parse_args()

    # Para dry-run crear una clase mock que solo registre
    if args.dry_run:
        logger.info("🔍 DRY-RUN MODE - Solo mostrando qué se eliminaría")
        cleanup = SystemCleanup(args.path)
        # En dry-run podríamos crear métodos que solo listen archivos sin eliminarlos
        print("\nModo dry-run no implementado completamente. Ejecuta sin --dry-run para limpiar realmente.")
        return

    cleanup = SystemCleanup(args.path)
    result = cleanup.perform_full_cleanup()

    return result

if __name__ == "__main__":
    main()
