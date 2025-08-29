#!/backtesting/main.py
"""
HRM Backtesting System - Ejecutor Principal
Prueba el sistema completo con datos reales de Binance
"""

import os
import sys
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

# Agregar paths del proyecto
sys.path.append('..')
sys.path.append('../l1_operational')
sys.path.append('../l2_tactic') 
sys.path.append('../l3_strategic')

from getdata import BinanceDataCollector
from hrm_tester import HRMStrategyTester
from performance_analyzer import PerformanceAnalyzer
from report_generator import ReportGenerator


class HRMBacktester:
    """Orquestador principal del backtesting HRM"""
    
    def __init__(self, config_path: str = "backtesting/config.json"):
        self.config = self._load_config(config_path)
        self.setup_logging()
        
        # Inicializar componentes
        self.data_collector = BinanceDataCollector(self.config['binance'])
        self.strategy_tester = HRMStrategyTester(self.config['testing'], self.data_collector)
        self.performance_analyzer = PerformanceAnalyzer(self.config['analysis'])
        self.report_generator = ReportGenerator(self.config['reporting'])
        
        # Estado del backtesting
        self.results = {
            'overall': {},
            'l1_models': {},
            'l2_model': {},
            'l3_models': {}
        }
        
    def _load_config(self, config_path: str) -> Dict:
        """Carga la configuración desde un archivo JSON o usa valores por defecto."""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            logging.warning("Archivo de configuración no encontrado, usando valores por defecto.")
            return {
                'binance': {
                    'symbols': ['BTCUSDT', 'ETHUSDT'],
                    'intervals': ['1m', '5m', '15m', '1h'],
                    'historical_days': 1
                },
                'testing': {
                    'mode': 'full',
                    'lookback_days': 7,
                    'symbols': ['BTCUSDT', 'ETHUSDT'],
                    'intervals': ['1h', '5m', '15m', '1m']
                },
                'analysis': {
                    'generate_charts': True,
                    'export_trades': True,
                    'metrics': ['sharpe', 'drawdown', 'win_rate']
                },
                'reporting': {
                    'output_dir': 'backtesting/results',
                    'generate_charts': True,
                    'detailed_logs': True,
                    'export_trades': True
                }
            }

    def setup_logging(self):
        # Implementación del método que faltaba
        log_level = self.config['reporting'].get('detailed_logs', True)
        if log_level:
            logging.basicConfig(level=logging.INFO,
                                format='%(asctime)s | %(levelname)-8s | %(name)-15s:%(funcName)-20s:%(lineno)-4d - %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S')
        else:
            logging.basicConfig(level=logging.WARNING,
                                format='%(asctime)s | %(levelname)-8s | %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S')
        logging.info("Logging configured successfully.")

    async def run_backtest(self):
        """Ejecuta el proceso de backtesting de punta a punta"""
        self.logger.info("Iniciando backtesting HRM...")
        
        try:
            # 1. Recolectar datos
            historical_data = await self.data_collector.collect_historical_data(
                self.config['testing']['symbols'],
                self.config['testing']['intervals'],
                self.config['testing']['lookback_days']
            )

            if not historical_data:
                self.logger.error("No se pudieron obtener datos históricos. Abortando.")
                return

            # 2. Ejecutar probador de estrategias
            testing_results = await self.strategy_tester.run_all_tests(historical_data)
            
            # 3. Analizar rendimiento
            analyzed_results = self.performance_analyzer.analyze_results(
                testing_results, 
                self.config['analysis']['metrics']
            )
            
            # 4. Generar reportes
            self.report_generator.generate_full_report(
                analyzed_results, 
                self.config['reporting']['output_dir']
            )

            self.results = analyzed_results
            self.logger.info("✅ Backtesting completado con éxito.")
            self.logger.info(f"Resultados finales: {self.results['overall']}")

        except Exception as e:
            self.logger.error(f"❌ Error crítico durante el backtesting: {e}")
            logging.exception("Detalles del error:")

async def main():
    """Función principal para ejecutar el backtester"""
    backtester = HRMBacktester()
    await backtester.run_backtest()

if __name__ == "__main__":
    asyncio.run(main())