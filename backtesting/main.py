#!/backtesting/main.py   & .\.venv\Scripts\python.exe -m backtesting.main
"""# Test Gemini model
$env:L2_MODEL = "gemini"; python -m backtesting.main
$env:L2_MODEL = "grok"; python -m backtesting.main
$env:L2_MODEL = "claude"; python -m backtesting.main

"""

import os
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.logging import logger  # Logger centralizado

# Imports relativos dentro del paquete backtesting
try:
    from .getdata import BinanceDataCollector
    from .hrm_tester import HRMStrategyTester
    from .performance_analyzer import PerformanceAnalyzer
    from .report_generator import ReportGenerator
except ImportError:
    # Fallback for direct execution
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
            'l3_models': {},
            'trades': []
        }
        
    def _load_config(self, config_path: str) -> Dict:
        """Carga configuración y aplica defaults para claves faltantes."""
        defaults = {
            'binance': {
                'symbols': ['BTCUSDT', 'ETHUSDT'],
                'intervals': ['1m', '5m', '15m', '1h'],
                'historical_days': 1,
                'api_key': '',
                'api_secret': '',
                'testnet': True,
            },
            'testing': {
                'mode': 'full',
                'lookback_days': 7,
                'symbols': ['BTCUSDT', 'ETHUSDT'],
                'intervals': ['5m', '15m', '1h'],
                'start_date': None,
                'end_date': None,
                'initial_capital': 3000.0,
                'position_size': 0.15,
                'confianza_minima': 0.65,
            },
            'analysis': {
                'generate_charts': True,
                'export_trades': True,
                'metrics': ['sharpe', 'drawdown', 'win_rate'],
            },
            'reporting': {
                'output_dir': 'backtesting/results',
                'generate_charts': True,
                'detailed_logs': True,
                'export_trades': True,
            },
        }

        def _merge(user: Dict, base: Dict) -> Dict:
            merged = dict(base)
            for k, v in (user or {}).items():
                if isinstance(v, dict) and isinstance(base.get(k), dict):
                    merged[k] = _merge(v, base[k])
                else:
                    merged[k] = v
            return merged

        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_cfg = json.load(f)
                return _merge(user_cfg, defaults)
            except Exception:
                logging.warning("Config inválida, usando defaults.")
                return defaults
        else:
            logging.warning("Archivo de configuración no encontrado, usando defaults.")
            return defaults

    def setup_logging(self):
        """Integra el logger centralizado y define self.logger."""
        try:
            self.logger = logger
            self.logger.info("Logging configured successfully.")
        except Exception:
            # Fallback básico si fallara el logger central (poco probable)
            logging.basicConfig(level=logging.INFO,
                                format='%(asctime)s | %(levelname)-8s | %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S')
            self.logger = logging.getLogger("backtesting")
            self.logger.info("Logging configured (fallback).")

    async def run_backtest(self):
        """Ejecuta el proceso de backtesting de punta a punta"""
        self.logger.info("Iniciando backtesting HRM...")
        
        try:
            # 1. Recolectar datos
            tcfg = self.config['testing']
            symbols = tcfg.get('symbols') or self.config['binance']['symbols']
            intervals = tcfg.get('intervals') or self.config['binance']['intervals']
            start_date = tcfg.get('start_date')
            end_date = tcfg.get('end_date')
            if not (start_date and end_date):
                # Usar todo el rango disponible del parquet (últimos 5 años)
                end_date = '2025-09-08'  # Fecha actual
                start_date = '2020-09-08'  # 5 años atrás

            historical_data = await self.data_collector.collect_historical_data(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                intervals=intervals,
            )

            if not historical_data:
                self.logger.error("No se pudieron obtener datos históricos. Abortando.")
                return

            # 2. Ejecutar probador de estrategias
            # Ejecutar estrategia (mock/simple) por ahora
            testing_results = await self.strategy_tester.run_hrm_strategy(historical_data)
            
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
            self.logger.exception("Detalles del error:")

    async def run_validation(self):
        cfg = self.config.get('validation', {})
        periods = cfg.get('periods', [])
        cost_profiles = cfg.get('cost_profiles', [])
        if not periods or not cost_profiles:
            self.logger.warning("Validation config incompleta; saltando validación múltiple.")
            return

        summary = []
        for period in periods:
            p_start = period.get('start_date')
            p_end = period.get('end_date')
            for prof in cost_profiles:
                # ajustar perfil en tester
                try:
                    self.strategy_tester.validation_profile = prof
                except Exception:
                    pass
                self.logger.info(f"🔎 Validando periodo {p_start}..{p_end} perfil {prof.get('name')}")
                data = await self.data_collector.collect_historical_data(
                    symbols=self.config['testing'].get('symbols') or self.config['binance']['symbols'],
                    start_date=p_start,
                    end_date=p_end,
                    intervals=self.config['testing'].get('intervals') or self.config['binance']['intervals'],
                )
                res = await self.strategy_tester.run_hrm_strategy(data)
                overall = res.get('overall', {})
                summary.append({
                    'period': f"{p_start}..{p_end}",
                    'profile': prof.get('name'),
                    'trades': overall.get('total_trades', 0),
                    'win_rate': overall.get('win_rate', 0),
                    'total_return': overall.get('total_return', 0),
                })

        # imprimir resumen
        self.logger.info("📋 Resumen validación:")
        for row in summary:
            self.logger.info(f" {row['period']} [{row['profile']}] trades={row['trades']} win={row['win_rate']:.2%} ret={row['total_return']:.2%}")

async def main():
    """Función principal para ejecutar el backtester"""
    backtester = HRMBacktester()
    await backtester.run_backtest()

if __name__ == "__main__":
    asyncio.run(main())
