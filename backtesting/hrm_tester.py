# /backtesting/hrm_tester.py

import logging
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from backtesting_utils import TestMode, TestLevel, TestResult, L1Model
from report_generator import ReportGenerator
from getdata import BinanceDataCollector
import asyncio

class HRMStrategyTester:
    """Clase principal para ejecutar y evaluar la estrategia HRM"""

    def __init__(self, config: Dict, data_collector: BinanceDataCollector):
        self.config = config
        self.data_collector = data_collector
        self.logger = logging.getLogger(__name__)

    async def run_full_backtest(self):
        # Ya no se accede a self.config['binance'], se usa self.data_collector
        data = await self.data_collector.collect_historical_data(
            symbols=self.config['binance']['symbols'],
            intervals=self.config['binance']['intervals'],
            historical_days=self.config['binance']['historical_days']
        )
        
        # 2. Ejecución de la estrategia
        strategy_results = await self.run_hrm_strategy(data)
        
        # 3. Generación de reportes
        report_results = await self.report_generator.generate_complete_report(strategy_results)
        
        self.logger.info("Backtest completed successfully")
        return report_results

    async def run_hrm_strategy(self, data: Dict) -> Dict:
        """Run the HRM trading strategy on the provided data."""
        self.logger.info("Running HRM strategy...")
        
        try:
            results = {
                'overall': {
                    'total_return': 0.0,
                    'annualized_return': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'win_rate': 0.0,
                    'profit_factor': 0.0,
                    'total_trades': 0
                },
                'l1_models': {},
                'l2_model': {},
                'l3_models': {},
                'trades': []
            }

            for symbol in data:
                for interval in data[symbol]:
                    if isinstance(data[symbol][interval], pd.DataFrame) and not data[symbol][interval].empty:
                        df = data[symbol][interval].copy()
                        df['signal'] = 0
                        df.loc[df['rsi'] < 30, 'signal'] = 1  # Buy
                        df.loc[df['rsi'] > 70, 'signal'] = -1  # Sell

                        trades_temp = []
                        position = 0
                        entry_price = 0
                        entry_timestamp = None
                        
                        for idx, row in df.iterrows():
                            if row['signal'] == 1 and position == 0:
                                entry_price = row['close']
                                entry_timestamp = idx
                                position = 1
                            elif row['signal'] == -1 and position == 1:
                                exit_price = row['close']
                                pnl = exit_price - entry_price
                                position = 0
                                trades_temp.append({
                                    'entry_timestamp': entry_timestamp,
                                    'exit_timestamp': idx,
                                    'symbol': symbol,
                                    'interval': interval,
                                    'entry_price': entry_price,
                                    'exit_price': exit_price,
                                    'pnl': pnl
                                })

                        if trades_temp:
                            results['trades'].extend(trades_temp)
                            total_trades = len(trades_temp)
                            wins = len([t for t in trades_temp if t['pnl'] > 0])
                            losses = len([t for t in trades_temp if t['pnl'] < 0])
                            total_pnl = sum(t['pnl'] for t in trades_temp)
                            
                            results['overall']['total_trades'] += total_trades
                            results['overall']['total_return'] += total_pnl / entry_price if entry_price else 0
                            results['overall']['win_rate'] = wins / total_trades if total_trades > 0 else 0
                            
                            gross_profit = sum(t['pnl'] for t in trades_temp if t['pnl'] > 0)
                            gross_loss = abs(sum(t['pnl'] for t in trades_temp if t['pnl'] < 0))
                            results['overall']['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')

            # Prueba de los 3 modelos L1
            l1_tester = L1Model()
            results['l1_models'] = l1_tester.predict(data)

            # Estas secciones siguen siendo mock data, las dejas aquí para la simulación
            results['l2_model'] = {
                'signal_quality': 0.75, 'sizing_efficiency': 0.80,
                'hit_rate': 0.65, 'risk_adjusted_return': 0.05
            }
            results['l3_models'] = {
                'strategic': {
                    'decision_accuracy': 0.90, 'regime_detection_accuracy': 0.85,
                    'strategic_value': 0.95, 'allocation_efficiency': 0.88
                }
            }
            self.logger.info("HRM strategy execution completed")
            return results

        except Exception as e:
            self.logger.error(f"Error running HRM strategy: {e}")
            raise