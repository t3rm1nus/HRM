import logging
import pandas as pd
import csv
import aiofiles
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import json
import asyncio

class PersistentLogger:
    def __init__(self, log_dir: str = "data/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Archivos de log CSV
        self.trades_file = self.log_dir / "trades_history.csv"
        self.signals_file = self.log_dir / "signals_history.csv" 
        self.market_file = self.log_dir / "market_data.csv"
        self.performance_file = self.log_dir / "performance.csv"
        self.cycles_file = self.log_dir / "cycles_history.csv"
        
        self._init_files()
    
    def _init_files(self):
        """Inicializar archivos CSV con headers si no existen."""
        
        # Trades ejecutados
        if not self.trades_file.exists():
            self._write_csv_header(self.trades_file, [
                'timestamp', 'cycle_id', 'symbol', 'side', 'quantity', 
                'entry_price', 'exit_price', 'status', 'signal_id', 
                'execution_id', 'slippage', 'fees', 'pnl', 'duration_ms',
                'stop_loss', 'take_profit', 'strategy'
            ])
        
        # Señales generadas
        if not self.signals_file.exists():
            self._write_csv_header(self.signals_file, [
                'timestamp', 'cycle_id', 'symbol', 'side', 'confidence', 
                'quantity', 'stop_loss', 'take_profit', 'signal_id', 
                'strategy', 'ai_score', 'tech_score', 'risk_score',
                'ensemble_decision', 'market_regime'
            ])
        
        # Datos de mercado
        if not self.market_file.exists():
            self._write_csv_header(self.market_file, [
                'timestamp', 'symbol', 'price', 'volume', 'high', 'low',
                'open', 'close', 'spread', 'liquidity', 'volatility',
                'rsi', 'macd', 'bollinger_upper', 'bollinger_lower'
            ])
        
        # Métricas de performance
        if not self.performance_file.exists():
            self._write_csv_header(self.performance_file, [
                'timestamp', 'cycle_id', 'portfolio_value', 'total_exposure',
                'btc_exposure', 'eth_exposure', 'cash_balance', 'total_pnl',
                'daily_pnl', 'win_rate', 'sharpe_ratio', 'max_drawdown',
                'correlation_btc_eth', 'signals_count', 'trades_count'
            ])
        
        # Historial de ciclos
        if not self.cycles_file.exists():
            self._write_csv_header(self.cycles_file, [
                'timestamp', 'cycle_id', 'duration_ms', 'signals_generated',
                'orders_executed', 'market_condition', 'btc_price', 'eth_price',
                'total_operations', 'successful_operations', 'failed_operations'
            ])
    
    def _write_csv_header(self, file_path: Path, headers: List[str]):
        """Escribir headers en archivo CSV."""
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    async def log_trade(self, trade_data: Dict[str, Any]):
        """Loggear una operación ejecutada."""
        try:
            row_data = [
                trade_data.get('timestamp', datetime.now().isoformat()),
                trade_data.get('cycle_id', 0),
                trade_data.get('symbol', ''),
                trade_data.get('side', ''),
                trade_data.get('quantity', 0),
                trade_data.get('entry_price', 0),
                trade_data.get('exit_price', 0),
                trade_data.get('status', ''),
                trade_data.get('signal_id', ''),
                trade_data.get('execution_id', ''),
                trade_data.get('slippage', 0),
                trade_data.get('fees', 0),
                trade_data.get('pnl', 0),
                trade_data.get('duration_ms', 0),
                trade_data.get('stop_loss', 0),
                trade_data.get('take_profit', 0),
                trade_data.get('strategy', '')
            ]
            
            async with aiofiles.open(self.trades_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                await writer.writerow(row_data)
                
            logging.info(f"✅ Trade logged: {trade_data.get('symbol')} {trade_data.get('side')}")
            
        except Exception as e:
            logging.error(f"❌ Error logging trade: {e}")
    
    async def log_signal(self, signal_data: Dict[str, Any]):
        """Loggear una señal generada."""
        try:
            row_data = [
                signal_data.get('timestamp', datetime.now().isoformat()),
                signal_data.get('cycle_id', 0),
                signal_data.get('symbol', ''),
                signal_data.get('side', ''),
                signal_data.get('confidence', 0),
                signal_data.get('quantity', 0),
                signal_data.get('stop_loss', 0),
                signal_data.get('take_profit', 0),
                signal_data.get('signal_id', ''),
                signal_data.get('strategy', ''),
                signal_data.get('ai_score', 0),
                signal_data.get('tech_score', 0),
                signal_data.get('risk_score', 0),
                signal_data.get('ensemble_decision', ''),
                signal_data.get('market_regime', '')
            ]
            
            async with aiofiles.open(self.signals_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                await writer.writerow(row_data)
                
            logging.info(f"📊 Signal logged: {signal_data.get('symbol')} {signal_data.get('side')}")
            
        except Exception as e:
            logging.error(f"❌ Error logging signal: {e}")
    
    async def log_market_data(self, market_data: Dict[str, Any]):
        """Loggear datos de mercado."""
        try:
            timestamp = datetime.now().isoformat()
            symbol = market_data.get('symbol', '')
            
            row_data = [
                timestamp,
                symbol,
                market_data.get('price', 0),
                market_data.get('volume', 0),
                market_data.get('high', 0),
                market_data.get('low', 0),
                market_data.get('open', 0),
                market_data.get('close', 0),
                market_data.get('spread', 0),
                market_data.get('liquidity', 0),
                market_data.get('volatility', 0),
                market_data.get('rsi', 0),
                market_data.get('macd', 0),
                market_data.get('bollinger_upper', 0),
                market_data.get('bollinger_lower', 0)
            ]
            
            async with aiofiles.open(self.market_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                await writer.writerow(row_data)
                
        except Exception as e:
            logging.error(f"❌ Error logging market data: {e}")
    
    async def log_performance(self, performance_data: Dict[str, Any]):
        """Loggear métricas de performance."""
        try:
            row_data = [
                performance_data.get('timestamp', datetime.now().isoformat()),
                performance_data.get('cycle_id', 0),
                performance_data.get('portfolio_value', 0),
                performance_data.get('total_exposure', 0),
                performance_data.get('btc_exposure', 0),
                performance_data.get('eth_exposure', 0),
                performance_data.get('cash_balance', 0),
                performance_data.get('total_pnl', 0),
                performance_data.get('daily_pnl', 0),
                performance_data.get('win_rate', 0),
                performance_data.get('sharpe_ratio', 0),
                performance_data.get('max_drawdown', 0),
                performance_data.get('correlation_btc_eth', 0),
                performance_data.get('signals_count', 0),
                performance_data.get('trades_count', 0)
            ]
            
            async with aiofiles.open(self.performance_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                await writer.writerow(row_data)
                
        except Exception as e:
            logging.error(f"❌ Error logging performance: {e}")
    
    async def log_cycle(self, cycle_data: Dict[str, Any]):
        """Loggear información del ciclo."""
        try:
            row_data = [
                cycle_data.get('timestamp', datetime.now().isoformat()),
                cycle_data.get('cycle_id', 0),
                cycle_data.get('duration_ms', 0),
                cycle_data.get('signals_generated', 0),
                cycle_data.get('orders_executed', 0),
                cycle_data.get('market_condition', ''),
                cycle_data.get('btc_price', 0),
                cycle_data.get('eth_price', 0),
                cycle_data.get('total_operations', 0),
                cycle_data.get('successful_operations', 0),
                cycle_data.get('failed_operations', 0)
            ]
            
            async with aiofiles.open(self.cycles_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                await writer.writerow(row_data)
                
            logging.info(f"🔄 Cycle {cycle_data.get('cycle_id')} logged")
            
        except Exception as e:
            logging.error(f"❌ Error logging cycle: {e}")
    
    async def log_state(self, state_data: Dict[str, Any], cycle_id: int):
        """Loggear estado completo del sistema."""
        try:
            timestamp = datetime.now().isoformat()
            state_filename = self.log_dir / f"state_{cycle_id}_{timestamp.replace(':', '-')}.json"
            
            async with aiofiles.open(state_filename, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(state_data, indent=2, default=str))
                
        except Exception as e:
            logging.error(f"❌ Error logging state: {e}")
    
    def get_log_stats(self) -> Dict[str, int]:
        """Obtener estadísticas de los logs."""
        stats = {}
        files = [
            self.trades_file, self.signals_file, self.market_file,
            self.performance_file, self.cycles_file
        ]
        
        for file in files:
            if file.exists():
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        reader = csv.reader(f)
                        rows = list(reader)
                        stats[file.stem] = len(rows) - 1  # Excluir header
                except:
                    stats[file.stem] = 0
            else:
                stats[file.stem] = 0
                
        return stats

# Instancia global
persistent_logger = PersistentLogger()