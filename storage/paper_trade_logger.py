"""
Paper Trade Logger - Sistema de persistencia para trades simulados/paper
Guarda trades de prueba en archivos separados para análisis posterior
"""

import os
import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from core.logging import logger

class PaperTradeLogger:
    """
    Logger especializado para guardar trades simulados/paper.
    Separa completamente los trades de prueba de los reales.
    """

    def __init__(self, base_dir: str = "data/paper_trades", clear_on_init: bool = False):
        """
        Inicializa el logger de paper trades

        Args:
            base_dir: Directorio base para guardar los trades
            clear_on_init: Si True, limpia todos los trades existentes al inicializar
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Archivos de salida
        self.trades_csv = self.base_dir / "paper_trades.csv"
        self.trades_json = self.base_dir / "paper_trades.json"
        self.summary_json = self.base_dir / "summary.json"

        # Estado en memoria
        self.trades = []
        self.daily_stats = {}
        self.session_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'total_fees': 0.0,
            'win_rate': 0.0,
            'avg_pnl_per_trade': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'start_time': datetime.now().isoformat(),
            'last_update': datetime.now().isoformat()
        }

        # Limpiar trades existentes si se solicita
        if clear_on_init:
            self._clear_existing_files()

        # Cargar trades existentes si los hay (solo si no se limpiaron)
        if not clear_on_init:
            self._load_existing_trades()

        logger.info(f"✅ PaperTradeLogger inicializado - Guardando en: {self.base_dir}")

    def _load_existing_trades(self):
        """Carga trades existentes desde archivos"""
        try:
            # Cargar desde JSON si existe
            if self.trades_json.exists():
                with open(self.trades_json, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.trades = data.get('trades', [])
                    self.session_stats.update(data.get('session_stats', {}))
                logger.info(f"📂 Cargados {len(self.trades)} trades existentes desde JSON")
                return

            # Cargar desde CSV si existe
            if self.trades_csv.exists():
                self.trades = []
                with open(self.trades_csv, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Convertir tipos de datos
                        trade = {
                            'timestamp': row.get('timestamp'),
                            'symbol': row.get('symbol'),
                            'side': row.get('side'),
                            'quantity': float(row.get('quantity', 0)),
                            'price': float(row.get('price', 0)),
                            'filled_price': float(row.get('filled_price', 0)),
                            'commission': float(row.get('commission', 0)),
                            'pnl': float(row.get('pnl', 0)),
                            'status': row.get('status', 'filled'),
                            'order_id': row.get('order_id'),
                            'cycle_id': row.get('cycle_id'),
                            'strategy': row.get('strategy', 'paper'),
                            'reason': row.get('reason', '')
                        }
                        self.trades.append(trade)
                logger.info(f"📂 Cargados {len(self.trades)} trades existentes desde CSV")

        except Exception as e:
            logger.warning(f"⚠️ Error cargando trades existentes: {e}")
            self.trades = []

    def _clear_existing_files(self):
        """Limpia todos los archivos de trades existentes"""
        try:
            files_to_clear = [self.trades_csv, self.trades_json, self.summary_json]
            cleared_count = 0

            for file_path in files_to_clear:
                if file_path.exists():
                    file_path.unlink()  # Eliminar archivo
                    cleared_count += 1

            if cleared_count > 0:
                logger.info(f"🧹 Limpiados {cleared_count} archivos de trades existentes para nueva sesión")
            else:
                logger.debug("ℹ️ No había archivos de trades para limpiar")

        except Exception as e:
            logger.warning(f"⚠️ Error limpiando archivos existentes: {e}")

    def log_paper_trade(self, order: Dict[str, Any], market_data: Optional[Dict[str, Any]] = None,
                       cycle_id: Optional[int] = None, strategy: str = "paper"):
        """
        Registra un trade simulado/paper

        Args:
            order: Orden ejecutada (debe tener status='filled')
            market_data: Datos de mercado para cálculos adicionales
            cycle_id: ID del ciclo donde se ejecutó
            strategy: Estrategia que generó el trade
        """
        try:
            # Validar que es una orden ejecutada
            if order.get('status') != 'filled':
                logger.debug(f"⚠️ Orden no ejecutada, ignorando: {order.get('status')}")
                return

            # Extraer datos de la orden
            symbol = order.get('symbol', 'UNKNOWN')
            side = order.get('side', 'unknown')
            quantity = abs(float(order.get('quantity', 0)))
            filled_price = float(order.get('filled_price', 0))
            commission = float(order.get('commission', 0))

            # Calcular PnL si es una venta (closing trade)
            pnl = 0.0
            if side.lower() == 'sell' and market_data:
                # Para ventas, calcular PnL basado en posición previa
                # Esto es simplificado - en un sistema real tendrías que trackear posiciones
                pnl = 0.0  # Por ahora, las ventas no generan PnL calculado

            # Crear registro del trade
            trade = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': filled_price,  # Precio de ejecución
                'filled_price': filled_price,
                'commission': commission,
                'pnl': pnl,
                'status': 'filled',
                'order_id': order.get('order_id', f"paper_{int(datetime.now().timestamp())}"),
                'cycle_id': cycle_id,
                'strategy': strategy,
                'reason': order.get('reason', ''),
                'market_conditions': self._extract_market_conditions(market_data) if market_data else {}
            }

            # Agregar a la lista
            self.trades.append(trade)

            # Actualizar estadísticas
            self._update_stats(trade)

            # Guardar inmediatamente
            self._save_trade(trade)

            logger.info(f"📝 Paper Trade Logged: {symbol} {side} {quantity:.6f} @ {filled_price:.2f} (fee: {commission:.4f})")

        except Exception as e:
            logger.error(f"❌ Error logging paper trade: {e}")

    def _extract_market_conditions(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extrae condiciones de mercado relevantes"""
        conditions = {}

        try:
            for symbol, data in market_data.items():
                if isinstance(data, dict) and 'close' in data:
                    conditions[f"{symbol}_price"] = data['close']
                elif isinstance(data, (dict,)) and len(data) > 0:
                    # Handle other formats
                    if 'close' in data:
                        conditions[f"{symbol}_price"] = data['close']
        except:
            pass

        return conditions

    def _update_stats(self, trade: Dict[str, Any]):
        """Actualiza estadísticas de sesión"""
        self.session_stats['total_trades'] += 1
        self.session_stats['total_fees'] += trade.get('commission', 0)
        self.session_stats['total_pnl'] += trade.get('pnl', 0)
        self.session_stats['last_update'] = datetime.now().isoformat()

        # Actualizar win/loss si hay PnL
        pnl = trade.get('pnl', 0)
        if pnl > 0:
            self.session_stats['winning_trades'] += 1
            self.session_stats['largest_win'] = max(self.session_stats['largest_win'], pnl)
        elif pnl < 0:
            self.session_stats['losing_trades'] += 1
            self.session_stats['largest_loss'] = min(self.session_stats['largest_loss'], pnl)

        # Calcular métricas derivadas
        total_trades = self.session_stats['total_trades']
        if total_trades > 0:
            self.session_stats['win_rate'] = (self.session_stats['winning_trades'] / total_trades) * 100
            self.session_stats['avg_pnl_per_trade'] = self.session_stats['total_pnl'] / total_trades

    def _save_trade(self, trade: Dict[str, Any]):
        """Guarda un trade individual en archivos"""
        try:
            # Guardar en CSV
            self._append_to_csv(trade)

            # Guardar JSON completo (sobrescribe)
            self._save_json()

        except Exception as e:
            logger.error(f"❌ Error guardando trade: {e}")

    def _append_to_csv(self, trade: Dict[str, Any]):
        """Agrega trade al archivo CSV"""
        try:
            file_exists = self.trades_csv.exists()

            with open(self.trades_csv, 'a', newline='', encoding='utf-8') as f:
                fieldnames = [
                    'timestamp', 'symbol', 'side', 'quantity', 'price', 'filled_price',
                    'commission', 'pnl', 'status', 'order_id', 'cycle_id', 'strategy', 'reason'
                ]

                writer = csv.DictWriter(f, fieldnames=fieldnames)

                if not file_exists:
                    writer.writeheader()

                # Preparar datos para CSV (solo campos principales)
                csv_trade = {
                    'timestamp': trade.get('timestamp'),
                    'symbol': trade.get('symbol'),
                    'side': trade.get('side'),
                    'quantity': trade.get('quantity'),
                    'price': trade.get('price'),
                    'filled_price': trade.get('filled_price'),
                    'commission': trade.get('commission'),
                    'pnl': trade.get('pnl'),
                    'status': trade.get('status'),
                    'order_id': trade.get('order_id'),
                    'cycle_id': trade.get('cycle_id'),
                    'strategy': trade.get('strategy'),
                    'reason': trade.get('reason', '')[:100]  # Limitar longitud
                }

                writer.writerow(csv_trade)

        except Exception as e:
            logger.error(f"❌ Error guardando en CSV: {e}")

    def _save_json(self):
        """Guarda todos los trades en JSON"""
        try:
            data = {
                'trades': self.trades,
                'session_stats': self.session_stats,
                'last_updated': datetime.now().isoformat(),
                'total_trades': len(self.trades)
            }

            with open(self.trades_json, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"❌ Error guardando en JSON: {e}")

    def get_session_summary(self) -> Dict[str, Any]:
        """Retorna resumen de la sesión actual"""
        return self.session_stats.copy()

    def get_recent_trades(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retorna los trades más recientes"""
        return self.trades[-limit:] if self.trades else []

    def export_for_analysis(self, output_file: Optional[str] = None) -> str:
        """
        Exporta todos los trades para análisis con el script semanal

        Args:
            output_file: Archivo de salida (opcional)

        Returns:
            Ruta del archivo exportado
        """
        if not output_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = self.base_dir / f"trades_export_{timestamp}.csv"

        try:
            # Crear DataFrame para facilitar exportación
            df = pd.DataFrame(self.trades)

            # Asegurar columnas necesarias
            required_cols = ['timestamp', 'symbol', 'side', 'quantity', 'price', 'realized_pnl', 'commission']
            for col in required_cols:
                if col not in df.columns:
                    if col == 'realized_pnl':
                        df[col] = df.get('pnl', 0)  # Mapear pnl a realized_pnl
                    else:
                        df[col] = ''

            # Seleccionar y renombrar columnas para compatibilidad con el analizador semanal
            export_df = df[required_cols].copy()
            export_df.to_csv(output_file, index=False)

            logger.info(f"📊 Trades exportados para análisis: {output_file}")
            return str(output_file)

        except Exception as e:
            logger.error(f"❌ Error exportando trades: {e}")
            return ""

    def clear_session(self):
        """Limpia la sesión actual (para testing)"""
        self.trades = []
        self.session_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'total_fees': 0.0,
            'win_rate': 0.0,
            'avg_pnl_per_trade': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'start_time': datetime.now().isoformat(),
            'last_update': datetime.now().isoformat()
        }
        logger.info("🧹 Sesión de paper trades limpiada")

    def get_stats_report(self) -> str:
        """Genera reporte de estadísticas formateado"""
        stats = self.session_stats

        report = f"""
📊 PAPER TRADES SESSION REPORT
{'='*50}
Total Trades: {stats['total_trades']}
Winning Trades: {stats['winning_trades']}
Losing Trades: {stats['losing_trades']}
Win Rate: {stats['win_rate']:.2f}%

Financial Summary:
- Total PnL: ${stats['total_pnl']:.2f}
- Total Fees: ${stats['total_fees']:.2f}
- Avg PnL per Trade: ${stats['avg_pnl_per_trade']:.2f}
- Largest Win: ${stats['largest_win']:.2f}
- Largest Loss: ${stats['largest_loss']:.2f}

Session Info:
- Started: {stats['start_time']}
- Last Update: {stats['last_update']}
- Trades File: {self.trades_csv}
"""
        return report


# Función de conveniencia para logging rápido
_paper_logger = None

def get_paper_logger(clear_on_init: bool = False) -> PaperTradeLogger:
    """
    Retorna instancia singleton del logger

    Args:
        clear_on_init: Si True, limpia trades existentes al inicializar (solo primera vez)
    """
    global _paper_logger
    if _paper_logger is None:
        _paper_logger = PaperTradeLogger(clear_on_init=clear_on_init)
    return _paper_logger

def log_paper_trade(order: Dict[str, Any], **kwargs):
    """Función de conveniencia para loggear trades paper"""
    logger = get_paper_logger()
    logger.log_paper_trade(order, **kwargs)

# Ejemplo de uso:
"""
from storage.paper_trade_logger import log_paper_trade, get_paper_logger

# Loggear un trade simulado
order = {
    'symbol': 'BTCUSDT',
    'side': 'buy',
    'quantity': 0.001,
    'filled_price': 50000.0,
    'commission': 0.05,
    'status': 'filled',
    'order_id': 'paper_123'
}

log_paper_trade(order, cycle_id=42, strategy='test_strategy')

# Obtener estadísticas
logger = get_paper_logger()
stats = logger.get_session_summary()
print(f"Total trades: {stats['total_trades']}")

# Exportar para análisis semanal
export_file = logger.export_for_analysis()
print(f"Trades exportados: {export_file}")
"""
