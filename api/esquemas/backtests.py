# Backtests - Resultados de backtesting histórico

from datetime import datetime, timedelta
from typing import Literal, Optional, Dict, Any, List
from pydantic import BaseModel


class TradeBacktest(BaseModel):
    """Trade individual en un backtest."""
    trade_id: str
    fecha: datetime
    simbolo: str
    lado: Literal["buy", "sell"]
    precio_entrada: float
    precio_salida: float
    cantidad: float
    comision: float
    pnl: float
    pnl_pct: float
    duración: timedelta
    régimen_mercado: str = "unknown"


class EstadisticasBacktest(BaseModel):
    """Estadísticas completas de un backtest."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    expectancy: float = 0.0


class ReporteBacktest(BaseModel):
    """Reporte completo de backtest."""
    nombre: str = "HRM Backtest"
    fecha_inicio: datetime
    fecha_fin: datetime
    periodo: str  # e.g., "1 year", "6 months"
    inicial_capital: float = 500.0
    capital_final: float = 500.0
    retorno_total: float = 0.0
    retorno_anualizado: float = 0.0
    estadísticas: EstadisticasBacktest = EstadisticasBacktest()
    lista_trades: List[TradeBacktest] = []
    parametros: Dict[str, Any] = {}
    creado_en: datetime = datetime.utcnow


class BacktestComparacion(BaseModel):
    """Comparación de múltiples backtests."""
    id: str
    nombre_a: str
    nombre_b: str
    metricas_comparadas: List[str] = []
    ganador: Optional[str] = None
    diferencia: Dict[str, float] = {}
    evaluado_en: datetime = datetime.utcnow