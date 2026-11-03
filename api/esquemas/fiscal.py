# Fiscal - Información fiscal y tributaria

from datetime import datetime
from typing import Literal, Optional, Dict, Any, List
from pydantic import BaseModel


class TransaccionFiscal(BaseModel):
    """Transacción para fines fiscales."""
    transaction_id: str
    fecha: datetime
    simbolo: str
    lado: Literal["buy", "sell"]
    cantidad: float
    precio_entrada: float
    precio_salida: float
    comision: float
    pnl: float
    pnl_pct: float
    holding_period: Optional[float] = None
    crypto_tax_id: Optional[str] = None


class ResumenFiscal(BaseModel):
    """Resumen fiscal de un período."""
    periodo_inicio: datetime
    periodo_fin: datetime
    ganancias_capital: float = 0.0
    perdidas_capital: float = 0.0
    ganancias_no_realizadas: float = 0.0
    perdidas_no_realizadas: float = 0.0
    total_comisiones: float = 0.0
    total_trades: int = 0
    trades_ganadores: int = 0
    trades_perdedores: int = 0
    win_rate: float = 0.0
    fecha_generacion: datetime = datetime.utcnow


class DataFiscalExport(BaseModel):
    """Datos listos para exportar."""
    resumen: ResumenFiscal
    transacciones: List[TransaccionFiscal]
    formato: Literal["csv", "excel", "pdf", "json"]
    creado_en: datetime = datetime.utcnow