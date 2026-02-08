"""
Configuración global para el sistema HRM.

✅ USA: config.paper_mode
✅ USA: config.trading.min_order_size
❌ NO USES: config['paper_mode']

Este módulo proporciona acceso tipado a la configuración mientras mantiene
compatibilidad hacia atrás con código existente que usa acceso por ключ.
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Any

# Cargar variables de entorno
from dotenv import load_dotenv
load_dotenv()

# Utility for safe float conversion
def safe_float(value, default=0.0):
    """Convert value to float safely"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# =============================================================================
# DATACLASSES - FUERTEMENTE TIPADOS
# =============================================================================

@dataclass
class RiskConfig:
    """Configuración de riesgo"""
    max_drawdown_limit: float = 0.01
    risk_limit_btc: float = 0.05
    risk_limit_eth: float = 1.0
    exposure_max_btc: float = 0.20
    exposure_max_eth: float = 0.15
    correlation_limit: float = 0.80


@dataclass
class PositionSizingConfig:
    """Configuración de sizing de posiciones"""
    high_confidence: float = 0.03    # 3% para confianza > 0.8
    medium_confidence: float = 0.02  # 2% para confianza > 0.6
    low_confidence: float = 0.01     # 1% para confianza <= 0.6


@dataclass
class RiskLimitsConfig:
    """Límites de riesgo"""
    max_drawdown_pct: float = 10.0
    max_position_size_pct: float = 50.0
    min_capital_requirement_usd: float = 100.0


@dataclass
class ValidationConfig:
    """Configuración de validación"""
    enable_order_size_check: bool = True
    enable_capital_check: bool = True
    enable_position_check: bool = True
    strict_mode: bool = True


@dataclass
class AllocationConfig:
    """Configuración de asignación"""
    dynamic_rebalancing: bool = True
    concentration_limit_pct: float = 30.0
    min_diversification_ratio: float = 0.40


@dataclass
class TradingConfig:
    """Configuración de trading"""
    min_order_size: int = 10          # Mínimo $10 por orden
    max_order_size: int = 100         # Máximo $100 por orden
    risk_per_trade: float = 0.02      # 2% riesgo por trade
    max_portfolio_risk: float = 0.10   # 10% riesgo total
    position_sizing: PositionSizingConfig = field(default_factory=PositionSizingConfig)
    min_order_size_usd: float = 10.0
    max_allocation_per_symbol_pct: float = 30.0
    available_trading_capital_pct: float = 80.0
    cash_reserve_pct: float = 20.0
    trading_fee_rate: float = 0.001
    max_daily_trades: int = 10
    risk_limits: RiskLimitsConfig = field(default_factory=RiskLimitsConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    allocation: AllocationConfig = field(default_factory=AllocationConfig)


@dataclass
class SignalsConfig:
    """Configuración de señales"""
    universe: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    technical_threshold: float = 0.1
    finrl_threshold: float = 0.2
    mean_reversion_threshold: float = 0.3
    min_signal_strength: float = 0.4
    enabled_generators: List[str] = field(default_factory=lambda: ["technical", "finrl"])


@dataclass
class FINRLConfig:
    """Configuración de FINRL"""
    model_path: str = "models/L2/ai_model_data_multiasset.zip"


@dataclass
class HRMAppConfig:
    """
    Configuración principal de la aplicación HRM.
    
    ✅ Acceso tipado: config.paper_mode
    ✅ Acceso tipado: config.trading.min_order_size
    ✅ Acceso legacy: config['paper_mode'] (mantiene compatibilidad)
    """
    # Modo de operación
    paper_mode: bool = True
    
    # Símbolos
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    
    # Balances iniciales simulados
    simulated_initial_balances: Dict[str, float] = field(
        default_factory=lambda: {"BTC": 0.01549, "ETH": 0.385, "USDT": 3000.0}
    )
    
    # API Keys
    binance_api_key: str = ""
    binance_api_secret: str = ""
    
    # Modo exchange
    use_testnet: bool = True
    mode: str = "TESTNET"
    
    # Sub-configuraciones
    risk_config: RiskConfig = field(default_factory=RiskConfig)
    finrl_config: FINRLConfig = field(default_factory=FINRLConfig)
    signals: SignalsConfig = field(default_factory=SignalsConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    
    # Integración L2-L3
    l2_l3_integration: bool = True
    l3_veto_enabled: bool = True
    
    # =====================================================================
    # COMPATIBILIDAD LEGACY - Permite config['key']
    # =====================================================================
    
    def __getitem__(self, key: str) -> Any:
        """
        Acceso estilo diccionario para compatibilidad legacy.
        ✅ Recomienda usar: config.paper_mode
        """
        # Buscar en atributos principales
        if hasattr(self, key):
            return getattr(self, key)
        
        # Buscar en sub-configuraciones
        sub_configs = ['risk_config', 'finrl_config', 'signals', 'trading']
        for sub in sub_configs:
            sub_obj = getattr(self, sub)
            if hasattr(sub_obj, key):
                return getattr(sub_obj, key)
            # Buscar en sub-sub-configuraciones
            if hasattr(sub_obj, 'position_sizing') and hasattr(sub_obj.position_sizing, key):
                return getattr(sub_obj.position_sizing, key)
            if hasattr(sub_obj, 'risk_limits') and hasattr(sub_obj.risk_limits, key):
                return getattr(sub_obj.risk_limits, key)
            if hasattr(sub_obj, 'validation') and hasattr(sub_obj.validation, key):
                return getattr(sub_obj.validation, key)
            if hasattr(sub_obj, 'allocation') and hasattr(sub_obj.allocation, key):
                return getattr(sub_obj.allocation, key)
        
        raise KeyError(f"Config key not found: {key}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get con valor por defecto"""
        try:
            return self[key]
        except KeyError:
            return default
    
    def setdefault(self, key: str, default: Any) -> Any:
        """
        Set default value if key doesn't exist (like dict.setdefault).
        Returns the value for the key.
        """
        if key not in self:
            # Intentar establecer el valor usando __setattr__
            # Pero los dataclasses frozen no permiten esto, así que solo retornamos el default
            pass
        return self.get(key, default)
    
    def __contains__(self, key: str) -> bool:
        """Verificar si existe una key"""
        try:
            self[key]
            return True
        except KeyError:
            return False
    
    def __repr__(self) -> str:
        return (
            f"HRMAppConfig(paper_mode={self.paper_mode}, "
            f"symbols={self.symbols}, "
            f"use_testnet={self.use_testnet})"
        )


# =============================================================================
# INSTANCIA GLOBAL
# =============================================================================

def _create_config() -> HRMAppConfig:
    """Crear instancia de configuración desde variables de entorno"""
    return HRMAppConfig(
        paper_mode=os.getenv("PAPER_MODE", "true").lower() == "true",
        symbols=os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT").split(','),
        binance_api_key=os.getenv("BINANCE_API_KEY") or "",
        binance_api_secret=os.getenv("BINANCE_API_SECRET") or "",
        use_testnet=os.getenv("USE_TESTNET", "false").lower() == "true",
        mode=os.getenv("BINANCE_MODE", "TESTNET")
    )


# ✅ Instancia global - USA ESTA
config: HRMAppConfig = _create_config()

# Para compatibilidad legacy con código que espera variables sueltas
PAPER_MODE = config.paper_mode
SYMBOLS = config.symbols
USE_TESTNET = config.use_testnet
MODE = config.mode
APAGAR_L3 = False  # L3 habilitado para señales más fuertes


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'config',                    # ✅ Objeto tipado (USA ESTO)
    'HRMAppConfig',              # Clase para type hints
    'TradingConfig',            # Sub-configuración
    'RiskConfig',               # Sub-configuración
    'SignalsConfig',            # Sub-configuración
    'PAPER_MODE',               # Variable suelta (legacy)
    'SYMBOLS',                  # Variable suelta (legacy)
    'USE_TESTNET',              # Variable suelta (legacy)
    'MODE',                     # Variable suelta (legacy)
    'APAGAR_L3',                # Variable suelta
]
