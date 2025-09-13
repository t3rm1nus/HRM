# config.py - Configuración para el módulo L2_tactic (adaptado para multiasset: BTC y ETH)

"""
Configuración para el módulo L2_tactic
=====================================

Parámetros configurables para señales, sizing, riesgo y modelo IA.
Adaptado para manejar múltiples activos (BTC/USDT y ETH/USDT).
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Any
from pathlib import Path

# Importamos modelos para consistencia tipada
from .models import PositionSize, RiskMetrics


@dataclass
class AIModelConfig:
    """Configuración del modelo de IA (multiasset)"""
    model_name: str = "modeloL2_multiasset"
    model_params: Dict[str, Any] = field(default_factory=dict)
    signal_horizon_minutes: int = 5
    model_path: str = "models/L2/ai_model_data_multiasset.zip"  # Path correcto para PPO SB3
    model_type: str = "stable_baselines3"  # tipo de modelo
    prediction_threshold: float = 0.3
    max_batch_size: int = 100
    cache_predictions: bool = True
    cache_ttl_seconds: int = 300
    fallback_enabled: bool = True
    preprocessing_config: Dict = field(default_factory=dict)


@dataclass
class SignalConfig:
    """Configuración de generación de señales (multiasset)"""
    min_signal_strength: float = 0.1
    strong_signal_threshold: float = 0.5
    signal_expiry_minutes: int = 15

    # Pesos para composición de señales
    ai_model_weight: float = 0.6  # Era 0.5
    technical_weight: float = 0.3
    pattern_weight: float = 0.1

    # Filtros de calidad
    min_confidence: float = 0.3
    require_volume_confirmation: bool = True
    max_conflicting_signals: int = 2

    # Nuevo: Universo de símbolos
    universe: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT", "USDT"])


@dataclass
class PositionSizingConfig:
    """Configuración de position sizing (multiasset)"""
    # Kelly Criterion
    kelly_fraction: float = 0.25
    max_kelly_fraction: float = 0.5
    min_kelly_fraction: float = 0.05

    # Vol-targeting
    target_volatility: float = 0.15  # 15% anualizado
    lookback_days: int = 20
    vol_adjustment_factor: float = 1.0

    # Límites por operación (específicos por asset) - OPTIMIZADOS
    max_position_pct: Dict[str, float] = field(default_factory=lambda: {
        "BTC/USDT": 0.40,  # Aumentado al 40%
        "ETH/USDT": 0.30   # Aumentado al 30%
    })
    min_position_usd: float = 25.0    # Reducido para más flexibilidad
    max_position_usd: float = 1500.0  # 50% del capital inicial

    # Ajustes por liquidez - MÁS AGRESIVOS
    liquidity_penalty_threshold: float = 0.7   # Más tolerante
    liquidity_size_reduction: float = 0.3      # Menor reducción

    # Capital de referencia actualizado
    capital_total_usd: float = 3000.0


@dataclass
class RiskConfig:
    """Configuración de controles de riesgo (multiasset)"""
    # Stop loss
    default_stop_pct: float = 0.02
    max_stop_pct: float = 0.05
    trailing_stop_enabled: bool = True
    trailing_stop_pct: float = 0.01

    # Take profit
    default_rr_ratio: float = 2.0
    max_rr_ratio: float = 5.0
    partial_profit_enabled: bool = True
    partial_profit_pct: float = 0.5

    # Correlación (cruzada BTC-ETH)
    max_correlation: float = 0.7
    correlation_lookback: int = 30
    max_correlation_btc_eth: float = 0.80  # Nuevo: Límite específico para correlación BTC-ETH

    # Límites de exposición (por asset)
    max_single_asset_exposure: Dict[str, float] = field(default_factory=lambda: {
        "BTC/USDT": 0.20,
        "ETH/USDT": 0.15
    })
    max_sector_exposure: float = 0.50

    # Drawdown
    position_dd_limit: float = 0.10
    daily_loss_limit: float = 0.05

    # Métricas avanzadas
    var_confidence: float = 0.95
    max_expected_vol: float = 0.30


@dataclass
class ProcessingConfig:
    """Configuración de procesamiento (multiasset)"""
    update_interval_seconds: int = 60
    batch_processing: bool = True
    max_concurrent_signals: int = 50
    signal_queue_size: int = 1000

    # Timeouts
    model_prediction_timeout: int = 30
    risk_calculation_timeout: int = 10

    # Logging
    log_level: str = "INFO"
    log_signals: bool = True
    log_performance: bool = True


@dataclass
class BusConfig:
    """Configuración del bus de mensajes (L3↔L2↔L1)"""
    broker_url: str = "mqtt://localhost:1883"
    topics_subscribe: List[str] = field(default_factory=lambda: [
        "signals/regime",
        "signals/universe",
        "allocations/updates"
    ])
    topics_publish: List[str] = field(default_factory=lambda: [
        "signals/tactical",
        "reports/performance"
    ])
    qos_level: int = 1
    reconnect_interval: int = 5


@dataclass
class L2Config:
    # --- Señales ---
    signal_threshold: float = 0.4
    max_signals: int = 20

    # --- Position sizing ---
    kelly_fraction: float = 0.5
    vol_target: float = 0.02

    # --- Riesgo global ---
    max_drawdown: float = 0.15
    max_position_risk: float = 0.02

    # --- Parámetros específicos de control de riesgo ---
    default_stop_pct: float = 0.02           # stop fijo por defecto (2%)
    atr_multiplier: float = 2.0              # multiplicador ATR para stops dinámicos
    trailing_stop_pct: float = 0.01          # trailing stop (1%)
    breakeven_threshold: float = 1.5         # pasar a BE cuando R multiple ≥ 1.5
    take_profit_rr_min: float = 1.5          # TP mínimo en múltiplos de R
    take_profit_rr_max: float = 2.5          # TP máximo en múltiplos de R
    max_correlation: float = 0.7             # correlación máxima permitida entre posiciones
    max_portfolio_heat: float = 0.8          # exposición total máxima (% capital)
    daily_loss_limit: float = 0.05           # pérdida diaria máxima (% capital)
    max_drawdown_limit: float = 0.15         # pérdida máxima acumulada (% capital)
    max_positions: int = 5                   # número máximo de posiciones abiertas
    max_signal_drawdown: float = 0.20        # DD máximo por señal
    max_strategy_drawdown: float = 0.25      # DD máximo por estrategia
    min_liquidity_notional: float = 25_000.0 # notional mínimo de liquidez
    min_liquidity_ratio: float = 0.02        # ratio mínimo de liquidez

    # --- AI Model ---
    ai_model_path: str = "models/l2_model.zip"
    ai_model_timeout_s: int = 3  

    # --- Performance optimizer ---
    max_cache_items: int = 500
    prediction_ttl_s: int = 60
    batch_size: int = 32
    enable_lazy_loading: bool = True
    parallel_workers: int = 4
    feature_ttl_s: int = 300
    thread_name_prefix: str = "L2Perf"
    rate_limit_qps: int = 20

    # --- Pesos para el compositor de señales ---
    ai_model_weight: float = 0.5
    technical_weight: float = 0.3
    pattern_weight: float = 0.2

    # --- Thresholds para validar señal compuesta ---
    min_signal_confidence: float = 0.3
   
    min_signal_strength: float = 0.05

    # --- Subconfiguraciones ---
    ai_model: AIModelConfig = field(default_factory=AIModelConfig)
    signals: SignalConfig = field(default_factory=SignalConfig)
    position_sizing: PositionSizingConfig = field(default_factory=PositionSizingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    bus: BusConfig = field(default_factory=BusConfig)

    # --- Paths ---
    base_path: str = "l2_tactic"
    data_path: str = "data/multiasset"
    models_path: str = "models/multiasset"
    cache_path: str = "cache/multiasset"

    @classmethod
    def from_env(cls) -> 'L2Config':
        """Crea configuración desde variables de entorno"""
        config = cls()

        # AI Model config desde env
        if os.getenv('L2_AI_MODEL_PATH'):
            config.ai_model.model_path = os.getenv('L2_AI_MODEL_PATH')
        if os.getenv('L2_AI_MODEL_TYPE'):
            config.ai_model.model_type = os.getenv('L2_AI_MODEL_TYPE')
        if os.getenv('L2_PREDICTION_THRESHOLD'):
            config.ai_model.prediction_threshold = float(os.getenv('L2_PREDICTION_THRESHOLD'))

        # Signal config desde env
        if os.getenv('L2_MIN_SIGNAL_STRENGTH'):
            config.signals.min_signal_strength = float(os.getenv('L2_MIN_SIGNAL_STRENGTH'))
        if os.getenv('L2_AI_MODEL_WEIGHT'):
            config.signals.ai_model_weight = float(os.getenv('L2_AI_MODEL_WEIGHT'))

        # Position sizing desde env
        if os.getenv('L2_KELLY_FRACTION'):
            config.position_sizing.kelly_fraction = float(os.getenv('L2_KELLY_FRACTION'))
        if os.getenv('L2_MAX_POSITION_PCT_BTC'):
            config.position_sizing.max_position_pct["BTC/USDT"] = float(os.getenv('L2_MAX_POSITION_PCT_BTC'))
        if os.getenv('L2_MAX_POSITION_PCT_ETH'):
            config.position_sizing.max_position_pct["ETH/USDT"] = float(os.getenv('L2_MAX_POSITION_PCT_ETH'))
        if os.getenv('L2_CAPITAL_TOTAL_USD'):
            config.position_sizing.capital_total_usd = float(os.getenv('L2_CAPITAL_TOTAL_USD'))

        # Risk config desde env
        if os.getenv('L2_DEFAULT_STOP_PCT'):
            config.risk.default_stop_pct = float(os.getenv('L2_DEFAULT_STOP_PCT'))
        if os.getenv('L2_MAX_CORRELATION'):
            config.risk.max_correlation = float(os.getenv('L2_MAX_CORRELATION'))
        if os.getenv('L2_MAX_CORRELATION_BTC_ETH'):
            config.risk.max_correlation_btc_eth = float(os.getenv('L2_MAX_CORRELATION_BTC_ETH'))

        # Bus config desde env
        if os.getenv('L2_BUS_BROKER_URL'):
            config.bus.broker_url = os.getenv('L2_BUS_BROKER_URL')

        return config

    @classmethod
    def from_file(cls, config_file: str) -> 'L2Config':
        """Carga configuración desde archivo YAML/JSON"""
        import yaml

        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")

        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)

        config = cls()

        def update(obj, updates: Dict):
            for key, value in updates.items():
                if hasattr(obj, key):
                    setattr(obj, key, value)

        if 'ai_model' in data:
            update(config.ai_model, data['ai_model'])
        if 'signals' in data:
            update(config.signals, data['signals'])
        if 'position_sizing' in data:
            update(config.position_sizing, data['position_sizing'])
        if 'risk' in data:
            update(config.risk, data['risk'])
        if 'processing' in data:
            update(config.processing, data['processing'])
        if 'bus' in data:
            update(config.bus, data['bus'])

        return config

    def get(self, key, default=None):
        return getattr(self, key, default)

    def validate(self) -> List[str]:
        """Valida la configuración y retorna lista de errores"""
        errors = []

        # AI Model
        if not Path(self.ai_model.model_path).exists():
            errors.append(f"AI model path does not exist: {self.ai_model.model_path}")
        if not 0.0 <= self.ai_model.prediction_threshold <= 1.0:
            errors.append("prediction_threshold debe estar entre 0.0 y 1.0")

        # Señales
        if not 0.0 <= self.signals.min_signal_strength <= 1.0:
            errors.append("min_signal_strength debe estar entre 0.0 y 1.0")

        total_weight = (
            self.signals.ai_model_weight +
            self.signals.technical_weight +
            self.signals.pattern_weight
        )
        if abs(total_weight - 1.0) > 0.01:
            errors.append(f"Los pesos de señales deben sumar 1.0, actual: {total_weight}")

        # Position sizing
        if self.position_sizing.kelly_fraction <= 0:
            errors.append("kelly_fraction debe ser positivo")
        for asset, pct in self.position_sizing.max_position_pct.items():
            if not (0 < pct <= 1.0):
                errors.append(f"max_position_pct para {asset} debe estar entre 0.0 y 1.0")
        if self.position_sizing.capital_total_usd <= 0:
            errors.append("capital_total_usd debe ser positivo")

        # Riesgo
        if self.risk.default_stop_pct <= 0:
            errors.append("default_stop_pct debe ser positivo")
        if self.risk.default_rr_ratio <= 0:
            errors.append("default_rr_ratio debe ser positivo")
        if not 0.0 < self.risk.var_confidence < 1.0:
            errors.append("var_confidence debe estar entre 0 y 1")

        # Bus
        if not self.bus.broker_url:
            errors.append("bus.broker_url no puede estar vacío")

        return errors


# Instancia por defecto
DEFAULT_L2_CONFIG = L2Config()