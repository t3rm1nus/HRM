"""
Configuración para el módulo L2_tactic
=====================================

Parámetros configurables para señales, sizing, riesgo y modelo IA.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class AIModelConfig:
    model_name: str = "modeloL2"
    model_params: Dict[str, Any] = field(default_factory=dict)
    signal_horizon_minutes: int = 5
    """Configuración del modelo de IA"""
    model_path: str = "models/ai_model_data"
    # Cambiar el tipo de modelo de "sklearn" a "stable_baselines3"
    model_type: str = "stable_baselines3"
    prediction_threshold: float = 0.6
    max_batch_size: int = 100
    cache_predictions: bool = True
    cache_ttl_seconds: int = 300
    fallback_enabled: bool = True
    preprocessing_config: Dict = field(default_factory=dict)


@dataclass
class SignalConfig:
    """Configuración de generación de señales"""
    # Umbrales de fuerza para diferentes acciones
    min_signal_strength: float = 0.3
    strong_signal_threshold: float = 0.7
    signal_expiry_minutes: int = 15
    
    # Pesos para composición de señales
    ai_model_weight: float = 0.6
    technical_weight: float = 0.3
    pattern_weight: float = 0.1
    
    # Filtros de calidad
    min_confidence: float = 0.5
    require_volume_confirmation: bool = True
    max_conflicting_signals: int = 2


@dataclass
class PositionSizingConfig:
    """Configuración de position sizing"""
    # Kelly Criterion
    kelly_fraction: float = 0.25  # Fracción conservadora del Kelly óptimo
    max_kelly_fraction: float = 0.5
    min_kelly_fraction: float = 0.05
    
    # Vol-targeting
    target_volatility: float = 0.15  # 15% anualizado
    lookback_days: int = 20
    vol_adjustment_factor: float = 1.0
    
    # Límites por operación
    max_position_pct: float = 0.10  # 10% del capital por operación
    min_position_usd: float = 50.0
    max_position_usd: float = 10000.0
    
    # Ajustes por liquidez
    liquidity_penalty_threshold: float = 0.5  # Score mínimo de liquidez
    liquidity_size_reduction: float = 0.5  # Reducir tamaño si baja liquidez


@dataclass
class RiskConfig:
    """Configuración de controles de riesgo"""
    # Stop loss
    default_stop_pct: float = 0.02  # 2% stop loss por defecto
    max_stop_pct: float = 0.05  # 5% stop loss máximo
    trailing_stop_enabled: bool = True
    trailing_stop_pct: float = 0.01  # 1% trailing
    
    # Take profit
    default_rr_ratio: float = 2.0  # Risk:Reward 1:2
    max_rr_ratio: float = 5.0
    partial_profit_enabled: bool = True
    partial_profit_pct: float = 0.5  # Tomar 50% en primer target
    
    # Correlación
    max_correlation: float = 0.7  # Correlación máxima entre posiciones
    correlation_lookback: int = 30  # días para calcular correlación
    
    # Límites de exposición
    max_single_asset_exposure: float = 0.20  # 20% en un activo
    max_sector_exposure: float = 0.50  # 50% en un sector (si aplica)
    
    # Drawdown
    position_dd_limit: float = 0.10  # 10% drawdown máximo por posición
    daily_loss_limit: float = 0.05  # 5% pérdida diaria máxima


@dataclass
class ProcessingConfig:
    """Configuración de procesamiento"""
    update_interval_seconds: int = 60  # Actualizar cada minuto
    batch_processing: bool = True
    max_concurrent_signals: int = 50
    signal_queue_size: int = 1000
    
    # Timeouts
    model_prediction_timeout: int = 30  # segundos
    risk_calculation_timeout: int = 10  # segundos
    
    # Logging
    log_level: str = "INFO"
    log_signals: bool = True
    log_performance: bool = True


@dataclass
class L2Config:
    ai_model: AIModelConfig = field(default_factory=AIModelConfig)
    signals: SignalConfig = field(default_factory=SignalConfig)
    position_sizing: PositionSizingConfig = field(default_factory=PositionSizingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    
    # Paths
    base_path: str = "l2_tactic"
    data_path: str = "data"
    models_path: str = "models"
    cache_path: str = "cache"
    
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
        if os.getenv('L2_MAX_POSITION_PCT'):
            config.position_sizing.max_position_pct = float(os.getenv('L2_MAX_POSITION_PCT'))
            
        # Risk config desde env
        if os.getenv('L2_DEFAULT_STOP_PCT'):
            config.risk.default_stop_pct = float(os.getenv('L2_DEFAULT_STOP_PCT'))
        if os.getenv('L2_MAX_CORRELATION'):
            config.risk.max_correlation = float(os.getenv('L2_MAX_CORRELATION'))
            
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
            
        # Crear instancia básica
        config = cls()
        
        # Actualizar con datos del archivo
        if 'ai_model' in data:
            for key, value in data['ai_model'].items():
                if hasattr(config.ai_model, key):
                    setattr(config.ai_model, key, value)
                    
        if 'signals' in data:
            for key, value in data['signals'].items():
                if hasattr(config.signals, key):
                    setattr(config.signals, key, value)
                    
        if 'position_sizing' in data:
            for key, value in data['position_sizing'].items():
                if hasattr(config.position_sizing, key):
                    setattr(config.position_sizing, key, value)
                    
        if 'risk' in data:
            for key, value in data['risk'].items():
                if hasattr(config.risk, key):
                    setattr(config.risk, key, value)
                    
        if 'processing' in data:
            for key, value in data['processing'].items():
                if hasattr(config.processing, key):
                    setattr(config.processing, key, value)
        
        return config
    
    def validate(self) -> List[str]:
        """Valida la configuración y retorna lista de errores"""
        errors = []
        
        # Validar AI model
        if not Path(self.ai_model.model_path).exists():
            errors.append(f"AI model path does not exist: {self.ai_model.model_path}")
            
        if not 0.0 <= self.ai_model.prediction_threshold <= 1.0:
            errors.append("prediction_threshold debe estar entre 0.0 y 1.0")
            
        # Validar signals
        if not 0.0 <= self.signals.min_signal_strength <= 1.0:
            errors.append("min_signal_strength debe estar entre 0.0 y 1.0")
            
        total_weight = (self.signals.ai_model_weight + 
                       self.signals.technical_weight + 
                       self.signals.pattern_weight)
        if abs(total_weight - 1.0) > 0.01:
            errors.append(f"Los pesos de señales deben sumar 1.0, actual: {total_weight}")
            
        # Validar position sizing
        if self.position_sizing.kelly_fraction <= 0:
            errors.append("kelly_fraction debe ser positivo")
            
        if self.position_sizing.max_position_pct <= 0 or self.position_sizing.max_position_pct > 1.0:
            errors.append("max_position_pct debe estar entre 0.0 y 1.0")
            
        # Validar risk
        if self.risk.default_stop_pct <= 0:
            errors.append("default_stop_pct debe ser positivo")
            
        if self.risk.default_rr_ratio <= 0:
            errors.append("default_rr_ratio debe ser positivo")
            
        return errors


# Instancia por defecto
DEFAULT_L2_CONFIG = L2Config()