"""
L3 Strategic Level - Strategic Decision Making
===========================================

El nivel L3_Strategic es el componente de más alto nivel del sistema HRM,
responsable del análisis macroeconómico, detección de régimen de mercado,
optimización de cartera y establecimiento de directrices estratégicas.

Módulos principales:
- config: Configuración de parámetros estratégicos
- models: Modelos de datos para análisis estratégico
- macro_analyzer: Análisis de condiciones macroeconómicas
- regime_detector: Detección de régimen de mercado con ML
- portfolio_optimizer: Optimización de cartera (Markowitz, Black-Litterman)
- sentiment_analyzer: Análisis de sentimiento con NLP
- risk_manager: Gestión estratégica de riesgo
- decision_maker: Tomador final de decisiones estratégicas
- data_provider: Proveedor de datos macro y de mercado

Flujo de operación:
1. Recolección de datos macro, de mercado y sentiment
2. Detección de régimen de mercado actual
3. Análisis de riesgo estratégico
4. Optimización de cartera
5. Generación de directrices estratégicas para L2
6. Monitoreo y ajuste continuo

Integración con L2:
- Proporciona régimen de mercado detectado
- Define asignación óptima de activos
- Establece apetito de riesgo
- Envía contexto de mercado y correlaciones
"""

import logging
import warnings
from typing import Optional, Dict, Any, List
from datetime import datetime
from core.logging import logger as core_logger

# Suprimir warnings innecesarios de ML libraries
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- Logging específico L3 ---
def setup_l3_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Configura logging específico para L3
    
    Args:
        log_level: Nivel de logging
        
    Returns:
        Logger configurado
    """
    _logger = logging.getLogger("l3_strategy")
    
    # Evitar múltiples handlers
    if not _logger.hasHandlers():
        # Handler consola
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '[%(asctime)s] L3-%(levelname)s [%(name)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        _logger.addHandler(console_handler)

        # Handler archivo
        try:
            import os
            log_dir = "data/logs/"
            os.makedirs(log_dir, exist_ok=True)
            
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                f"{log_dir}/l3_strategic.log",
                maxBytes=10*1024*1024,
                backupCount=5
            )
            file_handler.setFormatter(formatter)
            _logger.addHandler(file_handler)
        except Exception as e:
            _logger.warning(f"No se pudo configurar logging a archivo: {e}")

    # Nivel de logging
    _logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    _logger.info("Inicializando l3_strategy")
    
    return _logger

# Inicializar logger L3
_logger = setup_l3_logging()

# --- Importaciones principales ---
try:
    from .config import (
        MarketRegime, RiskAppetite, ModelConfig, RiskConfig,
        DataConfig, OptimizationConfig, ExecutionConfig,
        get_config, update_config, DEFAULT_CONFIG, API_KEYS, PATHS
    )
    
    from .models import (
        MacroIndicator, OnChainMetric, SentimentData, MarketData,
        RegimeAnalysis, RiskMetrics, AssetAllocation,
        StrategicGuidelines, MarketContext, StrategicSignal, StrategicPerformance,
        create_default_strategic_signal, validate_allocation, normalize_allocation
    )

    _CONFIG_LOADED = True
    _MODELS_LOADED = True
except ImportError as e:
    _logger.warning(f"No se pudieron importar algunos módulos de L3: {e}")
    _CONFIG_LOADED = False
    _MODELS_LOADED = False

__version__ = "1.0.0"
__author__ = "HRM System"
__status__ = "Development"

MODULE_INFO = {
    "name": "L3_Strategic",
    "version": __version__,
    "description": "Strategic Decision Making Level",
    "status": __status__,
    "config_loaded": _CONFIG_LOADED,
    "models_loaded": _MODELS_LOADED,
    "supported_assets": ["BTC", "ETH", "BNB", "ADA", "SOL"],
    "supported_regimes": [regime.value for regime in MarketRegime] if _CONFIG_LOADED else [],
    "last_updated": datetime.utcnow().isoformat()
}

def get_module_info() -> Dict[str, Any]:
    return MODULE_INFO.copy()

def check_dependencies() -> Dict[str, bool]:
    deps = {"config": _CONFIG_LOADED, "models": _MODELS_LOADED}
    for lib in ["numpy", "pandas", "sklearn", "tensorflow", "transformers", "yfinance", "requests"]:
        try:
            __import__(lib)
            deps[lib if lib != "sklearn" else "scikit_learn"] = True
        except ImportError:
            deps[lib if lib != "sklearn" else "scikit_learn"] = False
    return deps

def initialize_l3(config_updates: Optional[Dict[str, Dict[str, Any]]] = None, create_directories: bool = True) -> bool:
    _logger.info('l3_strategic')
    try:
        deps = check_dependencies()
        if not deps["config"] or not deps["models"]:
            _logger.error("Faltan dependencias críticas de L3")
            return False
        
        if create_directories and _CONFIG_LOADED:
            import os
            for path_name, path in PATHS.items():
                try:
                    os.makedirs(path, exist_ok=True)
                    _logger.debug(f"Directorio {path_name} verificado: {path}")
                except Exception as e:
                    _logger.warning(f"No se pudo crear directorio {path}: {e}")
        
        if config_updates and _CONFIG_LOADED:
            for section, updates in config_updates.items():
                try:
                    update_config(section, updates)
                    _logger.info(f"Configuración actualizada para sección: {section}")
                except Exception as e:
                    _logger.warning(f"Error actualizando configuración {section}: {e}")
        
        if _CONFIG_LOADED:
            missing_keys = [k for k, v in API_KEYS.items() if not v]
            if missing_keys:
                _logger.warning(f"API keys faltantes: {missing_keys}")
                _logger.info("Algunas funcionalidades pueden estar limitadas")
        
        _logger.info("L3 Strategic inicializado correctamente")
        return True
    except Exception as e:
        _logger.error(f"Error inicializando L3: {e}")
        return False

def get_strategic_capabilities() -> List[str]:
    capabilities = ["config_management", "data_models"]
    deps = check_dependencies()
    if deps.get("numpy") and deps.get("pandas"):
        capabilities += ["market_data_processing", "correlation_analysis", "risk_metrics"]
    if deps.get("scikit_learn"):
        capabilities += ["regime_detection", "portfolio_optimization"]
    if deps.get("tensorflow") or deps.get("transformers"):
        capabilities += ["sentiment_analysis", "volatility_forecasting"]
    if deps.get("yfinance") or deps.get("requests"):
        capabilities += ["external_data_fetching", "macro_data_integration"]
    return sorted(capabilities)

# Auto-inicialización básica
if __name__ != "__main__":
    _logger.info(f"L3 Strategic v{__version__} cargado")
    if not _CONFIG_LOADED or not _MODELS_LOADED:
        _logger.warning("Algunas funcionalidades de L3 no están disponibles")
    capabilities = get_strategic_capabilities()
    _logger.info(f"Capacidades disponibles: {len(capabilities)} - {', '.join(capabilities[:3])}{'...' if len(capabilities) > 3 else ''}")

# Funciones auxiliares
from .universe_filter import filtrar_universo

def procesar_l3(state):
    universo = ["BTC", "ETH", "USDT"]
    exposicion = {act: 0.5 for act in universo}
    state["universo"] = universo
    state["exposicion"] = exposicion
    return state
