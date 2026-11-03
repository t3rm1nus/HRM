# Servicio de Mercado - Datos y análisis técnico
#
# Proporciona acceso a velas OHLCV, indicadores técnicos y análisis de mercado.
# Reutiliza core/technical_indicators.py y l2_tactic/technical/.
#
from typing import Dict, Any, Optional, Literal
import pandas as pd
import numpy as np

# Rutas de importación reutilizables
try:
    from core.technical_indicators import calculate_technical_indicators
    TECHNICAL_INDICATORS_AVAILABLE = True
except ImportError:
    calculate_technical_indicators = None
    TECHNICAL_INDICATORS_AVAILABLE = False
try:
    from l2_tactic.technical.support_resistance import swing_pivots
    SUPPORT_RESISTANCE_AVAILABLE = True
except ImportError:
    swing_pivots = None
    SUPPORT_RESISTANCE_AVAILABLE = False
#
# Constantes
MAX_VELAS_LIMIT = 10000
DEFAULT_VELAS_LIMIT = min(MAX_VELAS_LIMIT, 1000)
#
