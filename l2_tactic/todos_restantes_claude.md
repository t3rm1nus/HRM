🛠️ Configuración
Archivo de configuración (config/l2_config.yaml)
yamlai_model:
  model_path: "models/my_model.zip"
  model_type: "sklearn"  # sklearn|pytorch|tensorflow|custom
  prediction_threshold: 0.6
  cache_predictions: true

signals:
  min_signal_strength: 0.3
  ai_model_weight: 0.6
  technical_weight: 0.3
  pattern_weight: 0.1
  signal_expiry_minutes: 15

position_sizing:
  kelly_fraction: 0.25
  target_volatility: 0.15
  max_position_pct: 0.10

risk:
  default_stop_pct: 0.02
  max_correlation: 0.7
  daily_loss_limit: 0.05
Variables de entorno
bash# Modelo IA
export L2_AI_MODEL_PATH="models/my_model.zip"
export L2_AI_MODEL_TYPE="sklearn"
export L2_PREDICTION_THRESHOLD="0.6"

# Señales
export L2_MIN_SIGNAL_STRENGTH="0.3"
export L2_AI_MODEL_WEIGHT="0.6"

# Position sizing
export L2_KELLY_FRACTION="0.25"
export L2_MAX_POSITION_PCT="0.10"

# Riesgo
export L2_DEFAULT_STOP_PCT="0.02"
export L2_MAX_CORRELATION="0.7"
🚀 Uso básico
1. Inicialización
pythonfrom l2_tactic import L2Config, SignalGenerator

# Configuración desde archivo
config = L2Config.from_file("config/l2_config.yaml")

# O desde variables de entorno
config = L2Config.from_env()

# Inicializar generador
signal_gen = SignalGenerator(config)
2. Generar señales
pythonimport pandas as pd

# Datos OHLCV
market_data = pd.read_csv("btc_1m.csv", parse_dates=["timestamp"], index_col="timestamp")

# Contexto opcional desde L3
regime_context = {
    "regime": "trending",  # trending|ranging|volatile
    "strength": 0.8,
    "preferred_assets": ["BTCUSDT", "ETHUSDT"]
}

# Generar señales
signals = signal_gen.generate_signals(
    market_data=market_data,
    symbol="BTCUSDT",
    regime_context=regime_context
)

# Procesar señales
for signal in signals:
    print(f"Signal: {signal.direction.value} {signal.symbol}")
    print(f"Strength: {signal.strength:.2f}, Confidence: {signal.confidence:.2f}")
    print(f"Source: {signal.source.value}")
    print(f"Metadata: {signal.metadata}")
3. Composición de señales
python# El SignalGenerator automáticamente:
# 1. Genera señales de múltiples fuentes
# 2. Aplica filtros de calidad
# 3. Compone señales con pesos dinámicos
# 4. Resuelve conflictos entre señales opuestas

# Las señales resultantes están listas para L1
🔧 Integración del modelo de IA
Formato esperado del .zip
my_model.zip
├── model.pkl                # Modelo principal (sklearn)
├── preprocessor.pkl         # Preprocessor (opcional)
├── feature_names.json       # Nombres de features (opcional)
└── metadata.json           # Metadata del modelo (opcional)
metadata.json (ejemplo)
json{
    "version": "1.0.0",
    "trained_on": "2025-01-15",
    "features_count": 42,
    "model_type": "RandomForestClassifier",
    "performance": {
        "accuracy": 0.68,
        "precision": 0.72,
        "recall": 0.64
    },
    "preprocessing": {
        "scaler": "StandardScaler",
        "feature_selection": true
    }
}
Features esperadas
El modelo debe estar entrenado con features similares a las generadas por data/loaders.py:

close, volume, delta_close
ema_10, ema_20, sma_10, sma_20
rsi, macd, macd_signal, macd_hist
vol_rel (volumen relativo)
Features 5m: close_5m, volume_5m, etc.

📊 Tipos de señales generadas
TacticalSignal
python@dataclass
class TacticalSignal:
    symbol: str                    # "BTCUSDT"
    direction: SignalDirection     # LONG|SHORT|NEUTRAL
    strength: float               # [0.0-1.0] Fuerza de la señal
    confidence: float             # [0.0-1.0] Confianza del modelo
    price: float                  # Precio de referencia
    timestamp: datetime           # Momento de generación
    source: SignalSource          # AI_MODEL|TECHNICAL|PATTERN|COMPOSITE
    metadata: Dict[str, Any]      # Info adicional específica
    expires_at: Optional[datetime] # Cuándo expira
Direcciones de señal

LONG: Comprar/mantener posición larga
SHORT: Vender/mantener posición corta
NEUTRAL: Sin sesgo direccional
CLOSE_LONG: Cerrar posición larga
CLOSE_SHORT: Cerrar posición corta

Fuentes de señal

AI_MODEL: Predicción del modelo de IA
TECHNICAL: Indicadores técnicos (RSI, MACD, BB)
PATTERN: Patrones de velas/formaciones
COMPOSITE: Composición ponderada de múltiples fuentes

🧪 Testing
Tests unitarios
bash# Instalar dependencias de testing
pip install pytest pytest-mock

# Ejecutar tests
pytest l2_tactic/tests/ -v

# Test específicos
pytest l2_tactic/tests/test_signal_generator.py -v
pytest l2_tactic/tests/test_ai_integration.py -v
Test de integración con datos reales
python# Ver ejemplo en tests/test_integration.py
python l2_tactic/tests/test_integration.py
📈 Monitoreo y métricas
Métricas de señales

Hit rate por fuente (AI, técnico, patrones)
Sharpe ratio por timeframe
Latencia de generación
Cache hit ratio

Logging estructurado
pythonimport logging
from core.logging import logger

# Los logs incluyen:
logger.info("Generated 3 AI signals for BTCUSDT", extra={
    "symbol": "BTCUSDT",
    "signal_count": 3,
    "source": "AI_MODEL",
    "avg_confidence": 0.72
})
🔄 Integración con otros niveles
Desde L3 (entrada)
python# L3 proporciona contexto de régimen
regime_context = {
    "regime": "trending",
    "volatility": "high", 
    "preferred_timeframe": "5m",
    "risk_budget": 0.15
}
Hacia L1 (salida)
python# L2 genera señales que L1 ejecuta
signals = [
    TacticalSignal(
        symbol="BTCUSDT",
        direction=SignalDirection.LONG,
        strength=0.75,
        # ... otros campos
    )
]
⚠️ Consideraciones importantes
Performance

Use cache de predicciones para evitar recálculos
Batch processing cuando sea posible
Limite el lookback de indicadores técnicos

Robustez

El sistema funciona aunque el modelo IA falle (fallback)
Validación exhaustiva de datos de entrada
Timeouts en predicciones para evitar bloqueos

Escalabilidad

Soporte para múltiples símbolos concurrentes
Pesos dinámicos que se adaptan a performance
Pluggable - fácil agregar nuevas fuentes de señales


🚧 TODOs pendientes (Fase 2+)

 Implementar PositionSizer (Kelly + vol-targeting)
 Implementar RiskControls (stops, correlaciones)
 Bus integration para comunicación L1↔L2↔L3
 Dashboard en tiempo real
 Backtesting histórico
 Optimización de hiperparámetros
 Soporte para más tipos de modelos (XGBoost, LightGBM)
 API REST para integración externa



Resumen de archivos creados:

__init__.py - Inicialización del módulo con exports principales
models.py - Estructuras de datos (TacticalSignal, PositionSize, RiskMetrics, L2State)
config.py - Configuración completa con carga desde archivo/env
ai_model_integration.py - Wrapper genérico para tu modelo .zip
signal_generator.py - Orquestador principal de señales
requirements.txt - Dependencias del módulo
README.md - Documentación completa del módulo

🎯 Características implementadas:
AIModelWrapper

✅ Soporte para sklearn, PyTorch, TensorFlow, custom
✅ Carga automática desde .zip con extracción
✅ Preprocessing y feature validation
✅ Cache de predicciones con TTL
✅ Fallback robusto en caso de errores
✅ Metadata y feature importance

SignalGenerator

✅ Integración con modelo de IA como señal primaria
✅ Indicadores técnicos: RSI, MACD, Bollinger Bands
✅ Reconocimiento de patrones: Hammer, Engulfing, S/R
✅ Sistema de pesos dinámicos basado en performance
✅ Filtros de calidad y resolución de conflictos
✅ Composición de señales múltiples

Configuración

✅ Config completo con validaciones
✅ Carga desde variables de entorno
✅ Carga desde archivos YAML
✅ Parámetros para todos los componentes

Archivos Generados
1. position_sizer.py - Sistema Avanzado de Position Sizing
Características principales:

✅ Kelly Criterion con ajustes conservadores
✅ Volatility Targeting para control de riesgo
✅ Risk Parity para contribución equitativa de riesgo
✅ Ensemble Sizer que combina múltiples métodos
✅ Position limits y validaciones exhaustivas
✅ Correlation adjustments entre posiciones
✅ Portfolio heat monitoring en tiempo real

2. risk_controls.py - Controles Dinámicos de Riesgo
Características principales:

✅ Dynamic Stop-Loss (fixed, ATR, volatility, S/R based)
✅ Trailing Stops con move-to-breakeven
✅ Portfolio Risk Manager con límites globales
✅ Correlation Risk management entre activos
✅ Drawdown Protection y daily loss limits
✅ Volatility Spike Detection en tiempo real
✅ Risk Alerts con severidad clasificada
✅ Position Monitoring continuo con MAE/MFE

3. bus_integration.py - Comunicación Asíncrona Completa
Características principales:

✅ MessageBus Integration completa con HRM
✅ Asynchronous Processing de decisiones L3→L2→L1
✅ Correlation Tracking entre mensajes
✅ Error Handling y retry logic
✅ Heartbeat & Monitoring en tiempo real
✅ Risk Alert Broadcasting automático
✅ Execution Report Processing de L1
✅ State Management completo de L2
Flujo de Integración Completo
textL3 Strategic Decision
        ↓
   L2 Bus Adapter (receives)
        ↓
Signal Generator + AI Models
        ↓
   Position Sizer (Kelly/Vol/Risk Parity)
        ↓
   Risk Controls (stops/limits/correlation)
        ↓
   Final Tactical Signal → L1
        ↓
   Execution Reports ← L1
        ↓
   Position Monitoring & Alerts
🚀 Próximos pasos:
Para usar este módulo necesitas:

Colocar tu modelo .zip en la carpeta models/
Ajustar la configuración según tu modelo específico
Instalar dependencias: pip install -r l2_tactic/requirements.txt
Probar con datos:
pythonfrom l2_tactic import L2Config, SignalGenerator
config = L2Config.from_env()
generator = SignalGenerator(config)


🔧 Personalización para tu modelo:
Si tu modelo tiene un formato específico, solo necesitas modificar el método _load_custom_model() en ai_model_integration.py para adaptarlo a tu estructura de archivos.
 