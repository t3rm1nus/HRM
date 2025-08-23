# 🎯 L2\_Tactical - Nivel Táctico de Ejecución

## ⚡ Objetivo

L2 es el **cerebro táctico** que convierte decisiones estratégicas de L3 en señales ejecutables para L1. Combina **modelos FinRL pre-entrenados** con análisis técnico avanzado y gestión de riesgo inteligente para generar señales de alta calidad en tiempo real (escala de minutos).

Generar señales de trading mediante composición de múltiples fuentes (IA + técnico + patrones)
Calcular position sizing óptimo usando Kelly Criterion y vol-targeting
Aplicar controles de riesgo pre-ejecución (stops, correlaciones, límites)
Adaptarse dinámicamente a diferentes regímenes de mercado

---

## 🚫 Lo que L2 NO hace

| ❌ No hace                                           |
| --------------------------------------------------- |
| No toma decisiones de asignación de capital global  |
| No define régimen de mercado (responsabilidad L3)   |
| No ejecuta órdenes directamente (responsabilidad L1)|
| No recolecta datos de mercado raw                   |
| No modifica parámetros de configuración global     |

---

## ✅ Lo que L2 SÍ hace

| ✅ Funcionalidad         | Descripción                                                                    |
| ----------------------- | ------------------------------------------------------------------------------ |
| Signal Generation       | Combina FinRL ensemble + análisis técnico para señales precisas               |
| Position Sizing         | Kelly fraccionado, vol-targeting y risk parity optimization                   |
| Risk Controls           | Stop-loss dinámico, take-profit inteligente, drawdown protection              |
| Multi-Timeframe         | Fusión de señales 1m, 5m, 15m, 1h con consensus scoring                       |
| Pattern Recognition     | Detección automática de patrones técnicos y breakouts                         |
| Model Integration       | Carga y gestión de modelos FinRL pre-entrenados (pkl/zip)                     |

---

## 🏗️ Arquitectura

```text
L3 (Strategic Decisions)
        ↓
┌─────────────────────────────────────────┐
│            L2_tactic                    │
│                                         │
│  ┌─────────────┐  ┌─────────────────┐   │
│  │ AI Model    │  │ Signal          │   │
│  │ Integration │──│ Generator       │   │
│  └─────────────┘  └─────────────────┘   │
│                           │             │
│  ┌─────────────┐  ┌───────▼─────────┐   │
│  │ Technical   │  │ Signal          │   │
│  │ Indicators  │──│ Composer        │   │
│  └─────────────┘  └─────────────────┘   │
│                           │             │
│  ┌─────────────┐  ┌───────▼─────────┐   │
│  │ Pattern     │  │ Position        │   │
│  │ Recognition │  │ Sizer           │   │
│  └─────────────┘  └─────────────────┘   │
│                           │             │
│                   ┌───────▼─────────┐   │
│                   │ Risk            │   │
│                   │ Controls        │   │
│                   └─────────────────┘   │
└─────────────────────────────────────────┘
        ↓
    L2 Signals → L1 (Execution)
```

### Componentes Principales

* `models.py` - Estructuras de datos (TacticalDecision, MarketFeatures, PositionIntent)
* `config.py` - Configuración L2 (model paths, thresholds, risk limits)
* `bus_adapter.py` - Comunicación asíncrona L3 ↔ L2 ↔ L1
* `signal_generator.py` - Orquestador principal de generación de señales
* `position_sizer.py` - Cálculo inteligente de tamaños de posición
* `risk_controls.py` - Gestión dinámica de riesgo y stops
* `finrl_models/` - Gestión de modelos FinRL pre-entrenados
* `technical/` - Análisis técnico multi-timeframe
* `ensemble/` - Lógica de ensemble y voting

---

## 📁 Estructura del Proyecto

```text
l2_tactical/
├── 📄 README.md              # Este archivo
├── 📄 __init__.py
├── 📄 models.py              # Estructuras de datos L2
├── 📄 config.py              # Configuración y parámetros
├── 📄 bus_adapter.py         # Comunicación con MessageBus
├── 📄 signal_generator.py    # Generador principal de señales
├── 📄 position_sizer.py      # Sizing inteligente de posiciones
├── 📄 risk_controls.py       # Controles dinámicos de riesgo
│
├── 📁 finrl_models/          # Gestión de modelos FinRL
│   ├── 📄 __init__.py
│   ├── 📄 model_loader.py    # Carga de pkl/zip pre-entrenados
│   ├── 📄 ensemble_manager.py # Gestión de múltiples modelos
│   ├── 📄 feature_processor.py # Pre/post-procesamiento features
│   └── 📁 saved_models/      # Directorio para modelos (.pkl/.zip)
│       ├── 📦 ensemble_btc_v1.pkl
│       ├── 📦 trend_agent_v2.pkl
│       └── 📦 volatility_agent_v1.pkl
│
├── 📁 technical/             # Análisis técnico avanzado
│   ├── 📄 __init__.py
│   ├── 📄 indicators.py      # RSI, MACD, BB multi-timeframe
│   ├── 📄 patterns.py        # Chart & candlestick patterns
│   ├── 📄 multi_timeframe.py # Fusión temporal de señales
│   └── 📄 support_resistance.py # Niveles dinámicos
│
├── 📁 ensemble/              # Lógica de ensemble
│   ├── 📄 __init__.py
│   ├── 📄 voting_strategy.py # Weighted/majority voting
│   ├── 📄 confidence_calc.py # Cálculo de confidence scores
│   └── 📄 consensus_builder.py # Construcción de consenso
│
├── 📁 tests/                 # Tests unitarios e integración
│   ├── 📄 test_signal_generator.py
│   ├── 📄 test_model_loader.py
│   ├── 📄 test_ensemble.py
│   ├── 📄 test_integration_l1.py
│   └── 📄 test_integration_l3.py
│
├── 📄 requirements.txt       # Dependencias L2
└── 📄 run_l2_tests.py       # Script de testing
```

---

## 🔄 Flujo de Procesamiento

```text
1. 📥 ENTRADA: Strategic Decision de L3
   ├─ Regime de mercado (trend/range/volatile)
   ├─ Universo de activos (BTC focus)
   ├─ Target exposure (0.0 - 1.0)
   └─ Risk appetite (conservative/aggressive)

2. 🧠 PROCESAMIENTO TÁCTICO:
   ├─ 📊 Market Features (multi-timeframe)
   ├─ 🤖 FinRL Model Predictions (ensemble)
   ├─ 📈 Technical Analysis (indicators + patterns)
   ├─ 🎯 Signal Fusion (weighted voting)
   ├─ 📏 Position Sizing (Kelly + vol-targeting)
   └─ 🛡️ Risk Controls (stops + limits)

3. 📤 SALIDA: Tactical Signal a L1
   ├─ symbol: "BTC/USDT"
   ├─ side: "buy"/"sell"/"hold"
   ├─ qty: 0.05 (BTC amount)
   ├─ confidence: 0.85
   ├─ stop_loss: 49000.0
   ├─ take_profit: 52000.0
   └─ reasoning: {"ensemble_vote": "bullish", "rsi_div": true}
```

---

## 🤖 Integración de Modelos FinRL

### Carga de Modelos Pre-entrenados

```python
from l2_tactical.finrl_models import ModelLoader, EnsembleManager

# Cargar modelos desde archivos pkl/zip
loader = ModelLoader()
models = {
    'trend_agent': loader.load_model('saved_models/trend_agent_v2.pkl'),
    'mean_revert_agent': loader.load_model('saved_models/mean_revert_v1.pkl'),
    'volatility_agent': loader.load_model('saved_models/volatility_agent_v1.pkl')
}

# Configurar ensemble
ensemble = EnsembleManager(models)
ensemble.set_weights({'trend_agent': 0.4, 'mean_revert_agent': 0.35, 'volatility_agent': 0.25})

# Generar predicción
features = get_current_market_features()
prediction = ensemble.predict(features)
```

### Estructura de Modelos Esperada

Los modelos FinRL deben incluir:

```python
# Estructura esperada del archivo .pkl
model_data = {
    'model': trained_model,           # Modelo entrenado (A3C/PPO/SAC)
    'scaler': feature_scaler,         # StandardScaler para features
    'feature_names': ['rsi', 'macd', ...],  # Nombres de features
    'action_space': 7,                # Número de acciones
    'specialization': 'trend_following',  # Especialización del modelo
    'performance_metrics': {          # Métricas de entrenamiento
        'sharpe_ratio': 2.1,
        'max_drawdown': 0.12,
        'win_rate': 0.58
    },
    'training_config': {              # Configuración de entrenamiento
        'lookback_window': 60,
        'timeframes': ['5m', '15m', '1h'],
        'training_period': '2023-01-01_2024-01-01'
    }
}
```

---

## ⚙️ Configuración

### Configuración Principal (config.py)

```python
# Model Configuration
MODEL_CONFIG = {
    'ensemble_models': {
        'trend_agent': {
            'path': 'saved_models/trend_agent_v2.pkl',
            'weight': 0.4,
            'specialization': 'trend_following',
            'timeframes': ['5m', '15m', '1h']
        },
        'mean_revert_agent': {
            'path': 'saved_models/mean_revert_v1.pkl', 
            'weight': 0.35,
            'specialization': 'mean_reversion',
            'timeframes': ['1m', '5m']
        },
        'volatility_agent': {
            'path': 'saved_models/volatility_agent_v1.pkl',
            'weight': 0.25,
            'specialization': 'volatility_breakout',
            'timeframes': ['1m', '5m', '15m']
        }
    },
    'consensus_threshold': 0.6,       # Mínimo consensus para señal
    'confidence_threshold': 0.7,      # Mínima confidence para ejecución
    'rebalance_frequency': '1h'       # Frecuencia de ajuste de pesos
}

# Risk Configuration
RISK_CONFIG = {
    'position_sizing': {
        'kelly_fraction': 0.25,        # 25% del Kelly óptimo
        'max_position_size': 0.1,      # 10% máximo por trade
        'volatility_target': 0.15,     # 15% vol anualizada objetivo
        'correlation_adjustment': True
    },
    'stop_loss': {
        'atr_multiplier': 2.0,         # Stop = 2 * ATR
        'max_loss_per_trade': 0.02,    # 2% máximo loss por trade
        'trailing_stop': True,
        'breakeven_threshold': 1.5     # Move to BE at 1.5R
    },
    'portfolio_limits': {
        'max_daily_trades': 10,
        'max_concurrent_positions': 3,
        'daily_loss_limit': 0.05,      # 5% daily loss limit
        'exposure_limit': 0.8          # 80% max exposure
    }
}

# Technical Analysis Configuration  
TECHNICAL_CONFIG = {
    'indicators': {
        'rsi_periods': [14, 21],
        'macd_config': [12, 26, 9],
        'bb_periods': [20, 2.0],
        'atr_period': 14
    },
    'patterns': {
        'candlestick_patterns': True,
        'chart_patterns': True,
        'support_resistance': True,
        'volume_analysis': True
    },
    'timeframes': {
        'primary': '1m',
        'secondary': ['5m', '15m', '1h'],
        'alignment_threshold': 0.7
    }
}
```

---

## 🔬 Testing

### Tests Unitarios

```bash
# Ejecutar todos los tests
python run_l2_tests.py

# Tests específicos
python -m pytest tests/test_signal_generator.py -v
python -m pytest tests/test_model_loader.py -v
python -m pytest tests/test_ensemble.py -v
```

### Tests de Integración

```bash
# Test integración con L1
python -m pytest tests/test_integration_l1.py -v

# Test integración con L3  
python -m pytest tests/test_integration_l3.py -v
```

### Validación de Modelos

```python
from l2_tactical.tests import validate_models

# Validar todos los modelos cargados
validation_results = validate_models('saved_models/')
print(f"Models validated: {validation_results['passed']}/{validation_results['total']}")
```

---

## 🚀 Uso Rápido

### Inicialización

```python
import asyncio
from comms.message_bus import MessageBus
from l2_tactical.signal_generator import TacticalSignalGenerator
from l2_tactical.bus_adapter import L2BusAdapter

# Configurar sistema
bus = MessageBus()
signal_generator = TacticalSignalGenerator()
adapter = L2BusAdapter(bus, signal_generator)

# Iniciar procesamiento
async def main():
    await adapter.start_processing()

asyncio.run(main())
```

### Ejemplo de Uso Directo

```python
from l2_tactical import TacticalSignalGenerator

# Crear generador
generator = TacticalSignalGenerator()

# Procesar decisión estratégica de L3
l3_decision = {
    'regime': 'trending',
    'target_exposure': 0.7,
    'risk_appetite': 'aggressive',
    'universe': ['BTC/USDT']
}

# Generar señal táctica
signal = await generator.process_strategic_decision(l3_decision)

print(f"Signal: {signal.side} {signal.qty} BTC @ confidence {signal.confidence}")
print(f"Stop Loss: {signal.stop_loss}, Take Profit: {signal.take_profit}")
```

---

## 📊 Métricas y Monitoring

### Métricas Clave L2

- **Signal Quality Score**: Precisión de señales generadas
- **Ensemble Consensus**: Grado de acuerdo entre modelos  
- **Confidence Distribution**: Histograma de confidence scores
- **Risk Adjusted Returns**: Sharpe ratio de señales ejecutadas
- **Model Performance Tracking**: Performance individual por modelo
- **Latency Metrics**: Tiempo de generación de señales

### Dashboard en Tiempo Real

```python
from l2_tactical.monitoring import L2Dashboard

dashboard = L2Dashboard()
dashboard.display_metrics()

# Output:
# ┌─────────────────────────────────────────────┐
# │              L2 TACTICAL METRICS            │
# ├─────────────────────────────────────────────┤  
# │ Signals Generated (1h): 12                  │
# │ Average Confidence: 0.78                    │
# │ Ensemble Consensus: 0.85                    │
# │ Active Positions: 2/3                       │
# │ Portfolio Heat: 0.65                        │
# │ Latency P95: 45ms                          │
# └─────────────────────────────────────────────┘
```

---

## 🔧 Instalación y Dependencias

### Instalación

```bash
cd l2_tactical/
pip install -r requirements.txt
```

### Dependencias Principales

```text
# FinRL y ML
finrl>=0.3.6
stable-baselines3>=1.7.0
torch>=1.12.0
scikit-learn>=1.1.0
pandas>=1.5.0
numpy>=1.21.0

# Análisis técnico
talib>=0.4.25
ta>=0.10.2

# Comunicaciones  
asyncio
aioredis>=2.0.0

# Testing
pytest>=7.0.0
pytest-asyncio>=0.21.0

# Monitoring
rich>=12.0.0
```

---

## 🎯 Objetivos de Performance

### Targets L2

| Métrica                    | Objetivo      | Actual |
|---------------------------|---------------|--------|
| Signal Generation Latency | < 100ms       | TBD    |
| Signal Accuracy           | > 65%         | TBD    |
| Average Confidence        | > 0.75        | TBD    |
| Ensemble Consensus        | > 0.70        | TBD    |  
| Sharpe Ratio (signals)    | > 2.0         | TBD    |
| Max Drawdown              | < 15%         | TBD    |
| Daily Uptime              | > 99.9%       | TBD    |

### Benchmarking

```python
from l2_tactical.benchmarking import run_benchmark

# Ejecutar benchmark completo
results = run_benchmark(
    models_path='saved_models/',
    test_period='2024-01-01_2024-03-01',
    symbols=['BTC/USDT']
)

print(f"Benchmark Results: {results}")
```

---

## 🛠️ Desarrollo y Contribución

### Roadmap de Desarrollo

#### Sprint 1-2: Core Infrastructure ⚡
- [x] Estructura base del proyecto
- [ ] Modelos de datos y configuración  
- [ ] Bus adapter y comunicaciones
- [ ] Model loader básico

#### Sprint 3-4: FinRL Integration 🤖  
- [ ] Ensemble manager completo
- [ ] Feature preprocessing pipeline
- [ ] Model validation framework
- [ ] Performance tracking

#### Sprint 5-6: Technical Analysis 📈
- [ ] Multi-timeframe indicators
- [ ] Pattern recognition system
- [ ] Support/resistance detection
- [ ] Signal fusion algorithms

#### Sprint 7-8: Risk & Sizing 🛡️
- [ ] Position sizing algorithms
- [ ] Dynamic risk controls
- [ ] Portfolio heat management
- [ ] Correlation adjustments

#### Sprint 9-10: Integration & Testing ✅
- [ ] L1/L3 integration testing
- [ ] Performance optimization
- [ ] Documentation completa
- [ ] Production deployment

### Guidelines de Contribución

1. **Fork** el repositorio
2. Crear **feature branch** (`git checkout -b feature/AmazingFeature`)  
3. **Commit** cambios (`git commit -m 'Add AmazingFeature'`)
4. **Push** a branch (`git push origin feature/AmazingFeature`)
5. Abrir **Pull Request**

### Estándares de Código

- **Python 3.9+**
- **Type hints** obligatorios
- **Docstrings** estilo Google
- **Tests** para toda funcionalidad nueva
- **Black** para formateo de código
- **flake8** para linting

---

## 📚 Referencias y Links

### Documentación Técnica
- [FinRL Documentation](https://finrl.readthedocs.io/)
- [Stable-Baselines3 Guide](https://stable-baselines3.readthedocs.io/)
- [Bitcoin Trading Strategies](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3870666)

### Papers Relevantes
- *"Ensemble Methods for Deep Reinforcement Learning"* (2021)
- *"Multi-Agent Reinforcement Learning for Cryptocurrency Trading"* (2023)  
- *"Risk-Aware Portfolio Management with Deep RL"* (2024)

### Recursos Externos
- [Crypto Feature Engineering](https://github.com/features/crypto)
- [Technical Analysis Library](https://github.com/bukosabino/ta)
- [FinRL Examples](https://github.com/AI4Finance-Foundation/FinRL)

---

## 📧 Contacto y Soporte

Para preguntas, issues o contribuciones:

- **Issues**: [GitHub Issues](https://github.com/tu-repo/l2_tactical/issues)
- **Discussions**: [GitHub Discussions](https://github.com/tu-repo/l2_tactical/discussions)
- **Email**: tu-email@ejemplo.com

---

## 📜 Licencia

Este proyecto está bajo la licencia MIT. Ver `LICENSE` para más detalles.

---

<div align="center">

**🚀 L2 Tactical - Where FinRL meets Real-Time Trading 🚀**

*Desarrollado con ❤️ para el Sistema HRM*

![Python](https://img.shields.io/badge/python-v3.9+-blue.svg)
![FinRL](https://img.shields.io/badge/FinRL-v0.3.6+-green.svg)
![Status](https://img.shields.io/badge/status-in_development-yellow.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

</div>