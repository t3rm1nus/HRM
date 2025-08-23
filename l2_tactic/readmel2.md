# 🎯 L2\_Tactical - Nivel Táctico de Ejecución

## ⚡ Objetivo

L2 es el **cerebro táctico** que convierte decisiones estratégicas de L3 en **señales ejecutables** para L1. Combina **modelos FinRL pre-entrenados**, análisis técnico avanzado, composición de señales multi-fuente y **gestión dinámica de riesgo** para generar señales de alta calidad en tiempo real (escala de minutos).

Genera y compone señales de trading (IA + técnico + patrones) → calcula el **position sizing óptimo** → aplica **controles de riesgo pre-ejecución** → entrega señales listas para L1.

---

## 🚫 Lo que L2 NO hace

| ❌ No hace                                            |
| ---------------------------------------------------- |
| No define régimen de mercado (responsabilidad L3)    |
| No toma decisiones de asignación global de capital   |
| No ejecuta órdenes directamente (responsabilidad L1) |
| No recolecta datos raw desde exchange                |
| No modifica parámetros de configuración global       |

---

## ✅ Lo que L2 SÍ hace

| ✅ Funcionalidad     | Descripción                                                                |
| ------------------- | -------------------------------------------------------------------------- |
| Signal Generation   | Combina ensemble FinRL + análisis técnico + patrones para señales precisas |
| Signal Composition  | Fusiona señales multi-fuente con pesos dinámicos según régimen             |
| Position Sizing     | Kelly fraccionado, vol-targeting y validación de límites                   |
| Risk Controls       | Stop-loss dinámico, TP inteligente, protección de drawdown                 |
| Multi-Asset         | Soporta BTC/USDT, ETH/USDT y extensible a más pares                        |
| Multi-Timeframe     | Fusión de señales 1m, 5m, 15m, 1h con consensus scoring                    |
| Pattern Recognition | Detección de patrones técnicos y breakouts                                 |
| Mock Data Mode      | Generación de datos simulados para pruebas sin conexión real               |
| Logging Enriquecido | Logs detallados con trazabilidad paso a paso y metadatos                   |

---

## 🏗️ Arquitectura

```text
L3 (Strategic Decisions)
        ↓
┌─────────────────────────────────────────┐
│              L2_tactic                  │
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

* `models.py` - Estructuras de datos (TacticalSignal, MarketFeatures, PositionIntent)
* `config.py` - Configuración L2 (modelos, thresholds, límites de riesgo)
* `bus_adapter.py` - Comunicación asíncrona L3 ↔ L2 ↔ L1
* `signal_generator.py` - Orquestador de generación de señales (IA + técnico + patrones)
* `signal_composer.py` - Composición dinámica y resolución de conflictos
* `position_sizer.py` - Cálculo inteligente de tamaños de posición
* `risk_controls.py` - Gestión dinámica de riesgo y stops
* `procesar_l2.py` - **Punto de entrada principal** para ejecución en local
* `finrl_models/` - Modelos FinRL pre-entrenados (.pkl/.zip)

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
├── 📄 signal_composer.py     # Composición y consenso de señales
├── 📄 position_sizer.py      # Sizing inteligente de posiciones
├── 📄 risk_controls.py       # Controles dinámicos de riesgo
├── 📄 procesar_l2.py         # Script orquestador / runner local
│
├── 📁 finrl_models/          # Modelos FinRL
│   ├── 📄 model_loader.py
│   ├── 📄 ensemble_manager.py
│   ├── 📄 feature_processor.py
│   └── 📁 saved_models/
│       ├── 📦 ensemble_btc_v1.pkl
│       ├── 📦 trend_agent_v2.pkl
│       └── 📦 volatility_agent_v1.pkl
│
├── 📁 technical/             # Indicadores técnicos avanzados
│   ├── 📄 indicators.py
│   ├── 📄 patterns.py
│   ├── 📄 multi_timeframe.py
│   └── 📄 support_resistance.py
│
├── 📁 ensemble/              # Lógica de ensemble
│   ├── 📄 voting_strategy.py
│   ├── 📄 confidence_calc.py
│   └── 📄 consensus_builder.py
│
├── 📁 tests/                 # Tests unitarios e integración
│   ├── 📄 test_signal_generator.py
│   ├── 📄 test_signal_composer.py
│   ├── 📄 test_position_sizer.py
│   ├── 📄 test_integration_l1.py
│   └── 📄 test_integration_l3.py
│
├── 📄 requirements.txt       # Dependencias L2
└── 📄 run_l2_tests.py        # Script de testing
```

---

## 🔄 Flujo de Procesamiento

```text
1. 📥 ENTRADA: Decisión estratégica de L3
   ├─ Regime de mercado (trend/range/volatile)
   ├─ Universo de activos (BTC, ETH, …)
   ├─ Target exposure (0.0 - 1.0)
   └─ Risk appetite (conservative/aggressive)

2. 🧠 PROCESAMIENTO TÁCTICO:
   ├─ 📊 Market Features (multi-timeframe)
   ├─ 🤖 FinRL Model Predictions (ensemble)
   ├─ 📈 Technical Analysis (indicators + patterns)
   ├─ 🎛 Signal Composition (consensus + dynamic weights)
   ├─ 📏 Position Sizing (Kelly + vol-targeting + limits)
   └─ 🛡 Risk Controls (stops + portfolio exposure)

3. 📤 SALIDA: Tactical Signal a L1
   ├─ symbol: "BTC/USDT"
   ├─ side: "buy" / "sell" / "hold"
   ├─ qty: 0.05 (BTC amount)
   ├─ confidence: 0.85
   ├─ stop_loss: 49000.0
   ├─ take_profit: 52000.0
   └─ metadata: {"ensemble_vote": "bullish", "weights": {…}}
```

---

## 🤖 Integración de Modelos FinRL

* Carga de modelos `.pkl` / `.zip` mediante `ModelLoader`.
* Ensemble dinámico con pesos configurables (`EnsembleManager`).
* Validación de modelos y tracking de performance.

Ejemplo:

```python
from l2_tactical.finrl_models import ModelLoader, EnsembleManager

loader = ModelLoader()
models = {
    'trend_agent': loader.load_model('saved_models/trend_agent_v2.pkl'),
    'volatility_agent': loader.load_model('saved_models/volatility_agent_v1.pkl')
}

ensemble = EnsembleManager(models)
ensemble.set_weights({'trend_agent': 0.6, 'volatility_agent': 0.4})

features = get_current_market_features()
prediction = ensemble.predict(features)
```

---

## ⚙️ Configuración

* **MODEL\_CONFIG**: rutas, pesos y thresholds de consenso.
* **RISK\_CONFIG**: fracción Kelly, límites de exposición, stops dinámicos.
* **TECHNICAL\_CONFIG**: parámetros de indicadores, timeframes y patrones.

---

## 🔬 Testing

```bash
# Ejecutar todos los tests
python run_l2_tests.py

# Tests unitarios
pytest tests/test_signal_generator.py -v
pytest tests/test_signal_composer.py -v
pytest tests/test_position_sizer.py -v

# Integración L1 / L3
test tests/test_integration_l1.py -v
test tests/test_integration_l3.py -v
```

---

## 🚀 Uso Rápido

### Runner Local (`procesar_l2.py`)

```bash
python procesar_l2.py --symbol BTC/USDT --regime trending
```

### Ejemplo de Uso en Código

```python
from l2_tactical.procesar_l2 import L2Processor

processor = L2Processor()
l3_decision = {
    'regime': 'trending',
    'target_exposure': 0.7,
    'risk_appetite': 'aggressive',
    'universe': ['BTC', 'ETH']
}

signals = processor.run(l3_decision)
for s in signals:
    print(s)
```

---

## 📊 Métricas y Monitoring

* **Signal Quality Score**: precisión histórica
* **Consensus Strength**: acuerdo entre modelos
* **Confidence Distribution**: histograma de confianza
* **Risk Adjusted Returns**: Sharpe de señales
* **Latency**: tiempo promedio de generación de señal

---

## 🔧 Instalación

```bash
cd l2_tactical/
pip install -r requirements.txt
```

Dependencias principales: `finrl`, `stable-baselines3`, `torch`, `pandas`, `numpy`, `ta-lib`, `rich`, `pytest`.

---

## 🎯 Performance Targets

| Métrica                | Objetivo |
| ---------------------- | -------- |
| Signal Latency         | < 100ms  |
| Signal Accuracy        | > 65%    |
| Average Confidence     | > 0.75   |
| Ensemble Consensus     | > 0.70   |
| Sharpe Ratio (signals) | > 2.0    |
| Max Drawdown           | < 15%    |

---

## 🛠️ Roadmap

* **Sprint 1-2**: Infraestructura base, model loader
* **Sprint 3-4**: Ensemble FinRL, feature pipeline
* **Sprint 5-6**: Technical + patterns, signal fusion
* **Sprint 7-8**: Sizing + risk controls
* **Sprint 9-10**: Integración completa con L1/L3, optimización

---

<div align="center">

**🚀 L2 Tactical - Where FinRL meets Real-Time Trading 🚀**

*Desarrollado con ❤️ para el Sistema HRM*

![Python](https://img.shields.io/badge/python-v3.9+-blue.svg)
![FinRL](https://img.shields.io/badge/FinRL-v0.3.6+-green.svg)
![Status](https://img.shields.io/badge/status-in_development-yellow.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

</div>
