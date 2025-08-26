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

| ✅ Funcionalidad     | Descripción                                                                  |
| ------------------- | ---------------------------------------------------------------------------- |
| Signal Generation   | Combina ensemble FinRL + análisis técnico + patrones para señales precisas   |
| Signal Composition  | Fusiona señales multi-fuente con pesos dinámicos según régimen               |
| Position Sizing     | Kelly fraccionado, vol-targeting y validación de límites                     |
| Risk Controls       | Stop-loss dinámico, TP inteligente, drawdown por señal/estrategia y liquidez |
| Multi-Asset         | Soporta BTC/USDT, ETH/USDT y extensible a más pares                          |
| Multi-Timeframe     | Fusión de señales 1m, 5m, 15m, 1h con consensus scoring                      |
| Pattern Recognition | Detección de patrones técnicos y breakouts                                   |
| Mock Data Mode      | Generación de datos simulados para pruebas sin conexión real                 |
| Logging Enriquecido | Logs detallados con trazabilidad paso a paso y metadatos                     |

---

## 🏗️ Arquitectura Modular

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
│                   │ Risk Controls   │   │
│                   │  ├─ DynamicStops│   │
│                   │  ├─ Portfolio   │   │
│                   │  └─ Liquidity   │   │
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
* `position_sizer.py` - Cálculo inteligente de tamaños de posición (Kelly + vol-targeting)
* `risk_controls/` - Módulo modularizado de gestión de riesgo

  * `dynamic_stops.py` - Cálculo y gestión de stops dinámicos
  * `portfolio_risk.py` - Riesgo de correlación, calor de portafolio y drawdown global
  * `liquidity_checks.py` - Validación de liquidez mínima y ratios
  * `manager.py` - Orquestador principal de controles de riesgo
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
├── 📁 risk_controls/         # Controles de riesgo (modularizados)
│   ├── 📄 dynamic_stops.py
│   ├── 📄 portfolio_risk.py
│   ├── 📄 liquidity_checks.py
│   └── 📄 manager.py
├── 📄 procesar_l2.py         # Script orquestador / runner local
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
│   ├── 📄 test_risk_controls.py
│   └── 📄 test_integration.py
│
├── 📄 requirements.txt       # Dependencias L2
└── 📄 run_l2_tests.py        # Script de testing
```

---

## 🔄 Risk Controls (modularizado)

Antes: todo en risk_controls.py (~600 líneas).
Ahora: separado en 6 módulos dentro de l2_tactic/risk_controls/.

l2_tactic/risk_controls/
 ├── __init__.py         # punto de entrada público
 ├── alerts.py           # enums y RiskAlert
 ├── stop_losses.py      # DynamicStopLoss y StopLossOrder
 ├── positions.py        # RiskPosition (posición normalizada)
 ├── portfolio.py        # PortfolioRiskManager (riesgo agregado)
 └── manager.py          # RiskControlManager (orquestador central)

Módulos

alerts.py

RiskLevel, AlertType, RiskAlert

Estructura estándar para todas las alertas.

stop_losses.py

DynamicStopLoss → stop inicial (ATR, vol, S/R, trailing, breakeven).

StopLossOrder → datos de un stop activo.

positions.py

RiskPosition → representación simplificada de una posición para gestión de riesgo.

portfolio.py

PortfolioRiskManager → chequea correlación, heat, drawdowns de cartera, límites de posiciones y métricas agregadas (volatilidad, Sharpe, retorno).

manager.py

RiskControlManager → integra todo:

Evalúa señales pre-trade (liquidez, correlación, drawdowns de señal/estrategia).

Ajusta tamaño o bloquea operaciones.

Mantiene stops dinámicos, trailing y TP.

Trackea drawdowns por señal y estrategia.

init.py

Exposición pública sencilla para evitar imports largos.

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
   └─ 🛡 Risk Controls (stops dinámicos + drawdowns + liquidez)

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

## 🔬 Testing

```bash
# Ejecutar todos los tests
python run_l2_tests.py

# Tests unitarios
pytest tests/test_signal_generator.py -v
pytest tests/test_signal_composer.py -v
pytest tests/test_position_sizer.py -v
pytest tests/test_risk_controls.py -v
```

---

<div align="center">

**🚀 L2 Tactical - Where FinRL meets Real-Time Trading 🚀**

*Desarrollado con ❤️ para el Sistema HRM*

![Python](https://img.shields.io/badge/python-v3.9+-blue.svg)
![FinRL](https://img.shields.io/badge/FinRL-v0.3.6+-green.svg)
![Status](https://img.shields.io/badge/status-in_development-yellow.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

</div>
