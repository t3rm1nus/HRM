# 🚀 Sistema de Trading Cripto con Modelo de Razonamiento Jerárquico (HRM)

## 1️⃣ Objetivo

- Diseñar, construir y operar un sistema de trading cripto totalmente automatizado que use un **modelo de razonamiento jerárquico (HRM)** para orquestar:
  - Investigación
  - Generación de señales
  - Gestión de riesgos
  - Ejecución en exchanges centralizados (CEX) y descentralizados (DEX)
- Desarrollar un sistema de trading adaptativo que opere en **múltiples timeframes**
- Implementar **razonamiento jerárquico** para optimizar decisiones de inversión
- Crear un **framework escalable** para diferentes estrategias
- Minimizar riesgos mediante análisis **multi-nivel**

---

## 2️⃣ Beneficios Esperados

✔️ Mayor precisión en predicciones de mercado  
✔️ Reducción de riesgos a través de **diversificación temporal**  
✔️ Adaptabilidad automática a diferentes condiciones de mercado  
✔️ Capacidad de razonamiento complejo sobre múltiples variables  

---

## 3️⃣ Flujo General del Sistema

```text
Nivel 4: Meta-Razonamiento (horas/días)
        ↓
Nivel 3: Análisis Estratégico (horas)
        ↓
Nivel 2: Táctica de Ejecución (minutos)
        ↓
Nivel 1: Ejecución + Gestión de Riesgo (segundos)
```

---

## 4️⃣ Jerarquía del Sistema de Trading (HRM extendido)

### 🔹 Nivel 4: Meta-Razonamiento (Horas/Días)
**Rol:** Reflexión y adaptación del sistema completo  

**Funciones principales:**
- Evaluar desempeño del HRM (Sharpe, drawdown, estabilidad por régimen)
- Detectar *concept drift*
- Ajustar parámetros globales (pesos, umbrales de riesgo, reglas)
- Selección automática de modelos (meta-aprendizaje)
- Gestión de capital y reequilibrio

**Ejemplo:**  
> La estrategia *mean reversion* pierde efectividad → el sistema reduce su peso y reasigna capital a *trend following*.

---

### 🔹 Nivel 3: Análisis Estratégico (Horas)
**Rol:** Planificación de alto nivel  

**Funciones principales:**
- Clasificación de régimen de mercado (tendencia, rango, volatilidad)
- Selección de sub-estrategias activas por régimen
- Priorización de activos (BTC, ETH, alts líquidos)
- Definición de metas intradía (exposición, riesgo máximo)

---

### 🔹 Nivel 2: Táctica de Ejecución (Minutos)
**Rol:** Conversión de decisiones estratégicas en operaciones  

**Funciones principales:**
- Composición de señales tácticas
- Cálculo de tamaño (vol-targeting, Kelly fraccionado)
- Stops y targets dinámicos
- Ajustes de posición según liquidez y volatilidad

---

### 🔹 Nivel 1: Ejecución y Gestión de Riesgo (Segundos)
**Rol:** Implementación en tiempo real  

**Funciones principales:**
- Selección de algoritmo de ejecución (TWAP, taker, iceberg)
- Control de slippage y latencia
- Circuit breakers y cancel-on-disconnect
- Monitoreo de PnL y límites de exposición

---

## 5️⃣ Arquitectura del Sistema

```text
┌─────────────────────────────────────────┐
│        NIVEL META-RAZONAMIENTO          │
│  ┌──────────────┐  ┌─────────────────┐  │
│  │ Performance  │  │ Concept Drift   │  │
│  │ Evaluation   │  │ Detection       │  │
│  └──────────────┘  └─────────────────┘  │
│  ┌──────────────┐  ┌─────────────────┐  │
│  │ Model/Strat  │  │ Capital & Risk  │  │
│  │ Selection    │  │ Allocation      │  │
│  └──────────────┘  └─────────────────┘  │
└─────────────┬───────────────────────────┘
              │ Ajustes Globales (Horas/Días)
┌─────────────▼───────────────────────────┐
│           NIVEL ESTRATÉGICO             │
│  ┌─────────────┐  ┌─────────────────┐   │
│  │ Macro       │  │ Portfolio       │   │
│  │ Analysis    │  │ Management      │   │
│  └─────────────┘  └─────────────────┘   │
└─────────────┬───────────────────────────┘
              │ Decisiones de Alto Nivel (Horas)
┌─────────────▼───────────────────────────┐
│            NIVEL TÁCTICO                │
│  ┌──────────┐ ┌──────────┐ ┌─────────┐  │
│  │Technical │ │Pattern   │ │Risk     │  │
│  │Analysis  │ │Recognition│ │Control  │  │
│  └──────────┘ └──────────┘ └─────────┘  │
└─────────────┬───────────────────────────┘
              │ Señales de Trading (Minutos)
┌─────────────▼───────────────────────────┐
│          NIVEL OPERACIONAL              │
│  ┌──────────┐ ┌──────────┐ ┌─────────┐  │
│  │Order     │ │Execution │ │Real-time│  │
│  │Management│ │Engine    │ │Monitor  │  │
│  └──────────┘ └──────────┘ └─────────┘  │
└─────────────────────────────────────────┘
```

---

## 6️⃣ Conexión entre Niveles

| Flujo | Descripción |
|-------|-------------|
| **Nivel 4 → Nivel 3** | El meta-razonamiento ajusta capital y parámetros globales |
| **Nivel 3 → Nivel 2** | El análisis estratégico selecciona sub-estrategias y universo de activos |
| **Nivel 2 → Nivel 1** | La táctica genera señales concretas (qty, stop, target) |
| **Nivel 1 → Exchange** | El nivel operacional envía y gestiona órdenes en tiempo real |

---

## 7️⃣ Estructura de Carpetas

```text
HMR/
│── docs/                # documentación
│
├── core/                # utilidades globales
│   ├── config/          # configs YAML/JSON
│   ├── logging.py
│   ├── scheduler.py
│   └── utils.py
│
├── comms/               # comunicaciones y eventos
│   ├── message_bus.py
│   ├── schemas.py
│   └── adapters/        # conectores externos (Kafka, Redis, etc.)
│
├── l4_meta/             # Meta-razonamiento (horas/días)
│   ├── drift_detector.py
│   ├── strategy_selector.py
│   ├── portfolio_allocator.py
│   └── __init__.py
│
├── l3_strategy/         # Nivel estratégico (intradiario)
│   ├── regime_classifier.py
│   ├── universe_filter.py
│   ├── exposure_manager.py
│   └── __init__.py
│
├── l2_tactic/           # Nivel táctico (señales, sizing)
│   ├── signal_generator.py
│   ├── position_sizer.py
│   ├── risk_controls.py
│   └── __init__.py
│
├── l1_operational/      # Nivel operacional (OMS/EMS)
│   ├── order_manager.py
│   ├── execution_algos.py
│   ├── realtime_risk.py
│   └── __init__.py
│
├── data/                # ingestión y almacenamiento
│   ├── connectors/      # binance, dydx, etc.
│   ├── loaders.py
│   ├── storage/         # parquet/csv
│   └── __init__.py
│
├── risk/                # librería transversal de riesgo
│   ├── limits.py
│   ├── var_es.py
│   ├── drawdown.py
│   └── __init__.py
│
├── monitoring/          # métricas y reporting
│   ├── dashboards/
│   ├── alerts.py
│   ├── telemetry.py
│   └── __init__.py
│
├── tests/               # unit & integration tests
│
└── main.py              # orquestador central
```

---

## 8️⃣ Puntos fuertes de este diseño

- Separación clara por niveles (`l4_meta/`, `l3_strategy/`, etc.) → cada capa se puede probar y mejorar de forma independiente.  
- `comms/` centralizado → define cómo se pasan mensajes entre módulos (JSON/Protobuf, colas asyncio, etc.).  
- `data/` desacoplado → cambiar de CEX a DEX no rompe los niveles.  
- `risk/` transversal → tanto L2 (stops, sizing) como L1 (hard limits) usan la misma librería.  
- `monitoring/` → logs, métricas en tiempo real, dashboards.  
- `core/` → configuración, logging, utilidades comunes.  

---

## 9️⃣ Flujo de Mensajes entre Carpetas

```text
l4_meta decide pesos de estrategias → manda mensaje strategy_update a comms
l3_strategy recibe → aplica a universo + régimen → manda tactic_targets
l2_tactic genera señales y sizing → manda execution_plan
l1_operational recibe plan → manda orders a exchange vía data/connectors
Feedback (fills, pnl_update, risk_alert) fluye de vuelta hacia arriba vía comms
```

---

✍️ **Autor:** Equipo de desarrollo HRM  
📌 **Versión:** 1.0  
📅 **Última actualización:** 2025  
