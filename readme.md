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
│── storage/                # modulo persistencia
│   ├── csv_writer.py
│   ├── sqlite_writer.py
│   └── __init__.py
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
├── l1_operational/      # Nivel operacional (OMS/EMS) - LIMPIO Y DETERMINISTA
│   ├── models.py        # Estructuras de datos tipadas (Signal, ExecutionReport, RiskAlert)
│   ├── config.py        # Configuración centralizada de límites de riesgo
│   ├── bus_adapter.py   # Interfaz con el bus de mensajes del sistema
│   ├── order_manager.py # Orquesta validación → ejecución → reporte
│   ├── risk_guard.py    # Valida límites de riesgo (sin modificar órdenes)
│   ├── executor.py      # Ejecuta órdenes pre-validadas en el exchange
│   ├── data_feed.py     # Obtiene datos de mercado y saldos
│   ├── binance_client.py # Cliente de Binance (sandbox por defecto)
│   ├── test_clean_l1.py # Pruebas de limpieza y determinismo
│   ├── README.md        # Documentación específica de L1
│   └── requirements.txt # Dependencias específicas de L1
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
│ └─ backtester.py
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
Todos los niveles trabajan sobre un **único `state`** en forma de diccionario
Cada ciclo, el sistema mantiene y actualiza un state con:

state = {
  "mercado": {...},       # precios actuales
  "estrategia": "...",    # estrategia activa (ej: agresiva/defensiva)
  "portfolio": {...},     # asignación de capital (en unidades)
  "universo": [...],      # activos disponibles
  "exposicion": {...},    # % de exposición por activo
  "senales": {...},       # señales tácticas
  "ordenes": [...],       # órdenes ejecutadas en L1
  "riesgo": {...},        # chequeo de riesgo
  "deriva": False,        # drift detection
  "ciclo_id": 1           # número de ciclo
}

Cada nivel actualiza su parte correspondiente del state.
Esto asegura trazabilidad y facilita debugging/backtesting.

---

## 🔒 L1_operational: LIMPIO Y DETERMINISTA

**L1 es el nivel de ejecución que SOLO ejecuta órdenes seguras, sin tomar decisiones estratégicas ni tácticas.**

### 🚫 Lo que L1 NO hace:
- ❌ **No modifica cantidades** de órdenes
- ❌ **No ajusta precios** de órdenes  
- ❌ **No toma decisiones** de timing de ejecución
- ❌ **No actualiza portfolio** (responsabilidad de L2/L3)
- ❌ **No actualiza datos** de mercado (responsabilidad de L2/L3)

### ✅ Lo que L1 SÍ hace:
- ✅ **Valida límites de riesgo** antes de ejecutar
- ✅ **Ejecuta órdenes** pre-validadas en el exchange
- ✅ **Genera reportes** de ejecución detallados
- ✅ **Mantiene trazabilidad** completa de todas las operaciones

### 🏗️ Nueva Arquitectura de L1:
```
L2/L3 (Señales) → Bus Adapter → Order Manager → Risk Guard → Executor → Exchange
                                    ↓
                              Execution Report → Bus Adapter → L2/L3
```

### 🧪 Verificación de Limpieza:
```bash
cd l1_operational
python test_clean_l1.py
```

Las pruebas verifican que L1 está completamente limpio y determinista.

## 9️⃣ Logging y Telemetría

Logging estructurado (JSON) → usando python-json-logger
Telemetry interna (monitoring/telemetry.py):
incr(metric_name) → contador
gauge(metric_name, value) → métrica instantánea
timing(metric_name, start_time) → latencia

## 9️⃣ Dashboard en Consola

Usamos rich para renderizar un mini-dashboard en cada ciclo

## 9️⃣ Persistencia de histórico

Cada ciclo se guarda en dos formatos:
CSV (data/historico.csv) → todas las variables del estado global por ciclo
SQLite (data/historico.db) → tabla ciclos con los mismos datos

Esto permite:
Exportar resultados para análisis en Pandas / Excel
Reproducir backtests
Consultar con SQL el rendimiento de la estrategia

---

✍️ **Autor:** Equipo de desarrollo HRM  
📌 **Versión:** 1.0  
📅 **Última actualización:** 2025  
