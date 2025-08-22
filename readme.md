# HRM — Hierarchical Reasoning Model para Trading Algorítmico

> **Estado**: Activo · **Lenguaje**: Python · **Dominio**: Cripto/Trading · **Arquitectura**: Multi‑nivel (L4→L1)  
> **Regla de oro**: *Si existe conflicto entre este README y los README de módulos, **prevalece el README del módulo**.*

---

## 🧭 TL;DR

HRM es un framework de **razonamiento jerárquico** para trading. Divide el problema en **4 niveles** que van desde la reflexión global (L4) hasta la ejecución determinista (L1).  
El objetivo es **decidir qué, cuándo y cuánto** operar, **limitando riesgo** mediante reglas duras en L1 y **aprendiendo** en niveles superiores (L2–L4). Incluye **bus de mensajes**, **telemetría**, **persistencia histórica**, **dataset multitimeframe** y **tests**.

---

## 1️⃣ Objetivo del proyecto

- Tomar decisiones de **trading razonadas y trazables** mediante una **jerarquía de agentes**.
- Aprender **qué señales mantienen performance** bajo distintos regímenes de mercado y cómo combinar niveles.
- Minimizar riesgos con **análisis multinivel** y **capa dura de seguridad** en ejecución.
- Crear un **framework reutilizable** para diferentes estrategias y universos de activos.

**Qué queremos aprender a nivel del sistema**:
- Si el **razonamiento multietapa** mejora la estabilidad vs. un agente monolítico.
- **Qué señales** funcionan en cada régimen y cómo **combinarlas** en L2/L3.
- Cómo **distribuir peso/capital** entre modelos/estrategias y detectar **concept drift** en L4.

---

## 2️⃣ Beneficios esperados

- Mayor **precisión** en predicciones (composición de señales).  
- **Reducción de riesgos** vía diversificación temporal y capa L1.  
- **Adaptabilidad** automática a distintos regímenes.  
- Capacidad de **razonamiento complejo** multi‑variable.

---

## 3️⃣ Flujo general (visión de tiempos)

```
Nivel 4: Meta‑Razonamiento (horas/días)
        ↓
Nivel 3: Análisis Estratégico (horas)
        ↓
Nivel 2: Táctica de Ejecución (minutos)
        ↓
Nivel 1: Ejecución + Gestión de Riesgo (segundos)
```

---

## 4️⃣ Jerarquía del sistema (HRM extendido)

### Nivel 4 — Meta‑Razonamiento (horas/días)
**Rol**: Reflexión y adaptación del sistema completo.  
**Funciones**: evaluación de desempeño (Sharpe, drawdown), **concept drift**, **selección de modelos/estrategias**, **asignación de capital** y **ajustes globales**.  
**Ejemplo**: si *mean reversion* pierde eficacia, **reduce su peso** y reasigna capital a *trend‑following*.

### Nivel 3 — Análisis Estratégico (horas)
**Rol**: Planificación de alto nivel.  
**Funciones**: **clasificación de régimen** (tendencia/rango/volatilidad), **selección de sub‑estrategias**, priorización de activos (BTC, ETH, alts líquidas), metas intradía (exposición, riesgo máximo).

### Nivel 2 — Táctica de Ejecución (minutos)
**Rol**: Convertir las decisiones estratégicas en operaciones concretas.  
**Funciones**: **composición de señales**, **position sizing** (vol‑targeting, Kelly fraccionado), **stops/targets dinámicos**, **ajustes por liquidez/volatilidad**.

### Nivel 1 — Ejecución y Riesgo (segundos)
**Rol**: **Implementación determinista** con **capa dura de seguridad**.  
**Funciones**: validación de **límites de riesgo**, envío de órdenes con **timeouts/retries**, **reportes** de ejecución y **métricas** (latencia, rechazos, parciales, snapshot de saldos).

---

## 5️⃣ Arquitectura (ASCII)

```
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
              │
┌─────────────▼────────────── Nivel Operacional ───────────────┐
│ Hard‑coded Safety Layer + Order Manager (determinista)       │
│ Executor determinista → Exchange                             │
└──────────────────────────────────────────────────────────────┘
```

---

## 6️⃣ Conexión entre niveles

| Flujo | Descripción |
|---|---|
| **L4 → L3** | Ajuste de capital y parámetros globales |
| **L3 → L2** | Selección de sub‑estrategias y universo de activos |
| **L2 → L1** | Señales concretas (qty, stop, target) |
| **L1 → Exchange** | Envío/gestión de órdenes en tiempo real |

---

## 7️⃣ Estructura de carpetas

```
HRM/
│── docs/                 # documentación
│
│── storage/              # módulo de persistencia
│   ├── csv_writer.py
│   ├── sqlite_writer.py
│   └── __init__.py
│
├── core/                 # utilidades globales
│   ├── config/           # configs YAML/JSON
│   ├── logging.py
│   ├── scheduler.py
│   └── utils.py
│
├── comms/                # comunicaciones y eventos
│   ├── message_bus.py
│   ├── schemas.py
│   └── adapters/         # conectores externos (Kafka, Redis, etc.)
│
├── l4_meta/              # meta-razonamiento (horas/días)
│   ├── drift_detector.py
│   ├── strategy_selector.py
│   ├── portfolio_allocator.py
│   └── __init__.py
│
├── l3_strategy/          # nivel estratégico (intradía)
│   ├── regime_classifier.py
│   ├── universe_filter.py
│   ├── exposure_manager.py
│   └── __init__.py
│
├── l2_tactic/            # nivel táctico (señales, sizing)
│   ├── signal_generator.py
│   ├── position_sizer.py
│   ├── risk_controls.py
│   └── __init__.py
│
├── l1_operational/       # nivel operacional (OMS/EMS) - limpio y determinista
│   ├── models.py         # Signal, ExecutionReport, RiskAlert, OrderIntent
│   ├── config.py         # límites de riesgo centralizados
│   ├── bus_adapter.py    # bus asíncrono (topics: signals, reports, alerts)
│   ├── order_manager.py  # orquesta validación → ejecución → reporte
│   ├── risk_guard.py     # stop-loss, capital, liquidez, exposición
│   ├── executor.py       # timeouts/retry + métricas
│   ├── data_feed.py      # datos de mercado y saldos
│   ├── binance_client.py # cliente Binance (sandbox por defecto)
│   ├── test_clean_l1.py  # pruebas de limpieza y determinismo
│   ├── README.md         # doc específica de L1
│   └── requirements.txt  # dependencias L1
│
├── data/                 # ingestión y almacenamiento
│   ├── connectors/       # binance, dydx, etc.
│   ├── loaders.py
│   ├── storage/          # parquet/csv
│   └── __init__.py
│
├── risk/                 # librería transversal de riesgo
│   ├── limits.py
│   ├── var_es.py
│   ├── drawdown.py
│   └── __init__.py
│
├── monitoring/           # métricas y reporting
│   ├── dashboards/
│   ├── alerts.py
│   ├── telemetry.py
│   └── __init__.py
│
├── tests/                # unit & integration tests
│   └── backtester.py
└── main.py               # orquestador central
```

> **Nota:** Esta estructura resume el proyecto real y es suficiente para navegar y extender el código.

---

## 8️⃣ Flujo de mensajes y *state* global

Cada ciclo del sistema trabaja sobre un **único `state`** (diccionario) y **cada nivel actualiza su sección**. Esto garantiza **trazabilidad** y facilita **debugging/backtesting**.

```python
state = {
    "mercado": {...},       # precios actuales
    "estrategia": "...",    # estrategia activa (agresiva/defensiva)
    "portfolio": {...},     # asignación de capital (unidades)
    "universo": [...],      # activos disponibles
    "exposicion": {...},    # % exposición por activo
    "senales": {...},       # señales tácticas
    "ordenes": [...],       # órdenes ejecutadas en L1
    "riesgo": {...},        # chequeo de riesgo
    "deriva": False,        # drift detection
    "ciclo_id": 1           # número de ciclo
}
```

**Flujo L1 (ejecución determinista):**  
`L2/L3 (Señales) → Bus Adapter → Order Manager → Risk Guard → Executor → Exchange → Execution Report → Bus Adapter → L2/L3`

---

## 9️⃣ L1_operational — “limpio y determinista”

> **L1 SOLO ejecuta órdenes seguras.** No decide estrategia ni táctica.

**Lo que L1 _no_ hace**
- ❌ No modifica cantidades
- ❌ No ajusta precios
- ❌ No decide *timing*
- ❌ No actualiza portfolio
- ❌ No actualiza datos de mercado

**Lo que L1 _sí_ hace**
- ✅ Valida **límites de riesgo** antes de ejecutar
- ✅ **Ejecuta** órdenes pre‑validadas en el exchange
- ✅ **Genera reportes** detallados de ejecución
- ✅ Mantiene **trazabilidad completa**

**Verificación de limpieza**
```bash
python l1_operational/test_clean_l1.py
```

---

## 🔌 Mensajería, logging y telemetría

- **Mensajería**: `comms/` define **esquemas** y el **bus** (JSON/Protobuf; colas asyncio; adapters Kafka/Redis si se desea).
- **Logging estructurado**: JSON (p. ej. `python-json-logger`).
- **Telemetría (`monitoring/telemetry.py`)**:  
  - `incr(name)` → contadores  
  - `gauge(name, value)` → métricas instantáneas  
  - `timing(name, start)` → latencias

- **Dashboard en consola**: con `rich` para un mini‑panel por ciclo.

---

## 🗃️ Persistencia de histórico

Cada ciclo se guarda en:
- **CSV**: `data/historico.csv` (todas las variables del `state`)
- **SQLite**: `data/historico.db` (tabla `ciclos` con los mismos datos)

Esto permite **exportar a Pandas/Excel**, **reproducir backtests** y **consultar con SQL**.

---

## 🧪 Dataset & features (BTC/USDT)

Generador de *features* en `data/loaders.py` (limpio y autocontenido).  
Soporta BTC/USDT o BTC/USD con **índice datetime** y columna **`close`** (opcional `volume`).

**Features incluidas**
- **Precio**: `delta_close`, `ema_10/20`, `sma_10/20`
- **Volumen**: `vol_rel` vs. media *N* (20 por defecto)
- **Momentum**: `rsi`, `macd`, `macd_signal`, `macd_hist`
- **Multi‑timeframe**: 1m + 5m (sufijos `_5m`, reindex 1m)

**Uso básico**
```python
import pandas as pd
from data.loaders import prepare_btc_features

# 1) Cargar velas 1m con índice datetime y columna 'close'
df_1m = pd.read_csv("data/btc_1m.csv", parse_dates=["timestamp"], index_col="timestamp")

# 2) Generar features 1m+5m y split temporal (80/20 por defecto)
train, test = prepare_btc_features(df_1m, test_size=0.2)

# 3) Guardar datasets
train.to_csv("data/btc_features_train.csv")
test.to_csv("data/btc_features_test.csv")
```

> Si ya tienes velas 5m, puedes pasarlas como `df_5m` y evitar resampleo.  
> Si tu CSV trae `BTC_close`, `normalize_btc_columns` lo mapea a `close` automáticamente.

---

## ⚙️ Puesta en marcha

### Requisitos
- Python 3.10+ recomendado
- Cuenta de exchange (modo **sandbox** si es posible) si vas a **ejecutar** L1
- Credenciales/API Keys (usa variables de entorno o `.env`)
- `pip`, `venv` o `uv` (opcional)

### Instalación rápida
```bash
# 1) Clonar
git clone https://github.com/t3rm1nus/HRM.git
cd HRM

# 2) Entorno
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)

# 3) Dependencias (L1)
pip install -r l1_operational/requirements.txt

# 4) (Opcional) Dependencias extra según conectores/adapters
# pip install -r requirements.txt  # si existe en la raíz / módulos
```

### Ejecución (modo demo)
```bash
python main.py
```
Configura en `core/config/` y variables de entorno los parámetros de conexión y límites de riesgo.

---

## ✅ Buenas prácticas de riesgo (resumen)

- **Hard limits** en L1: stop‑loss obligatorio, límites de capital por trade, exposición máxima, chequeos de liquidez/saldo y drawdown.
- **Determinismo** en ejecución: una oportunidad de orden por señal; si no cumple reglas → **rechazo** y **reporte**.
- **Separación de responsabilidades**: señal (L2/L3) ≠ ejecución (L1).
- **Backtesting** con histórico persistido y *state* reproducible.

---

## 🧩 Tests e integración

- **Pruebas de limpieza L1**: `l1_operational/test_clean_l1.py`
- **Backtester** de extremo a extremo: `tests/backtester.py`
- **Métricas/alertas**: `monitoring/`

---

## 📈 Roadmap (alto nivel)

- [ ] Meta‑aprendizaje para **selección dinámica de estrategias** (L4)
- [ ] Mejores **clasificadores de régimen** (L3)
- [ ] **Ensamble multi‑señal** robusto (L2)
- [ ] Integración multi‑exchange/DEX y **simulador de *slippage*** (L1)
- [ ] Dashboards enriquecidos (web) y **alertas proactivas**

---

## 👥 Autoría y licencia

- Autoría: **Equipo de desarrollo HRM**
- Versión: **1.0**
- Última actualización: **2025**
- Licencia: ver archivo `LICENSE` si aplica

---

> **Envío a otras IA**: Este README está diseñado para ser **autosuficiente**. Describe jerarquía, arquitectura, flujos, estructura de código, dataset, telemetría, persistencia y puesta en marcha para que un agente externo pueda **comprender y operar** el proyecto sin consultar otros documentos.
