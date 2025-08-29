🔱 HRM — Hierarchical Reasoning Model para Trading Algorítmico
Estado: Activo · Lenguaje: Python · Dominio: Cripto / Trading · Arquitectura: Multi-nivel (L4 → L1)
Regla de oro: Si existe conflicto entre este README y los README de módulos, prevalece el README del módulo.
🧭 TL;DR
HRM es un framework de razonamiento jerárquico para trading algorítmico multiactivo (p. ej. BTC, ETH). Divide la toma de decisiones en 4 niveles —desde meta-razonamiento (L4) hasta ejecución determinista y segura (L1)— combinando reglas hard-coded y modelos IA (Logistic Regression, Random Forest, LightGBM) en L1. Soporta bus de mensajes, telemetría, persistencia histórica, dataset multitimeframe y tests robustos. Objetivo: decidir qué, cuándo y cuánto operar con trazabilidad y control de riesgo (incl. correlación BTC–ETH).
🆕 Integración con Binance (real o testnet)
✅ El sistema está totalmente implementado para operar en modo LIVE con conexión directa a Binance Spot.
Modos de operación
表格
复制
Modo	Descripción
PAPER	Simulación completa sin conexión real.
LIVE	Ejecución real en Binance Spot (requiere claves API).
REPLAY	Reproducción con datasets históricos.
Activar modo LIVE
bash
复制
export BINANCE_MODE=LIVE
export USE_TESTNET=false
export BINANCE_API_KEY=your_real_key
export BINANCE_API_SECRET=your_real_secret
python main.py
1️⃣ Objetivo del proyecto
Tomar decisiones de trading razonadas y trazables para múltiples activos (BTC, ETH) mediante una jerarquía de agentes.
Aprender qué señales funcionan bajo distintos regímenes y cómo combinar niveles (L2/L3) para optimizar ejecución en L1 con modelos IA.
Minimizar riesgos con análisis multinivel, capa dura de seguridad en L1 y gestión de correlación BTC–ETH.
Crear un framework reutilizable para distintos universos de activos líquidos.
Qué queremos aprender a nivel de sistema
Si el razonamiento multietapa mejora la estabilidad frente a un agente monolítico.
Qué señales funcionan en cada régimen y cómo combinarlas en L2/L3.
Cómo distribuir capital/ponderaciones entre modelos/estrategias y detectar concept drift en L4.
2️⃣ Beneficios esperados
Mayor precisión mediante composición multiasset y modelos IA (LogReg, RF, LightGBM).
Reducción de riesgo vía diversificación temporal, límite rígido en L1 y gestión de correlación BTC–ETH.
Adaptabilidad automática a distintos regímenes de mercado.
Razonamiento multi-variable con métricas granulares por activo (latencia, slippage, tasa de éxito).
⚙️ 3️⃣ Flujo general (visión de tiempos)
Nivel 4: Meta-Razonamiento — horas/días
Nivel 3: Análisis Estratégico — horas
Nivel 2: Táctica de Ejecución — minutos
Nivel 1: Ejecución + Gestión de Riesgo — segundos
🧭 4️⃣ Jerarquía del sistema (HRM extendido)
🔮 Nivel 4 — Meta-Razonamiento (horas/días)
Rol: Reflexión y adaptación del sistema completo.
Funciones: Evaluación de desempeño (Sharpe, drawdown), detección de drift, selección de modelos/estrategias, asignación de capital y ajustes globales.
Ejemplo: Si mean reversion pierde eficacia, reducir su peso y reasignar capital a trend-following.
🧭 Nivel 3 — Análisis Estratégico (horas)
Rol: Planificación de alto nivel.
Funciones: Clasificación de régimen (tendencia/rango/volatilidad), selección de sub-estrategias, priorización de activos (BTC, ETH), metas intradía (exposición, riesgo máximo).
🚧 Por desarrollar:
Integración con indicadores macroeconómicos (FRED, OECD).
Modelos de Black-Litterman para asignación dinámica.
Detección de eventos de riesgo sistémico.
Escenarios de estrés y rebalanceo automático.
⚔️ Nivel 2 — Táctica de Ejecución (minutos)
Rol: Convertir decisiones estratégicas en operaciones concretas.
Funciones: Composición de señales, position sizing (vol-targeting, Kelly fracc.), stops/targets dinámicos, ajustes por liquidez/volatilidad.
⚙️ Nivel 1 — Ejecución y Riesgo (segundos)
Rol: Implementación determinista con capa dura de seguridad y modelos IA.
Funciones clave:
Validación de límites por símbolo (stop-loss, exposición, correlación BTC–ETH).
Filtrado de señales con IA (modelo1_lr.pkl, modelo2_rf.pkl, modelo3_lgbm.pkl).
Ejecución optimizada (fraccionamiento, timing, reducción de slippage).
Envío de órdenes con timeouts/retries.
Reportes y métricas por activo (BTC/USDT, ETH/USDT): latencia, slippage, exposición, tasas de éxito.
🆕 Features incluidas (actualizado)
表格
复制
Tipo	Descripción
Precio	delta_close, EMA/SMA
Volumen	volumen relativo
Momentum	RSI, MACD
Multi-timeframe	1m + 5m
Cross-asset	ETH/BTC ratio, correlación rolling, divergencias
Real-time data	Desde Binance Spot (modo LIVE) o testnet
⚙️ Puesta en marcha (actualizado)
Requisitos
Python 3.10+
Cuenta en Binance (Spot o Futures)
Credenciales/API keys (ya cargadas en .env o variables de entorno)
Instalación rápida
bash
复制
# 1) Clonar
git clone https://github.com/t3rm1nus/HRM.git
cd HRM

# 2) Entorno
python -m venv .venv && source .venv/bin/activate

# 3) Dependencias
pip install -r l1_operational/requirements.txt

# 4) Configurar entorno (ejemplo .env)
export BINANCE_API_KEY=your_real_key
export BINANCE_API_SECRET=your_real_secret
export BINANCE_MODE=LIVE
export USE_TESTNET=false

# 5) Ejecutar
python main.py
✅ Buenas prácticas de riesgo (resumen actualizado)
表格
复制
Concepto	Valor real
Stop-loss	Obligatorio
Límites por trade	BTC: 0.05, ETH: 1.0
Exposición máxima	BTC: 20%, ETH: 15%
Correlación BTC-ETH	Monitoreada en tiempo real
Modo LIVE	Implementado y validado
Determinismo	Una orden por señal → si falla → rechazo y reporte
Separación L2/L3 ≠ L1	Responsabilidades claramente separadas
🏗️ 5️⃣ Arquitectura (ASCII actualizada)
复制
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
│  ┌─────────────┐  ┌─────────────────┐   │
│  │ Sentiment   │  │ Risk Appetite   │   │
│  │ Analysis    │  │ Calculator      │   │
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
│ Hard-coded Safety Layer + Order Manager (determinista)       │
│ AI Models (LogReg, RF, LightGBM) + Multiasset Execution      │
│ Executor determinista → Exchange (Binance real o testnet)    │
└──────────────────────────────────────────────────────────────┘
🔗 6️⃣ Conexión entre niveles (resumen actualizado)
表格
复制
Flujo	Descripción
L4 → L3	Ajuste de capital y parámetros globales
L3 → L2	Selección de sub-estrategias y universo (BTC, ETH)
L2 → L1	Señales concretas (cantidad, stop, target) por símbolo
L1 → Exchange	Envío/gestión de órdenes en tiempo real para BTC/USDT y ETH/USDT desde Binance Spot o testnet


## 📂 7️⃣ Estructura de carpetas

```text
HRM/
│── docs/                      
│
│── storage/                   
│   ├── csv_writer.py
│   ├── sqlite_writer.py
│   └── __init__.py
│
├── core/                      
│   ├── logging.py
│   ├── scheduler.py
│   └── utils.py
│
├── comms/                     
│   ├── config/                
│   ├── message_bus.py
│   ├── schemas.py
│   └── adapters/
│
├── l4_meta/                   
│   ├── drift_detector.py
│   ├── strategy_selector.py
│   ├── portfolio_allocator.py
│   └── __init__.py
│
├── l3_strategy/              
│   ├── __init__.py
│   ├── README.md  
│   ├── models.py
│   ├── config.py
│   ├── strategic_processor.py
│   ├── bus_integration.py
│   ├── performance_tracker.py
│   ├── metrics.py
│   ├── procesar_l3.py
│   ├── ai_model_loader.py
│   └── ai_models/
│       ├── unified_decision_model.py
│       ├── regime_detector.py
│       └── risk_assessor.py
│
├── l2_tactic/                 
│   ├── signal_generator.py
│   ├── position_sizer.py
│   ├── risk_controls.py
│   └── __init__.py
│
├── l1_operational/            
│   ├── models.py
│   ├── config.py
│   ├── bus_adapter.py
│   ├── order_manager.py
│   ├── risk_guard.py
│   ├── executor.py
│   ├── data_feed.py
│   ├── binance_client.py
│   ├── ai_models/
│   │   ├── modelo1_lr.pkl
│   │   ├── modelo2_rf.pkl
│   │   └── modelo3_lgbm.pkl
│   ├── test_clean_l1_multiasset.py
│   ├── README.md
│   └── requirements.txt
│
├── models/                    
│   ├── L1/
│   │   ├── modelo1_lr.pkl
│   │   ├── modelo2_rf.pkl
│   │   └── modelo3_lgbm.pkl
│   ├── L2/
│   ├── L3/
│   └── L4/
│
├── data/                      
│   ├── connectors/
│   │   └── binance_connector.py
│   ├── loaders.py
│   ├── storage/
│   └── __init__.py
│
├── risk/                      
│   ├── limits.py
│   ├── var_es.py
│   ├── drawdown.py
│   └── __init__.py
│
├── monitoring/                
│   ├── dashboards/
│   ├── alerts.py
│   ├── telemetry.py
│   └── __init__.py
│
├── tests/                     
│   └── backtester.py
└── main.py
```

> **Nota:** Esta estructura resume el proyecto real y es suficiente para navegar y extender el código.

---

## 🔁 8️⃣ Flujo de mensajes y state global

Cada ciclo trabaja sobre un único `state` (dict). Cada nivel actualiza su sección para trazabilidad y debugging.

```python
state = {
    "mercado": {...},       # precios actuales por símbolo (BTC, ETH)
    "estrategia": "...",    # estrategia activa (agresiva/defensiva)
    "portfolio": {...},     # asignación de capital
    "universo": [...],      # activos (BTC/USDT, ETH/USDT)
    "exposicion": {...},    # % exposición por activo
    "senales": {...},       # señales tácticas por símbolo
    "ordenes": [...],       # órdenes ejecutadas en L1
    "riesgo": {...},        # chequeo de riesgo (incluye correlación BTC-ETH)
    "deriva": False,        # drift detection
    "ciclo_id": 1
}
```

**Flujo L1 (ejecución determinista):**
L2/L3 → Bus Adapter → Order Manager → Hard-coded Safety → AI Models (LogReg, RF, LightGBM) → Risk Rules → Executor → Exchange → Execution Report → Bus Adapter → L2/L3

---

## ✅ 9️⃣ L1\_operational — “limpio y determinista”

**L1 NO hace**

* ❌ No modifica cantidades ni precios de señales estratégicas.
* ❌ No decide estrategia ni táctica.
* ❌ No actualiza portfolio completo (responsabilidad de L2/L3).
* ❌ No recolecta ni procesa datos de mercado (responsabilidad de L2/L3).

**L1 SÍ hace**

* ✅ Validar límites de riesgo por símbolo (stop-loss, exposición, correlación BTC-ETH).
* ✅ Filtrar señales con modelos IA para confirmar tendencias.
* ✅ Ejecutar órdenes pre-validadas (modo PAPER simulado).
* ✅ Generar reportes detallados por activo.
* ✅ Mantener trazabilidad completa con métricas por símbolo.

**Verificación de limpieza:**
`python l1_operational/test_clean_l1_multiasset.py`

---

## 🔌 Mensajería, logging y telemetría

* **Mensajería:** `comms/` define esquemas y bus (JSON/Protobuf). Colas asyncio; adapters Kafka/Redis opcionales.
* **Logging estructurado:** JSON (ej.: `python-json-logger`) con etiquetas por símbolo (`[BTC]`, `[ETH]`).
* **Telemetría (monitoring/telemetry.py):**

  * `incr(name)` → contadores (órdenes por símbolo)
  * `gauge(name, value)` → métricas instantáneas (exposición, correlación)
  * `timing(name, start)` → latencias por ejecución

**Dashboard consola:** Visualización con `rich` por ciclo (métricas por activo).

---

## 🗃️ Persistencia de histórico

Cada ciclo se guarda en:

* **CSV:** `data/historico.csv` (todas las variables del `state`).
* **SQLite:** `data/historico.db` (tabla `ciclos` con los mismos datos).

Permite exportar a pandas/Excel, reproducir backtests y consultar con SQL.

---

## 🧪 Dataset & features (BTC/USDT, ETH/USDT)

Generador de features en `data/loaders.py`. Soporta 1m + 5m (multi-timeframe). Índice `datetime` y columna `close`.

**Features incluidas**

* Precio: `delta_close`, `ema_10/20`, `sma_10/20`
* Volumen: `vol_rel` vs media N (20)
* Momentum: `rsi`, `macd`, `macd_signal`, `macd_hist`
* Multi-timeframe: 1m + 5m (`_5m`)
* Cruzadas: `ETH/BTC ratio`, correlación rolling, divergencias

**Uso básico**

```python
import pandas as pd
from data.loaders import prepare_features

# 1) Cargar velas 1m
df_btc_1m = pd.read_csv("data/btc_1m.csv", parse_dates=["timestamp"], index_col="timestamp")
df_eth_1m = pd.read_csv("data/eth_1m.csv", parse_dates=["timestamp"], index_col="timestamp")

# 2) Generar features 1m+5m y split temporal (80/20 por defecto)
train_btc, test_btc = prepare_features(df_btc_1m, test_size=0.2, symbol="BTC")
train_eth, test_eth = prepare_features(df_eth_1m, test_size=0.2, symbol="ETH")

# 3) Guardar datasets
train_btc.to_csv("data/btc_features_train.csv")
test_btc.to_csv("data/btc_features_test.csv")
train_eth.to_csv("data/eth_features_train.csv")
test_eth.to_csv("data/eth_features_test.csv")
```

> **Nota:** Si ya tienes velas 5m, pásalas como `df_5m` para evitar resampleo. Si tu CSV trae `BTC_close` o `ETH_close`, `normalize_columns` lo mapea a `close` automáticamente.

---

## ⚙️ Puesta en marcha

**Requisitos**

* Python 3.10+
* Cuenta de exchange (modo sandbox recomendado para L1)
* Credenciales/API Keys (env vars o `.env`)
* `pip`, `venv`

**Instalación rápida**

```bash
# 1) Clonar
git clone https://github.com/t3rm1nus/HRM.git
cd HRM

# 2) Entorno
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3) Dependencias (L1)
pip install -r l1_operational/requirements.txt

# 4) (Opcional) Dependencias extra según adapters
# pip install -r requirements.txt
```

**Ejecución (demo)**

```bash
python main.py
```

Configurar parámetros y límites en `core/config/` y en variables de entorno.

---

## ✅ Buenas prácticas de riesgo (resumen)

* **Hard limits en L1:** Stop-loss obligatorio.
* Límites por trade: BTC: `0.05` max, ETH: `1.0` max.
* Exposición máxima: BTC: `20%`, ETH: `15%`.
* Chequeos de liquidez/saldo, drawdown y correlación BTC-ETH.
* **Determinismo:** Una oportunidad de orden por señal; si no cumple reglas → rechazo y reporte.
* **Separación de responsabilidades:** Señal (L2/L3) ≠ Ejecución (L1).
* **Backtesting:** Histórico persistido + state reproducible.

---

## 🧩 Tests e integración

* Pruebas de limpieza L1: `l1_operational/test_clean_l1_multiasset.py`
* Backtester E2E: `tests/backtester.py`
* Métricas/alertas: `monitoring/` (métricas por símbolo y correlación)

---

## 🛣️ Roadmap (alto nivel)

* Meta-aprendizaje para selección dinámica de estrategias (L4)
* Mejores clasificadores de régimen (L3)
* Ensamble multi-señal robusto (L2)
* Integración multi-exchange/DEX y simulador de slippage (L1)
* Dashboards web y alertas proactivas con métricas por activo

---

## 👥 Autoría y licencia

**Autoría:** Equipo de desarrollo HRM
**Versión:** 1.0
**Última actualización:** 2025
**Licencia:** Ver `LICENSE` si aplica

---

## 📝 Notas finales

Este README está diseñado para ser **autosuficiente**: describe la jerarquía, arquitectura, flujos, estructura de código, dataset, telemetría, persistencia y puesta en marcha para que un agente externo/colaborador comprenda y opere el proyecto sin necesidad inmediata de otros documentos.
Si quieres, te lo puedo convertir a un `README.md` listo para push a GitHub, añadir badges (build, coverage), o generar una versión en inglés. ¿Qué prefieres ahora?
