# 🔱 HRM — Hierarchical Reasoning Model para Trading Algorítmico
**Estado: PRODUCCIÓN** · **Lenguaje:** Python 3.10+ · **Dominio:** Cripto Trading · **Arquitectura:** L2 Táctico + L1 Operacional

## 🧭 TL;DR
HRM es un sistema de trading algorítmico **REAL Y FUNCIONAL** que opera con BTC y ETH en Binance Spot. Combina **análisis técnico avanzado**, **modelos FinRL pre-entrenados**, **gestión dinámica de riesgo**, **stop-loss/take-profit automáticos** y **ejecución determinista**. El sistema genera señales inteligentes cada 10 segundos, calcula posiciones óptimas y ejecuta órdenes con controles de seguridad multi-nivel.

## ✅ SISTEMA OPERATIVO - FUNCIONALIDAD REAL
**🚀 El sistema HRM está completamente operativo y ejecutándose en producción:**
- ✅ **Conexión real a Binance Spot** (modo LIVE y TESTNET)
- ✅ **Generación de señales cada 10 segundos** con indicadores técnicos
- ✅ **Modelos IA integrados** (FinRL + análisis técnico)  
- ✅ **Gestión de portfolio automática** con tracking en CSV
- ✅ **Logging persistente** completo en data/logs/
- ✅ **Controles de riesgo dinámicos** y stops inteligentes
- ✅ **Stop-Loss y Take-Profit automáticos** integrados
- ✅ **Costos reales de trading** (comisiones 0.1% Binance)
- ✅ **Monitoreo de posiciones** en tiempo real
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
Cómo distribuir capital/ponderaciones entre modelos/estrategias.
2️⃣ Beneficios esperados
Mayor precisión mediante composición multiasset y modelos IA (LogReg, RF, LightGBM).
Reducción de riesgo vía diversificación temporal, límite rígido en L1 y gestión de correlación BTC–ETH.
Adaptabilidad automática a distintos regímenes de mercado.
Razonamiento multi-variable con métricas granulares por activo (latencia, slippage, tasa de éxito).
⚙️ 3️⃣ Flujo general (visión de tiempos)
Nivel 3: Análisis Estratégico — horas
Nivel 2: Táctica de Ejecución — minutos
Nivel 1: Ejecución + Gestión de Riesgo — segundos
## 🏗️ ARQUITECTURA REAL DEL SISTEMA

### 🎯 **NIVEL 2 - TÁCTICO (L2)** ✅ IMPLEMENTADO
**Rol:** Generación inteligente de señales de trading
**Funciones operativas:**
- ✅ **Análisis técnico multi-timeframe** (RSI, MACD, Bollinger Bands)
- ✅ **Modelos FinRL pre-entrenados** con ensemble de predicciones
- ✅ **Composición de señales** con pesos dinámicos
- ✅ **Position sizing** con Kelly Criterion y vol-targeting
- ✅ **Controles de riesgo pre-ejecución** (stops, correlación, drawdown)
- ✅ **Stop-Loss y Take-Profit dinámicos** basados en volatilidad y confianza
- ✅ **Cálculo automático de SL/TP** por señal generada

### ⚙️ **NIVEL 1 - OPERACIONAL (L1)** ✅ IMPLEMENTADO  
**Rol:** Ejecución determinista y segura de órdenes
**Funciones operativas:**
- ✅ **Validación de señales** con 3 modelos IA (LogReg, RF, LightGBM)
- ✅ **Trend AI** con ensemble de modelos ML
- ✅ **Gestión de portfolio** automática (BTC, ETH, USDT)
- ✅ **Conexión a Binance Spot** (real y testnet)
- ✅ **Order management** con timeouts y reintentos
- ✅ **Logging persistente** y métricas en tiempo real
- ✅ **Monitoreo de posiciones** con activación automática de SL/TP
- ✅ **Costos reales de trading** (comisiones 0.1% Binance)
- ✅ **RiskControlManager** integrado para gestión de riesgo

### 🚧 **NIVEL L3** - NO IMPLEMENTADO
- **L3 Estratégico:** Planificado pero no desarrollado
- **Nota:** El sistema actual opera efectivamente con L2+L1
- ✅ **Modelos IA L1:** **FUNCIONALES** (LogReg, RF, LightGBM en models/L1/)

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
## 🚀 EJECUCIÓN DEL SISTEMA

### ⚡ **INICIO RÁPIDO**
```bash
# 1) Configurar variables de entorno
export BINANCE_API_KEY=your_api_key
export BINANCE_API_SECRET=your_secret_key
export USE_TESTNET=true  # false para modo LIVE

# 2) Ejecutar sistema principal
python main.py

# 3) Para ejecución nocturna continua
python run_overnight.py
```

### 📊 **FUNCIONAMIENTO EN TIEMPO REAL**
El sistema ejecuta un **ciclo principal cada 10 segundos**:

1. **📈 Recolección de datos:** Obtiene OHLCV de Binance para BTC/ETH
2. **🧮 Cálculo de indicadores:** RSI, MACD, Bollinger Bands, volatilidad
3. **🤖 Procesamiento L2:** Genera señales con modelos FinRL + análisis técnico  
4. **🛡️ Cálculo SL/TP:** Stop-loss y take-profit dinámicos por señal
5. **⚙️ Procesamiento L1:** Valida señales y ejecuta órdenes seguras
6. **💰 Actualización portfolio:** Tracking automático con costos reales
7. **🔍 Monitoreo posiciones:** Activación automática de SL/TP
8. **📝 Logging persistente:** Guarda métricas en data/logs/ y data/portfolio/
- L2/L1 se ejecuta **cada 10 segundos** de forma independiente.
- L3 se ejecuta **cada 10 minutos** en segundo plano.
- Si L3 falla o se retrasa >30s, L2 sigue usando la última estrategia conocida (fallback).

### **VENTAJAS DEL FALLBACK**
- L2/L1 nunca se bloquea si L3 falla.
- Última estrategia válida de L3 se mantiene.
- Logs centralizados registran errores y warnings.

### 🎛️ **MODOS DE OPERACIÓN**
| Modo | Descripción | Activación |
|------|-------------|------------|
| **TESTNET** | Binance testnet (recomendado) | `USE_TESTNET=true` |
| **LIVE** | Binance Spot real | `USE_TESTNET=false` |
| **PAPER** | Simulación local | Configuración interna |

✅ Buenas prácticas de riesgo (resumen actualizado)
表格
复制
Concepto	Valor real
Stop-loss	Obligatorio + automático
Take-profit	Dinámico basado en volatilidad
Límites por trade	BTC: 0.05, ETH: 1.0
Exposición máxima	BTC: 20%, ETH: 15%
Correlación BTC-ETH	Monitoreada en tiempo real
Costos reales	Comisiones 0.1% Binance aplicadas
Monitoreo posiciones	Activación automática SL/TP
Modo LIVE	Implementado y validado
Determinismo	Una orden por señal → si falla → rechazo y reporte
Separación L2/L3 ≠ L1	Responsabilidades claramente separadas

🏗️ 5️⃣ Arquitectura (ASCII actualizada)
┌─────────────▼───────────────────────────┐
│           NIVEL ESTRATÉGICO (L3)       │
│  ┌─────────────┐  ┌─────────────────┐   │
│  │ Macro       │  │ Portfolio       │   │
│  │ Analysis    │  │ Management      │   │
│  └─────────────┘  └─────────────────┘   │
│  ┌─────────────┐  ┌─────────────────┐   │
│  │ Sentiment   │  │ Risk Appetite   │   │
│  │ Analysis    │  │ Calculator      │   │
│  └─────────────┘  └─────────────────┘   │
│  ⚡ Ejecuta periódicamente (10 min)      │
│  ⚡ Fallback automático si L3 falla      │
└─────────────┬───────────────────────────┘
              │ Decisiones Estratégicas → L2
┌─────────────▼───────────────────────────┐
│            NIVEL TÁCTICO (L2)           │
│  ┌──────────┐ ┌──────────┐ ┌─────────┐  │
│  │Technical │ │Pattern   │ │Risk     │  │
│  │Analysis  │ │Recognition│ │Control  │  │
│  └──────────┘ └──────────┘ └─────────┘  │
│  ⚡ Loop principal cada 10 segundos      │
│  ⚡ Genera señales tácticas basadas en L3│
└─────────────┬───────────────────────────┘
              │ Señales de Trading → L1
┌─────────────▼────────────── Nivel Operacional (L1) ───────────────┐
│ Hard-coded Safety Layer + Order Manager (determinista)             │
│ AI Models (LogReg, RF, LightGBM) + Multiasset Execution           │
│ Executor determinista → Exchange (Binance real o testnet)         │
│ ⚡ Recibe señales L2 y valida límites de riesgo                    │
│ ⚡ Ejecuta órdenes pre-validadas, mantiene trazabilidad completa   │
└───────────────────────────────────────────────────────────────────┘



🔗 6️⃣ Conexión entre niveles (resumen actualizado)

Flujo	Descripción
L3 → L2	Selección de sub-estrategias y universo (BTC, ETH)
L2 → L1	Señales concretas (cantidad, stop, target) por símbolo
L1 → Exchange	Envío/gestión de órdenes en tiempo real para BTC/USDT y ETH/USDT desde Binance Spot o testnet

### MÓDULOS CORE ✅ IMPLEMENTADOS
Funcionalidades esenciales:
core/state_manager.py - Gestión del estado del sistema
core/portfolio_manager.py - Tracking y gestión de portfolio
core/technical_indicators.py - Cálculo de indicadores
core/feature_engineering.py - Preparación de features para L2
🔹 Logging centralizado:
Todos los módulos usan un único logger centralizado en core/logging.py, que combina:
  - Logging estándar de Python.
  - Loguru para formatos enriquecidos y colores en consola.
  - Trazabilidad de ciclo, símbolo y nivel.

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
│   ├── __init__.py
│   ├── state_manager.py         # Gestión del estado global
│   ├── portfolio_manager.py     # Gestión de portfolio y CSV
│   ├── technical_indicators.py  # Cálculo de indicadores técnicos
│   ├── feature_engineering.py   # Preparación de features para L2          
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

## 🔁 TABLA DE TIEMPOS/FRECUENCIAS
| Nivel | Frecuencia              |
| ----- | ----------------------- |
| L3    | 10 min (periódico)      |
| L2    | 10 s                    |
| L1    | subsegundos / inmediato |


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
