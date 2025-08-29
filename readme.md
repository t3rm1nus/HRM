HRM — Hierarchical Reasoning Model para Trading Algorítmico
Estado: Activo · Lenguaje: Python · Dominio: Cripto/Trading · Arquitectura: Multi-nivel (L4→L1)Regla de oro: Si existe conflicto entre este README y los README de módulos, prevalece el README del módulo.

🧭 TL;DR
HRM es un framework de razonamiento jerárquico para trading algorítmico, diseñado para operar con múltiples activos (BTC, ETH). Divide el problema en 4 niveles, desde la reflexión global (L4) hasta la ejecución determinista con IA avanzada (L1).El objetivo es decidir qué, cuándo y cuánto operar, limitando riesgo mediante reglas hard-coded y modelos IA (Logistic Regression, Random Forest, LightGBM) en L1, con gestión de correlación BTC-ETH y métricas granulares por activo. Incluye bus de mensajes, telemetría, persistencia histórica, dataset multitimeframe y tests robustos.

1️⃣ Objetivo del proyecto

Tomar decisiones de trading razonadas y trazables para múltiples activos (BTC, ETH) mediante una jerarquía de agentes.
Aprender qué señales mantienen performance bajo distintos regímenes de mercado, cómo combinar niveles, y optimizar ejecución con modelos IA (Logistic Regression, Random Forest, LightGBM) en L1.
Minimizar riesgos con análisis multinivel, capa dura de seguridad, y gestión de correlación BTC-ETH en ejecución.
Crear un framework reutilizable para diferentes estrategias y universos de activos, con soporte nativo para múltiples activos líquidos.

Qué queremos aprender a nivel del sistema:

Si el razonamiento multietapa mejora la estabilidad vs. un agente monolítico.
Qué señales funcionan en cada régimen y cómo combinarlas en L2/L3.
Cómo distribuir peso/capital entre modelos/estrategias y detectar concept drift en L4.


2️⃣ Beneficios esperados

Mayor precisión en predicciones mediante composición de señales multiasset y modelos IA (Logistic Regression, Random Forest, LightGBM).
Reducción de riesgos vía diversificación temporal, capa dura de seguridad en L1, y gestión de correlación BTC-ETH.
Adaptabilidad automática a distintos regímenes de mercado para múltiples activos.
Capacidad de razonamiento complejo multi-variable con métricas granulares por activo.


3️⃣ Flujo general (visión de tiempos)

Nivel 4: Meta-Razonamiento (horas/días)
Nivel 3: Análisis Estratégico (horas)
Nivel 2: Táctica de Ejecución (minutos)
Nivel 1: Ejecución + Gestión de Riesgo (segundos)


4️⃣ Jerarquía del sistema (HRM extendido)
Nivel 4 — Meta-Razonamiento (horas/días)
Rol: Reflexión y adaptación del sistema completo.Funciones: Evaluación de desempeño (Sharpe, drawdown), concept drift, selección de modelos/estrategias, asignación de capital y ajustes globales.Ejemplo: Si mean reversion pierde eficacia, reduce su peso y reasigna capital a trend-following.
Nivel 3 — Análisis Estratégico (horas)
Rol: Planificación de alto nivel.Funciones: Clasificación de régimen (tendencia/rango/volatilidad), selección de sub-estrategias, priorización de activos (BTC, ETH, alts líquidas), metas intradía (exposición, riesgo máximo).
Nivel 2 — Táctica de Ejecución (minutos)
Rol: Convertir las decisiones estratégicas en operaciones concretas.Funciones: Composición de señales, position sizing (vol-targeting, Kelly fraccionado), stops/targets dinámicos, ajustes por liquidez/volatilidad.
Nivel 1 — Ejecución y Riesgo (segundos)
Rol: Implementación determinista para múltiples activos (BTC, ETH) con capa dura de seguridad y modelos IA avanzados.Funciones: 

Validación de límites de riesgo por símbolo (stop-loss, exposición, correlación BTC-ETH aplicada desde L2/L3).
Filtrado de señales con modelos IA (modelo1_lr.pkl, modelo2_rf.pkl, modelo3_lgbm.pkl) para confirmar tendencias.
Ejecución optimizada con lógica determinista (fraccionamiento, timing, reducción de slippage).
Envío de órdenes con timeouts/retries.
Reportes detallados por activo (BTC/USDT, ETH/USDT).
Métricas granulares (latencia, slippage, exposición, tasas de éxito por símbolo).


5️⃣ Arquitectura (ASCII)
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
│ Hard-coded Safety Layer + Order Manager (determinista)       │
│ AI Models (LogReg, RF, LightGBM) + Multiasset Execution      │
│ Executor determinista → Exchange                             │
└──────────────────────────────────────────────────────────────┘


6️⃣ Conexión entre niveles



Flujo
Descripción



L4 → L3
Ajuste de capital y parámetros globales


L3 → L2
Selección de sub-estrategias y universo de activos (BTC, ETH)


L2 → L1
Señales concretas (qty, stop, target) por símbolo


L1 → Exchange
Envío/gestión de órdenes en tiempo real para BTC/USDT y ETH/USDT



7️⃣ Estructura de carpetas
HRM/
│── docs/                      # documentación
│
│── storage/                   # módulo de persistencia
│   ├── csv_writer.py
│   ├── sqlite_writer.py
│   └── __init__.py
│
├── core/                      # utilidades globales
│   ├── logging.py
│   ├── scheduler.py
│   └── utils.py
│
├── comms/                     # comunicaciones y eventos
│   ├── config/                # configs YAML/JSON
│   ├── message_bus.py
│   ├── schemas.py
│   └── adapters/              # conectores externos (Kafka, Redis, etc.)
│
├── l4_meta/                   # meta-razonamiento (horas/días)
│   ├── drift_detector.py
│   ├── strategy_selector.py
│   ├── portfolio_allocator.py
│   └── __init__.py
│
├── l3_strategy/               # nivel estratégico actual (intradía)
│   ├── regime_classifier.py
│   ├── universe_filter.py
│   ├── exposure_manager.py
│   └── __init__.py
│
├── l3_strategic/              # FUTURA arquitectura estratégica (multi-nivel)
│   ├── __init__.py
│   ├── README.md  
│   ├── models.py                    # Estructuras de datos L3
│   ├── config.py                    # Configuración estratégica simplificada
│   ├── strategic_processor.py       # Procesador principal L3
│   ├── bus_integration.py           # Comunicación L4 ↔ L3 ↔ L2
│   ├── performance_tracker.py       # Tracking performance estratégico
│   ├── metrics.py                   # Métricas L3
│   ├── procesar_l3.py               # Entry-point local para pruebas
│   ├── ai_model_loader.py           # Cargador de los 3 modelos IA
│   └── ai_models/                   # Solo 3 modelos ligeros
│       ├── __init__.py
│       ├── unified_decision_model.py # Modelo 1: Decisiones estratégicas unificadas
│       ├── regime_detector.py        # Modelo 2: Detección de régimen de mercado  
│       └── risk_assessor.py          # Modelo 3: Evaluación de riesgo integrada
│
├── l2_tactic/                 # nivel táctico (señales, sizing)
│   ├── signal_generator.py
│   ├── position_sizer.py
│   ├── risk_controls.py
│   └── __init__.py
│
├── l1_operational/            # nivel operacional (OMS/EMS)
│   ├── models.py
│   ├── config.py
│   ├── bus_adapter.py
│   ├── order_manager.py
│   ├── risk_guard.py
│   ├── executor.py
│   ├── data_feed.py
│   ├── binance_client.py
│   ├── ai_models/             # modelos IA entrenados (L1)
│   │   ├── modelo1_lr.pkl
│   │   ├── modelo2_rf.pkl
│   │   └── modelo3_lgbm.pkl
│   ├── test_clean_l1_multiasset.py
│   ├── README.md
│   └── requirements.txt
│
├── models/                    # modelos IA externos centralizados
│   ├── L1/
│   │   ├── modelo1_lr.pkl
│   │   ├── modelo2_rf.pkl
│   │   └── modelo3_lgbm.pkl
│   ├── L2/
│   │   ├── _stable_baselines3_version
│   │   ├── data/
│   │   ├── policy.optimizer.pth
│   │   ├── policy.pth
│   │   ├── pytorch_variables.pth
│   │   └── system_info
│   ├── L3/
│   └── L4/
│
├── data/                      # ingestión y almacenamiento
│   ├── connectors/
│   ├── loaders.py
│   ├── storage/
│   └── __init__.py
│
├── risk/                      # librería transversal de riesgo
│   ├── limits.py
│   ├── var_es.py
│   ├── drawdown.py
│   └── __init__.py
│
├── monitoring/                # métricas y reporting
│   ├── dashboards/
│   ├── alerts.py
│   ├── telemetry.py
│   └── __init__.py
│
├── tests/                     # unit & integration tests
│   └── backtester.py
└── main.py                    # orquestador central


Nota: Esta estructura resume el proyecto real y es suficiente para navegar y extender el código.

8️⃣ Flujo de mensajes y state global
Cada ciclo del sistema trabaja sobre un único state (diccionario) y cada nivel actualiza su sección. Esto garantiza trazabilidad y facilita debugging/backtesting.
state = {
    "mercado": {...},       # precios actuales por símbolo (BTC, ETH)
    "estrategia": "...",    # estrategia activa (agresiva/defensiva)
    "portfolio": {...},     # asignación de capital (unidades por activo)
    "universo": [...],      # activos disponibles (BTC/USDT, ETH/USDT)
    "exposicion": {...},    # % exposición por activo (BTC, ETH)
    "senales": {...},       # señales tácticas por símbolo
    "ordenes": [...],       # órdenes ejecutadas en L1 por símbolo
    "riesgo": {...},        # chequeo de riesgo (incluye correlación BTC-ETH)
    "deriva": False,        # drift detection
    "ciclo_id": 1           # número de ciclo
}

Flujo L1 (ejecución determinista):L2/L3 (Señales BTC/ETH) → Bus Adapter → Order Manager → Hard-coded Safety → AI Models (LogReg, RF, LightGBM) → Risk Rules → Executor → Exchange → Execution Report → Bus Adapter → L2/L3

9️⃣ L1_operational — “limpio y determinista”
L1 SOLO ejecuta órdenes seguras para múltiples activos (BTC, ETH). No decide estrategia ni táctica.
Lo que L1 no hace

❌ No modifica cantidades ni precios de señales estratégicas
❌ No decide timing fuera de optimización determinista
❌ No actualiza portfolio completo (responsabilidad de L2/L3)
❌ No recolecta ni procesa datos de mercado (responsabilidad de L2/L3)

Lo que L1 sí hace

✅ Valida límites de riesgo por símbolo (stop-loss, exposición, correlación BTC-ETH aplicada desde L2/L3)
✅ Filtra señales con modelos IA (modelo1_lr.pkl, modelo2_rf.pkl, modelo3_lgbm.pkl) para confirmar tendencias
✅ Ejecuta órdenes pre-validadas en el exchange con optimización de slippage (simulada en modo PAPER)
✅ Genera reportes detallados por activo (BTC/USDT, ETH/USDT)
✅ Mantiene trazabilidad completa con métricas granulares (latencia, slippage, tasas de éxito)

Verificación de limpieza:
python l1_operational/test_clean_l1_multiasset.py


🔌 Mensajería, logging y telemetría

Mensajería: comms/ define esquemas y el bus (JSON/Protobuf; colas asyncio; adapters Kafka/Redis si se desea).
Logging estructurado: JSON (p. ej. python-json-logger) con etiquetas por símbolo ([BTC], [ETH]).
Telemetría (monitoring/telemetry.py):  
incr(name) → contadores (órdenes por símbolo)
gauge(name, value) → métricas instantáneas (exposición, correlación)
timing(name, start) → latencias por ejecución



Dashboard en consola: Ejemplo de métricas consolidadas generadas por L1, visualizadas con rich por ciclo, mostrando métricas por activo (BTC/USDT, ETH/USDT). La visualización es manejada por componentes externos.

🗃️ Persistencia de histórico
Cada ciclo se guarda en:

CSV: data/historico.csv (todas las variables del state, incluyendo métricas por símbolo)
SQLite: data/historico.db (tabla ciclos con los mismos datos)

Esto permite exportar a Pandas/Excel, reproducir backtests y consultar con SQL.

🧪 Dataset & features (BTC/USDT, ETH/USDT)
Generador de features en data/loaders.py (limpio y autocontenido).Soporta: BTC/USDT y ETH/USDT (extensible a otros activos líquidos) con índice datetime y columna close.
Features incluidas:

Precio: delta_close, ema_10/20, sma_10/20
Volumen: vol_rel vs. media N (20 por defecto)
Momentum: rsi, macd, macd_signal, macd_hist
Multi-timeframe: 1m + 5m (sufijos _5m, reindex 1m)
Cruzadas: ETH/BTC ratio, correlación rolling, divergencias

Uso básico:
import pandas as pd
from data.loaders import prepare_features

# 1) Cargar velas 1m con índice datetime y columna 'close'
df_btc_1m = pd.read_csv("data/btc_1m.csv", parse_dates=["timestamp"], index_col="timestamp")
df_eth_1m = pd.read_csv("data/eth_1m.csv", parse_dates=["timestamp"], index_col="timestamp")

# 2) Generar features 1m+5m y split temporal (80/20 por defecto)
train_btc, test_btc = prepare_features(df_btc_1m, test_size=0.2, symbol="BTC")
train_eth, test_eth = prepare_features(df_eth_1m, test_size=0.2, symbol="ETH")

# 3) Guardar datasets
train_btc.to_csv("data/btc_features_train.csv")
test_btc.to_csv("data/btc_features_test.csv")
train_eth.to_csv("data/eth_features_train.csv")
test_eth.to LGBTQ

System: **to_csv("data/eth_features_test.csv")

Nota: Si ya tienes velas 5m, puedes pasarlas como df_5m y evitar resampleo. Si tu CSV trae BTC_close o ETH_close, normalize_columns lo mapea a close automáticamente.

⚙️ Puesta en marcha
Requisitos

Python 3.10+ recomendado
Cuenta de exchange (modo sandbox si es posible) si vas a ejecutar L1
Credenciales/API Keys (usa variables de entorno o .env)
pip, venv o uv (opcional)

Instalación rápida
# 1) Clonar
git clone https://github.com/t3rm1nus/HRM.git
cd HRM

# 2) Entorno
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)

# 3) Dependencias (L1)
pip install -r l1_operational/requirements.txt

# 4) (Opcional) Dependencias extra según conectores/adapters
# pip install -r requirements.txt  # si existe en la raíz / módulos

Ejecución (modo demo)
python main.py

Configura en core/config/ y variables de entorno los parámetros de conexión y límites de riesgo por símbolo.

✅ Buenas prácticas de riesgo (resumen)

Hard limits en L1: 
Stop-loss obligatorio
Límites de capital por trade (BTC: 0.05 max, ETH: 1.0 max)
Exposición máxima (BTC: 20%, ETH: 15%)
Chequeos de liquidez/saldo
Drawdown y correlación BTC-ETH (calculada en L2/L3, aplicada en L1)


Determinismo en ejecución: Una oportunidad de orden por señal; si no cumple reglas → rechazo y reporte.
Separación de responsabilidades: Señal (L2/L3) ≠ ejecución (L1).
Backtesting: Con histórico persistido y state reproducible.


🧩 Tests e integración

Pruebas de limpieza L1: l1_operational/test_clean_l1_multiasset.py
Backtester de extremo a extremo: tests/backtester.py
Métricas/alertas: monitoring/ (métricas por símbolo y correlación)


📈 Roadmap (alto nivel)

Meta-aprendizaje para selección dinámica de estrategias (L4)
Mejores clasificadores de régimen (L3)
Ensamble multi-señal robusto (L2)
Integración multi-exchange/DEX y simulador de slippage (L1)
Dashboards enriquecidos (web) y alertas proactivas con métricas por activo


👥 Autoría y licencia

Autoría: Equipo de desarrollo HRM
Versión: 1.0
Última actualización: 2025
Licencia: Ver archivo LICENSE si aplica


Envío a otras IA: Este README está diseñado para ser autosuficiente. Describe jerarquía, arquitectura, flujos, estructura de código, dataset, telemetría, persistencia y puesta en marcha para que un agente externo pueda comprender y operar el proyecto sin consultar otros documentos.