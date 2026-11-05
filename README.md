# HRM — Sistema jerárquico de trading de criptomonedas

> **Estado de esta documentación:** revisión estática del ZIP `HRM-master.zip` recibido el 2 de noviembre de 2026.  
> **Importante:** este README describe lo que aparece en el código y diferencia las funciones documentadas de las que están realmente verificadas. No constituye una auditoría financiera ni una garantía de funcionamiento.

## 1. ¿Qué es HRM?

HRM es un proyecto Python orientado al trading algorítmico de criptomonedas, principalmente BTC/USDT y ETH/USDT. Su diseño pretende combinar:

- Datos de mercado en tiempo real.
- Análisis técnico y generación de señales.
- Modelos de aprendizaje automático y aprendizaje por refuerzo.
- Clasificación del régimen de mercado y análisis de sentimiento.
- Gestión de cartera, dimensionamiento de posiciones y controles de riesgo.
- Ejecución de órdenes mediante Binance o un exchange simulado.
- Registro de operaciones, métricas y un sistema de autoaprendizaje/reentrenamiento.

La arquitectura se organiza conceptualmente en tres niveles:

| Nivel | Nombre | Responsabilidad prevista |
|---|---|---|
| L1 | Operacional | Convierte señales en intenciones y órdenes, valida y gestiona su ejecución. |
| L2 | Táctico | Analiza el mercado a corto plazo y genera señales tácticas. |
| L3 | Estratégico | Evalúa el contexto general: régimen, sentimiento, volatilidad y asignación de cartera. |

El proyecto también contiene servicios transversales: configuración, estado, comunicación entre módulos, logging, persistencia, backtesting y autoaprendizaje.

**No debe confundirse “existe código para una función” con “la función está integrada y probada de extremo a extremo”.** En el estado recibido hay piezas incompletas, artefactos ausentes y errores de sintaxis.

## 2. Arquitectura y flujo de funcionamiento

### 2.1. Flujo lógico esperado

El flujo de alto nivel que persigue el sistema es:

```text
Fuentes de mercado
       |
       v
MarketDataManager / DataFeed / RealTimeDataLoader
       |
       v
L3 — Contexto estratégico
     - Régimen de mercado
     - Sentimiento
     - Volatilidad
     - Optimización de cartera
       |
       v
L2 — Análisis táctico
     - Indicadores y características
     - Modelos / ensemble
     - Señal y confianza
     - Controles de riesgo tácticos
       |
       v
L1 — Capa operacional
     - Construcción de intención
     - Validación de orden
     - Dimensionamiento
     - Risk Guard
       |
       v
ExecutionGateway / OrderManager
       |
       +------> Exchange simulado (paper)
       |
       +------> Binance (solo si se configura y habilita expresamente)
       |
       v
PortfolioManager / métricas / logs / almacenamiento
       |
       v
Auto-learning y evaluación de modelos
```

El diseño incluye un `MessageBus` para intercambiar mensajes entre componentes y un `StateCoordinator`/gestor de estado para compartir el estado del sistema.

### 2.2. L1 — Capa operacional

Carpeta: `l1_operational/`

Componentes principales:

- `binance_client.py`: cliente basado en `ccxt.async_support`; contempla modo paper, testnet y endpoints públicos de Binance.
- `data_feed.py`: descarga datos OHLCV y gestiona el acceso al exchange.
- `realtime_loader.py`: carga de datos en tiempo real.
- `order_manager.py`: coordina órdenes.
- `order_intent_builder.py`: construye intenciones de orden a partir de señales.
- `order_validators.py`: validaciones previas.
- `risk_guard.py`: controles de riesgo antes de ejecutar.
- `executor.py`, `order_executors.py`: piezas relacionadas con la ejecución.
- `simulated_exchange_client.py`: exchange simulado para paper trading.
- `portfolio.py`, `position_manager.py`: gestión de posiciones y cartera.
- `ai_pipeline.py`, `trend_ai.py`, `signal_processor.py`: análisis y procesamiento auxiliar.

La intención arquitectónica es que las señales no lleguen directamente al exchange: antes deben convertirse en órdenes, superar validaciones y pasar los controles de riesgo.

### 2.3. L2 — Capa táctica

Carpeta: `l2_tactic/`

Se ocupa de convertir los datos y características del mercado en señales tácticas.

Incluye:

- `tactical_signal_processor.py`: procesador táctico principal.
- `signal_generators.py`, `signal_generator_refactored.py`: generadores de señales.
- `generators/`: análisis técnico y reversión a la media.
- `indicators/` y `technical/`: indicadores, patrones, análisis multitemporal y soportes/resistencias.
- `ensemble/`: combinación y votación de modelos.
- `feature_extractors.py` y `observation_builders.py`: preparación de observaciones/características.
- `finrl_processor.py`, `finrl_wrapper.py` y módulos `finrl_*`: integración con componentes de FinRL y Stable-Baselines3.
- `model_loaders.py`: carga de modelos serializados.
- `risk_controls/`, `risk_overlay.py`, `position_sizer.py`: controles tácticos y dimensionamiento.

La carpeta contiene una mezcla de implementaciones, wrappers, módulos alternativos y pruebas. La existencia de esos ficheros no acredita que todos estén conectados al camino de ejecución de `main.py`.

### 2.4. L3 — Capa estratégica

Carpeta: `l3_strategy/`

Pretende aportar una visión de contexto más amplia:

- `regime_classifier.py` y `regime_features.py`: clasificación del régimen del mercado.
- `sentiment_inference.py`: análisis de sentimiento con Transformers/BERT y fuentes externas.
- `volatility_inference_pipeline.py`: componentes de previsión de volatilidad.
- `hrm_bl.py`: componentes relacionados con Black-Litterman.
- `exposure_manager.py`, `risk_manager.py`, `universe_filter.py`: exposición, riesgo y selección del universo.
- `l3_aggregator.py` y `l3_inference_pipeline.py`: agregación e inferencia.
- `data_provider.py`: proveedor de datos con fuentes locales y opciones online.

La configuración de L3 contempla regímenes alcistas, bajistas, laterales, volátiles y de transición. También se describen modelos especializados por régimen.

**Advertencia:** `runtime_loop.py` importa `l3_strategy.decision_maker`, pero ese módulo no aparece en el ZIP. Es una de las señales de que el runtime necesita reparación antes de poder considerarse ejecutable.

### 2.5. Servicios comunes

| Carpeta / archivo | Función |
|---|---|
| `core/` | Estado, cartera, validación, indicadores, métricas, configuración, errores y utilidades. |
| `system/` | Bootstrap, coordinación de estado, pipeline, datos de mercado, limpieza y recuperación. |
| `comms/` | Configuración común, bus de mensajes, esquemas y validación de mensajes. |
| `storage/` | Registro de operaciones paper y escritura de datos. |
| `sentiment/` | Gestión auxiliar de textos de sentimiento. |
| `backtesting/` | Descarga de datos, ejecución de pruebas y generación/análisis de informes. |
| `ml_training/` | Scripts de entrenamiento offline para modelos L1, L2 y L3. |
| `hacienda/` | Seguimiento fiscal y cálculo FIFO; no forma parte del ciclo activo de trading según la documentación del propio proyecto. |
| `scripts/` | Arranque, comprobaciones, diagnóstico y utilidades. |
| `tests/` | Pruebas unitarias, de integración y scripts de comprobación. |
| `archived/` | Configuraciones y estados antiguos archivados. |

## 3. ¿Qué IAs y modelos utiliza?

Es importante separar tres cosas: modelos ML entrenados localmente, modelos de aprendizaje por refuerzo y servicios de IA externos. En el código inspeccionado no son equivalentes.

### 3.1. Modelos L1

La documentación interna menciona estos modelos:

- Logistic Regression.
- Random Forest.
- LightGBM.
- Combinación/ensemble de modelos.

Los artefactos que se citan en la documentación son `modelo1_lr.pkl`, `modelo2_rf.pkl` y `modelo3_lgbm.pkl`, entre otros.

**Estado del ZIP:** no contiene la carpeta `models/` ni esos ficheros serializados. Por tanto, no se puede verificar la carga ni la inferencia de esos modelos con el paquete recibido.

### 3.2. Modelos L2 y nombres de proveedores

La documentación `docs/LISTADO_9_IAS_Y_MODELOS.md` enumera:

- Claude — Anthropic.
- DeepSeek.
- GPT — OpenAI.
- Grok — xAI.
- Kimi — Moonshot.
- Gemini — Google.
- DeepSeek en una segunda variante.
- Modelos PPO de Stable-Baselines3/FinRL.

El código de `l2_tactic/model_loaders.py` contiene funciones con nombres como `load_claude_model`, `load_deepseek_model`, `load_kimi_model` y `load_gpt_model`. Esas funciones intentan cargar archivos ZIP mediante Stable-Baselines3/PPO y PyTorch.

Esto **no demuestra que el sistema esté llamando a Claude, GPT, Gemini, Grok, Kimi o DeepSeek como servicios conversacionales remotos**. En el código inspeccionado no aparecen integraciones directas identificables con los SDK habituales de OpenAI, Anthropic o Google Generative AI. Los nombres parecen identificar artefactos/modelos entrenados o empaquetados para el sistema, no necesariamente los modelos comerciales originales.

Además:

- La carpeta `models/L2/` no está incluida en el ZIP.
- No aparecen `claude.zip`, `deepseek.zip`, `gemini.zip`, `gpt.zip`, `grok.zip` ni `kimi.zip`.
- `l2_tactic/config.py` contiene rutas absolutas de Windows del tipo `C:/proyectos/HRM/models/L2/...`, que no son portables.
- `model_loaders.py` puede cargar modelos PPO, pero no se puede probar sin los artefactos y dependencias correspondientes.

**Conclusión:** el repositorio contiene una capa de carga y referencias a modelos con esos nombres, pero el ZIP no permite confirmar que los siete modelos estén disponibles ni operativos.

### 3.3. Modelos L3

La documentación y el código hacen referencia a:

- BERT para sentimiento, mediante Hugging Face Transformers y PyTorch.
- GARCH para volatilidad.
- LSTM para volatilidad.
- Modelos de clasificación de régimen, incluidos ensembles.
- Optimización de cartera Black-Litterman.

Los artefactos esperados (por ejemplo, pesos `.pkl`, `.h5`, `.safetensors` y archivos de tokenizer) tampoco están incluidos en el ZIP.

### 3.4. Las “9 IAs” del autoaprendizaje

El documento `docs/LISTADO_9_IAS_Y_MODELOS.md` denomina “9 IAs” a nueve mecanismos de protección del reentrenamiento. No son nueve modelos generativos ni nueve proveedores de IA.

| Nº | Mecanismo | Objetivo descrito |
|---:|---|---|
| 1 | `AntiOverfitValidator` | Validación en ventanas temporales para limitar el sobreajuste. |
| 2 | `AdaptiveRegularizer` | Ajuste adaptativo de regularización. |
| 3 | `DiverseEnsembleBuilder` | Promover diversidad en el ensemble. |
| 4 | `ConceptDriftDetector` | Detectar cambios en la distribución de los datos. |
| 5 | `SmartEarlyStopper` | Detener entrenamientos cuando dejan de mejorar. |
| 6 | `TimeBasedTrigger` | Activar revisión/reentrenamiento por tiempo. |
| 7 | `PerformanceBasedTrigger` | Activarlo por deterioro de métricas. |
| 8 | `RegimeChangeTrigger` | Activarlo ante cambios de régimen. |
| 9 | `DataVolumeTrigger` | Activarlo al acumular suficientes operaciones/datos. |

En el documento se citan parámetros como cinco ventanas de validación, score mínimo de 0,55, máximo de diez modelos y un umbral de similitud de 0,85. Son valores documentados en el proyecto; no deben interpretarse como resultados de rendimiento demostrados.

## 4. Integraciones y servicios externos

### Binance

La integración más clara del proyecto es Binance, a través de `ccxt.async_support` y llamadas a endpoints HTTP públicos.

Se contemplan:

- Datos públicos de mercado.
- Velas OHLCV y precios.
- Modo testnet.
- Modo paper con cartera simulada.
- Cliente de exchange para operaciones.

El código lee estas variables de entorno:

- `BINANCE_API_KEY`
- `BINANCE_API_SECRET`
- `BINANCE_MODE`
- `PAPER_MODE`

En paper trading, el diseño pretende utilizar datos reales de mercado y simular las operaciones localmente. La disponibilidad de datos públicos no requiere claves privadas; las operaciones autenticadas sí.

### Reddit

`l3_strategy/sentiment_inference.py` utiliza `asyncpraw` y contempla:

- `REDDIT_CLIENT_ID`
- `REDDIT_CLIENT_SECRET`
- `REDDIT_USER_AGENT`

Si no están configuradas las credenciales, el código contempla datos sintéticos de respaldo. Eso permite que ciertas rutas de sentimiento no dependan siempre de Reddit, pero **los datos sintéticos no son sentimiento real del mercado**.

### NewsAPI

El mismo módulo contempla NewsAPI y la variable:

- `NEWS_API_KEY`

Si no hay clave o falla la consulta, el código contempla caché y datos sintéticos. De nuevo, un resultado sintético no equivale a noticias reales.

### Yahoo Finance, FRED y fuentes locales

`l3_strategy/data_provider.py` describe:

- CSV locales.
- Yahoo Finance mediante `yfinance`, si se habilitan fuentes online.
- FRED mediante `fredapi` y `FRED_API_KEY`, de forma opcional.
- Datos on-chain locales o puntos de integración.
- Datos sintéticos de respaldo en determinados casos.

La variable `USE_ONLINE_SOURCES` controla la activación de fuentes online en ese proveedor.

### FinRL, Stable-Baselines3, PyTorch y Hugging Face

El proyecto contiene integración con:

- FinRL.
- Stable-Baselines3.
- PyTorch.
- Gymnasium.
- Transformers/Hugging Face.

Son bibliotecas/frameworks de ML y aprendizaje por refuerzo, no proveedores de señales financieras garantizadas.

### Resumen de integraciones

| Servicio / tecnología | Uso previsto | Qué se puede afirmar del ZIP |
|---|---|---|
| Binance / CCXT | Mercado y ejecución | Código presente; operación real no verificada. |
| Reddit / Async PRAW | Textos para sentimiento | Código presente; requiere credenciales para datos reales. |
| NewsAPI | Noticias | Código presente; requiere `NEWS_API_KEY`. |
| Yahoo Finance | Datos históricos/auxiliares | Integración opcional descrita. |
| FRED | Datos macroeconómicos | Integración opcional descrita. |
| FinRL / SB3 / PyTorch | Modelos RL | Código presente; artefactos ausentes. |
| Hugging Face Transformers | BERT/sentimiento | Código presente; pesos/modelos ausentes. |
| OpenAI / Anthropic / Google / xAI / Moonshot APIs | LLM comerciales | No se ha podido verificar una integración API directa en el código inspeccionado. |

## 5. Instalación: preparación del entorno

### 5.1. Requisitos generales

El proyecto es Python y contiene código asíncrono. La documentación no fija una versión única de Python ni proporciona un manifiesto completo de dependencias.

Hay bytecode `.pyc` de Python 3.10 en el ZIP, lo que sugiere que se utilizó Python 3.10 en algún entorno, pero no demuestra que sea la única versión compatible.

Recomendación para empezar:

- Windows 10/11 o Linux.
- Python 3.10 en un entorno virtual independiente.
- Git (opcional, si se clona el repositorio).
- Acceso a Internet para las fuentes externas.
- No configurar claves privadas durante la primera prueba.

### 5.2. Crear y activar un entorno virtual

Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

Linux/macOS:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### 5.3. Instalar dependencias

**El repositorio no trae un `requirements.txt` raíz.** Solo aparecen manifiestos parciales:

- `l1_operational/requirements.txt`
- `l2_tactic/requeriments.txt` (el nombre está escrito con un error: `requeriments`).

El primero enumera `ccxt`, `loguru`, `pydantic` y `dataclasses-json`. El segundo incluye librerías científicas y de ML, como NumPy, pandas, SciPy, scikit-learn, joblib, Stable-Baselines3 y PyTorch, además de dependencias opcionales comentadas.

Para instalar esos manifiestos:

```bash
pip install -r l1_operational/requirements.txt
pip install -r l2_tactic/requeriments.txt
```

No debe asumirse que esto instala todo lo necesario. El código también importa, entre otras, bibliotecas como `python-dotenv`, `colorama`, `aiohttp`, `requests`, `asyncpraw`, `transformers`, `torch`, `gymnasium` y, en rutas concretas, TensorFlow, `yfinance` o `fredapi`.

La instalación de PyTorch/TensorFlow depende del sistema operativo y del hardware. Conviene instalarlos siguiendo las instrucciones oficiales de cada proyecto y no forzar una combinación de versiones al azar.

**No se proporciona un comando de instalación que garantice un entorno completo y reproducible.** Antes de desplegarlo habría que consolidar dependencias y fijar versiones.

## 6. Configuración

### 6.1. Variables de entorno

Crea un archivo `.env` en la raíz solo si vas a utilizar las integraciones que lo necesitan. No subas ese archivo al repositorio.

Plantilla orientativa, sin credenciales:

```dotenv
# Binance: dejar vacío durante las primeras pruebas
BINANCE_API_KEY=
BINANCE_API_SECRET=

# Seguridad: empezar siempre en paper
PAPER_MODE=true
BINANCE_MODE=paper

# Sentimiento: opcional
NEWS_API_KEY=
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=
REDDIT_USER_AGENT=

# Datos macro / proveedor L3 opcionales
USE_ONLINE_SOURCES=false
FRED_API_KEY=
```

La aplicación carga `python-dotenv` en varios módulos. Las variables exactas que se utilizan dependen del componente y de la ruta de ejecución.

No pongas claves reales en el código, en archivos JSON versionados, en capturas de pantalla ni en logs.

### 6.2. Configuraciones YAML

Hay dos configuraciones principales:

- `configs/config_backtest.yaml`
- `configs/config_live.yaml`

La primera está orientada a backtesting y desactiva la sincronización con el exchange. La segunda contiene parámetros etiquetados como producción/live y `binance_testnet: false`.

**No uses `config_live.yaml` para experimentar.** Sus valores no convierten el sistema en seguro ni demuestran que la ruta live esté correctamente probada.

### 6.3. Estado inicial y capital simulado

`config_loader.py` define valores por defecto para `initial_state.json`, entre ellos:

- Capital simulado: 3000 USDT.
- BTC inicial: 0.
- ETH inicial: 0.
- Modo: paper.
- Auto-learning: `fix`.

El archivo puede generarse al cargar la configuración si no existe. Revisa siempre el capital y los balances simulados antes de iniciar el sistema.

Hay referencias a otros valores de capital en distintos módulos y documentos. La configuración de capital no está completamente unificada en el repositorio recibido.

## 7. Cómo intentar arrancarlo

### 7.1. Arranque manual

Desde la raíz del proyecto, con el entorno virtual activado y dependencias instaladas:

```bash
python main.py
```

El archivo principal es `main.py`. Su función `main()` configura manejadores de apagado, intenta limpiar el estado, inicializa el coordinador, el gestor de datos, la cartera, el gestor de órdenes y los procesadores.

Sin embargo, **este comando es una referencia al punto de entrada, no una garantía de que el proyecto arranque**. La revisión estática ha encontrado errores que deben corregirse primero (ver sección 10).

### 7.2. Script PowerShell incluido

Existe:

```powershell
.\scripts\start.ps1
```

Este script:

1. Escribe un `initial_state.json` con 3000 USDT simulados.
2. Elimina archivos de estado y operaciones anteriores en varias rutas.
3. Crea `.venv` si no existe.
4. Instala un pequeño conjunto de dependencias.
5. Ejecuta `python main.py`.

**Precaución importante:** el script borra datos persistentes anteriores y reemplaza la configuración inicial. No lo ejecutes sobre una instalación que contenga operaciones, balances o estados que quieras conservar. Además, su instalación de dependencias es parcial.

Para una primera revisión, es preferible preparar el entorno manualmente y hacer una copia de seguridad de cualquier archivo de estado.

## 8. Paper trading, testnet y trading real

Estos modos no son lo mismo:

- **Paper trading:** el sistema simula cartera y ejecuciones; puede consumir precios reales, pero no debería enviar órdenes reales.
- **Testnet:** entorno de pruebas del exchange, separado de la cuenta real.
- **Live:** órdenes reales con fondos reales.

El código de `main.py` comprueba `BINANCE_MODE` y `PAPER_MODE`. Si se solicita `BINANCE_MODE=live` sin establecer explícitamente `PAPER_MODE=false`, fuerza el modo paper. Si ambos valores permiten live, muestra una advertencia y espera diez segundos antes de continuar.

Esto es una protección parcial, no una garantía de seguridad. Hay varias configuraciones y módulos que contienen valores distintos, y el ZIP no permite certificar todas las rutas de ejecución.

**No habilites live ni introduzcas claves con permisos de retirada.** Antes de cualquier operación real harían falta pruebas completas, revisión de órdenes, límites, stop-loss, persistencia, recuperación y validación independiente.

## 9. Backtesting, entrenamiento y fiscalidad

### Backtesting

La carpeta `backtesting/` contiene herramientas de descarga, preparación de datos, ejecución de pruebas y análisis de rendimiento. También hay configuraciones YAML para backtest.

No se ha podido certificar un resultado de backtest reproducible a partir del ZIP: no se han ejecutado las pruebas y faltan dependencias y potenciales artefactos/datasets.

### Entrenamiento

`ml_training/` contiene scripts para entrenar modelos L1, L2 y L3. Son herramientas de entrenamiento offline, no necesariamente procesos que se ejecuten automáticamente al iniciar el bot.

### Autoaprendizaje

`auto_learning_system.py` define componentes de evaluación, regularización, diversidad, detección de drift y triggers de reentrenamiento. `integration_auto_learning.py` intenta conectar el sistema con componentes del runtime.

La presencia de estas clases no demuestra que el reentrenamiento esté completándose, que los nuevos modelos se desplieguen correctamente o que mejoren el rendimiento. Para certificarlo hacen falta datos, artefactos, ejecución controlada y pruebas.

### Hacienda

`hacienda/` incluye un módulo de seguimiento fiscal y cálculo FIFO. La documentación del proyecto indica que es auxiliar y no está integrado en el ciclo de trading activo.

## 10. Estado real del repositorio recibido

Esta sección es especialmente importante. Resume lo que se ha podido comprobar directamente en el ZIP.

### 10.1. Hallazgos bloqueantes o relevantes

| Hallazgo | Evidencia | Impacto |
|---|---|---|
| No hay README raíz | No existe `README.md` en la raíz del ZIP. | Falta documentación de entrada. |
| No hay dependencias completas | No hay `requirements.txt` raíz ni `pyproject.toml`; solo dos manifiestos parciales. | Instalación no reproducible. |
| No hay artefactos de modelos | El ZIP no contiene carpeta `models/`. | No se pueden cargar los modelos citados en la documentación. |
| Rutas absolutas de Windows | `l2_tactic/config.py` apunta a `C:/proyectos/HRM/models/L2/...`. | No es portable y depende de una estructura local concreta. |
| Error de sintaxis | `backtesting/performance_analyzer.py`, `backtesting/probador_estrategia2.py`, `l1_operational/signal_processor.py`, `tests/test_profit_taking.py`, `tests/test_trend_following.py` y `tests/test_l3_unified.py`. | Esos módulos no pueden importarse/ejecutarse tal como están. |
| Importación a módulo ausente | `runtime_loop.py` importa `l3_strategy.decision_maker`, que no aparece en el ZIP. | Puede impedir el arranque de esa ruta del runtime. |
| Sin plantilla de entorno | No hay `.env.example`. | Configuración externa poco clara. |
| Configuración inconsistente | Existen valores de capital y modos distintos en diferentes módulos/configuraciones. | Riesgo de comportamiento inesperado. |

La comprobación de sintaxis se realizó con `compileall` sobre el código Python del ZIP. Detectó los errores enumerados; por tanto, no se puede afirmar que todos los módulos compilen.

### 10.2. Qué parece implementado en código

- Estructura modular L1/L2/L3.
- Cliente Binance basado en CCXT.
- Cliente simulado y lógica paper.
- Gestores de órdenes, cartera y riesgo.
- Procesamiento de datos y señales.
- Módulos de clasificación de régimen y sentimiento.
- Integración con frameworks de ML/RL.
- Componentes de autoaprendizaje.
- Scripts de entrenamiento y backtesting.
- Logging, estado y utilidades de limpieza.

### 10.3. Qué no se puede dar por funcional

- Arranque completo de `main.py` en un entorno limpio.
- Ejecución de todos los ciclos L1-L2-L3 de extremo a extremo.
- Carga de los modelos citados en los documentos.
- Conexión API directa con los proveedores LLM comerciales.
- Reentrenamiento automático completo y despliegue seguro de modelos.
- Backtests reproducibles y con resultados verificados.
- Trading real seguro en Binance.
- Funcionamiento de todas las pruebas.

### 10.4. Veredicto

**Estado: prototipo avanzado / código en desarrollo, no validado para producción.**

El repositorio tiene una cantidad considerable de código y una arquitectura ambiciosa, pero el ZIP recibido no representa una distribución lista para instalar y operar. Los errores de sintaxis, los módulos ausentes, las dependencias incompletas y la ausencia de modelos impiden certificar su funcionamiento integral.

No sería correcto afirmar que “las nueve IAs” o todos los modelos de proveedores externos están funcionando. Tampoco sería correcto afirmar que el bot puede operar de forma rentable o segura.

## 11. Orden recomendado para poner el proyecto en condiciones

1. **Congelar el modo paper.** No introducir claves de trading real.
2. **Consolidar dependencias.** Crear un `requirements.txt` raíz con versiones compatibles y separar extras opcionales (ML, sentimiento, entrenamiento).
3. **Corregir sintaxis.** Resolver los errores detectados por `compileall`.
4. **Resolver imports ausentes.** En particular, revisar `l3_strategy.decision_maker` y todas las dependencias internas.
5. **Eliminar rutas absolutas.** Usar rutas relativas a la raíz del proyecto y configuración externa.
6. **Aportar los artefactos.** Incorporar o documentar cómo obtener los modelos y datasets requeridos, con hashes/versiones.
7. **Unificar configuración.** Definir una única fuente de verdad para modo, capital, símbolos y límites de riesgo.
8. **Añadir `.env.example`.** Documentar variables y marcar cuáles son opcionales.
9. **Ejecutar pruebas por capas.** Primero validadores y datos; después L1; luego L2; luego L3; por último integración completa.
10. **Probar paper trading prolongado.** Verificar órdenes, fills, balances, PnL, reinicios y recuperación.
11. **Auditar el backtesting.** Evitar look-ahead bias, leakage, costes omitidos y sobreajuste.
12. **Solo después evaluar live.** Revisión de seguridad independiente, límites de API, claves sin permisos de retirada y capital mínimo de prueba.

## 12. Mapa rápido de archivos

```text
HRM-master/
├── main.py                      # Punto de entrada principal del backend
├── frontend/                    # Aplicación web (React + TypeScript + Vite, versión corregida)
├── bootstrap.py                 # Inicialización y apagado
├── runtime_loop.py              # Bucle de ejecución alternativo
├── execution_gateway.py         # Gateway de ejecución y riesgo
├── auto_learning_system.py      # Componentes de autoaprendizaje
├── integration_auto_learning.py # Integración del autoaprendizaje
├── fix_l3_dominance.py          # Reglas de dominancia L3
├── config_loader.py             # Estado/configuración inicial
├── comms/                       # Bus de mensajes y esquemas
├── core/                        # Estado, cartera, validación y métricas
├── system/                      # Bootstrap, pipeline y coordinación
├── l1_operational/              # Órdenes, exchange y ejecución
├── l2_tactic/                   # Señales tácticas y modelos
├── l3_strategy/                 # Régimen, sentimiento y estrategia
├── configs/                     # YAML de backtest y live
├── backtesting/                 # Pruebas históricas
├── ml_training/                 # Entrenamiento offline
├── sentiment/                   # Gestión de sentimiento
├── storage/                     # Persistencia y logs de paper
├── hacienda/                    # Herramientas fiscales FIFO
├── scripts/                     # Arranque y diagnóstico
├── tests/                       # Pruebas
├── docs/                        # Documentación técnica existente
└── archived/                    # Configuración/estado archivado
```

## 13. Documentación interna consultada

Esta documentación se ha elaborado contrastando el código y los documentos incluidos en el ZIP, entre ellos:

- `ESTADO_ACTUAL.md`
- `DIRECTORY STRUCTURE.md`
- `CONTENT FILES.md`
- `docs/LISTADO_9_IAS_Y_MODELOS.md`
- `docs/AUTOLEARNING_ANALYSIS_AND_PLAN.md`
- `docs/Sistema_Normalizado.md`
- `docs/PATCH_README.md`
- `configs/config_live.yaml`
- `configs/config_backtest.yaml`

Los documentos internos pueden describir objetivos, estados anteriores o funcionalidades previstas. Cuando no coinciden con el código disponible, este README prioriza lo comprobable en el ZIP.

---

**Aviso de riesgo:** HRM es software experimental de trading. Las criptomonedas son volátiles y se puede perder todo el capital invertido. No utilices este repositorio con fondos reales sin una auditoría técnica y financiera independiente.


## 14. Interfaz web y control de arranque en Windows

La versión corregida del proyecto incorpora una interfaz web en la carpeta de frontend, basada en React, TypeScript y Vite. El backend Python conserva `main.py` como punto de entrada.

> **Nota sobre nombres de carpeta:** los scripts de Windows buscan automáticamente una carpeta llamada `frontend`, `front end` o `front-end` en la raíz del repositorio. La carpeta válida debe contener `package.json`.

### 14.1. Requisitos

- Windows 10/11.
- Python instalado y dependencias del backend preparadas.
- Node.js y npm instalados.
- Dependencias del frontend instaladas previamente desde su carpeta (`npm install`).

Los scripts de control no instalan paquetes, no crean ni reinician entornos virtuales y no modifican configuraciones.

### 14.2. Archivos de control

Copia estos cuatro archivos en la raíz del repositorio, junto a `main.py`:

- `HRM_INICIAR.bat`: inicia backend y frontend en dos ventanas de consola independientes.
- `HRM_DETENER.bat`: detiene únicamente los procesos que haya iniciado el controlador de HRM.
- `HRM_REINICIAR.bat`: ejecuta la parada y después vuelve a iniciar ambos servicios.
- `hrm_control.ps1`: implementación común utilizada por los tres archivos `.bat`.

El backend se inicia con `python main.py`. El frontend se inicia mediante el script `dev` declarado en el `package.json` (habitualmente `npm run dev`, propio de Vite). Si no existe el script `dev`, el controlador prueba `start`.

### 14.3. Uso

Haz doble clic en el archivo correspondiente o ejecútalo desde una terminal:

```bat
HRM_INICIAR.bat
HRM_DETENER.bat
HRM_REINICIAR.bat
```

El controlador guarda los PID de las ventanas que abre en `.hrm_runtime/`. La parada utiliza esos PID y termina sus procesos descendientes; no busca ni mata indiscriminadamente todos los procesos Python o Node del equipo.

Si una ventana se cierra manualmente, su PID puede dejar de estar activo. En ese caso, vuelve a ejecutar `HRM_INICIAR.bat`.

### 14.4. Comprobaciones y límites

- El arranque comprueba que `main.py`, la carpeta del frontend, `package.json`, Python, Node.js y npm estén disponibles.
- El controlador no borra `initial_state.json`, historiales, operaciones, modelos, logs ni otros datos persistentes.
- El controlador no cambia `PAPER_MODE`, `BINANCE_MODE` ni las claves de API. Mantén HRM en paper mientras no se haya validado de forma independiente.
- Los scripts abren los servicios, pero no certifican que el backend complete su inicialización ni que el frontend compile correctamente. Comprueba los mensajes de ambas ventanas y los tests del proyecto.
- La URL local concreta del frontend depende de la configuración de Vite. Normalmente Vite muestra la dirección en la propia consola.


## 15. Almacenamiento PAPER separado por perfil

HRM mantiene dos perfiles PAPER físicamente independientes. El modo configurado se aplica en el siguiente arranque y `system.preparar_sesion` selecciona el perfil antes de inicializar la cartera, el logger de operaciones y la API.

### 15.1. Estructura de datos

```text
data/
└── paper/
    ├── persistente/
    │   ├── portfolio_state.json
    │   ├── portfolio_log.csv
    │   └── paper_trades/
    │       ├── paper_trades.json
    │       ├── paper_trades.csv
    │       └── summary.json
    └── limpia/
        ├── portfolio_state.json
        ├── portfolio_log.csv
        └── paper_trades/
            ├── paper_trades.json
            ├── paper_trades.csv
            └── summary.json
```

`persistent_state/modo_sesion.json` sigue guardando únicamente el modo seleccionado. No contiene balances ni historial. El estado LIVE mantiene su ruta independiente `persistent_state/portfolio_state_live.json`; el perfil PAPER no escribe en ella.

### 15.2. Comportamiento de los perfiles

- **PERSISTENTE:** conserva cartera, operaciones, estadísticas y evolución de cartera entre reinicios.
- **LIMPIA:** utiliza otro directorio. Ese directorio se vacía al arrancar en modo limpia, pero nunca borra ni modifica el perfil persistente.
- **Cambio de modo:** se guarda la selección y se aplica al reiniciar HRM. Al volver de limpia a persistente se recupera el historial persistente anterior, sin mezclar operaciones.
- **LIVE:** no aplica la limpieza PAPER ni utiliza los archivos de los perfiles PAPER.
- **Botón «Reiniciar datos PAPER»:** es una acción distinta de cambiar a LIMPIA. Solo aparece en PAPER persistente, exige escribir `CONFIRMAR` y programa un reinicio diferido del perfil persistente. No borra nada al pulsarlo; el borrado ocurre en el siguiente arranque PAPER persistente. No elimina el perfil LIMPIA ni configuración, credenciales, modelos o logs técnicos.

### 15.3. Migración de instalaciones anteriores

En el primer arranque con esta versión, HRM copia al perfil persistente los datos heredados que encuentre:

- `persistent_state/portfolio_state_live.json`, solo si el contenido identifica explícitamente una cartera `simulated` o `paper`. Si no contiene el modo, no se migra automáticamente para evitar importar por error un estado LIVE.
- Los archivos JSON/CSV de `data/paper_trades/`.
- `data/portfolios/portfolio_log.csv`.

La migración **copia** los archivos y no elimina ni sobrescribe el origen. Si el archivo de destino ya existe, se respeta el destino y no se reemplaza. Se registra una única marca `persistent_state/migracion_paper_perfiles_v1.json`: así, un reinicio explícito del perfil persistente no vuelve a importar los archivos legacy conservados como copia de seguridad. Los originales antiguos permanecen intactos; a partir de la migración, HRM utiliza exclusivamente las rutas del perfil activo.

### 15.4. Componentes que respetan el perfil

- `storage/perfil_datos.py`: fuente común de rutas y selección de perfil.
- `storage/paper_trade_logger.py`: instancia un logger independiente por directorio de perfil. No crea un logger global al importar el módulo.
- `core/portfolio_manager.py`: guarda y restaura la cartera en el perfil activo. Al restaurar PAPER, hidrata los balances del exchange simulado antes de sincronizar la cartera.
- `api/servicios/servicio_operaciones.py`: consulta el JSON de operaciones del perfil activo. La variable `HRM_RUTA_OPERACIONES`, si está definida, sigue teniendo prioridad como ruta explícita.
- `system/preparar_sesion.py`: realiza la migración y, en modo limpia, borra exclusivamente el directorio de ese perfil. También consume el marcador de reinicio persistente solo después de completar su limpieza.
- `POST /sesion/reiniciar-datos`: valida modo operativo PAPER, perfil persistente activo/configurado y confirmación exacta antes de crear el marcador diferido.
- `bootstrap.py` y `system/system_cleanup.py`: la limpieza técnica ya no borra operaciones PAPER en cada arranque.

### 15.5. Verificación

Desde la raíz del repositorio, ejecuta las pruebas focalizadas:

```bash
pytest -q tests/test_preparar_sesion.py tests/test_perfil_datos.py tests/api/test_rutas_operaciones.py
```

Después de actualizar, arranca primero en PAPER persistente y comprueba que aparecen las operaciones antiguas. Cambia a limpia, reinicia y verifica que empieza sin operaciones. Vuelve a persistente, reinicia y comprueba que reaparecen exactamente la cartera y el historial anteriores. No utilices LIVE para validar esta funcionalidad.

**Importante:** cambiar a LIMPIA y pulsar «Reiniciar datos PAPER» son operaciones distintas. LIMPIA solo vacía `data/paper/limpia`; el botón programa el borrado de `data/paper/persistente` para el siguiente arranque PAPER persistente.
