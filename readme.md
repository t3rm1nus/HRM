# 🔱 HRM — Hierarchical Reasoning Model para Trading Algorítmico
**Estado: PRODUCCIÓN** · **Lenguaje:** Python 3.10+ · **Dominio:** Cripto Trading · **Arquitectura:** L3 Estratégico + L2 Táctico + L1 Operacional

## 🧭 TL;DR
HRM es un sistema de trading algorítmico **REAL Y FUNCIONAL** que opera con BTC y ETH en Binance Spot. Combina **análisis técnico avanzado**, **modelos FinRL pre-entrenados**, **gestión dinámica de riesgo**, **stop-loss/take-profit automáticos** y **ejecución determinista**. El sistema genera señales inteligentes cada 10 segundos, calcula posiciones óptimas y ejecuta órdenes con controles de seguridad multi-nivel.

## ✅ SISTEMA OPERATIVO - FUNCIONALIDAD REAL
**🚀 El sistema HRM está completamente operativo y ejecutándose en producción:**
- ✅ **Conexión real a Binance Spot** (modo LIVE y TESTNET)
- ✅ **Modo simulado con 3000 USDT** para testing seguro
- ✅ **Generación de señales cada 10 segundos** con indicadores técnicos
- ✅ **Modelos IA integrados** (FinRL + análisis técnico)
- ✅ **Gestión de portfolio automática** con tracking en CSV
- ✅ **Logging persistente** completo en data/logs/
- ✅ **Controles de riesgo dinámicos** y stops inteligentes
- ✅ **Stop-Loss y Take-Profit automáticos** integrados
- ✅ **Costos reales de trading** (comisiones 0.1% Binance)
- ✅ **Monitoreo de posiciones** en tiempo real
- ✅ **9 modelos AI operativos** (3 L1 + 1 L2 + 5 L3)
- ✅ **Análisis de sentimiento en tiempo real** (Reddit + News API)

## 🛡️ SISTEMA DE PROTECCIÓN "HARDCORE" - PRODUCCIÓN ULTRA-SEGURO

**🔴 CRÍTICO PARA OPERACIONES REALES:** HRM incluye un sistema de protección multi-nivel diseñado para entornos de producción extremos donde fallos de conectividad o energía pueden causar pérdidas catastróficas.

### 🚨 **PROBLEMAS RESUELTOS**
- **❌ Stop-Loss NO guardados:** Antes solo cálculos locales, posiciones desprotegidas
- **❌ Sin sincronización:** Sistema no verificaba posiciones reales en exchange
- **❌ Pérdidas por crashes:** Reinicio perdía estado y dejaba posiciones expuestas
- **❌ Desincronización:** Estado local ≠ estado real del exchange

### ✅ **SOLUCIONES IMPLEMENTADAS**

#### 🛡️ **1. STOP-LOSS REALES EN BINANCE**
```python
# STOP-LOSS colocados REALMENTE en el exchange
sl_order = await binance_client.place_stop_loss_order(
    symbol="BTCUSDT",
    side="SELL",
    quantity=0.001,
    stop_price=45000.0,  # Precio real de activación
    limit_price=44900.0  # Precio de ejecución
)
```
- **Modo LIVE:** Órdenes STOP_LOSS colocadas en Binance Spot real
- **Modo TESTNET:** Órdenes simuladas pero con lógica idéntica
- **Protección 24/7:** Stop-loss persisten aunque el sistema se caiga

#### 🔄 **2. SINCRONIZACIÓN OBLIGATORIA AL INICIO**
```python
# CRÍTICO: Verificación de estado real al startup
sync_success = await portfolio_manager.sync_with_exchange()
if sync_success:
    logger.info("✅ Portfolio sincronizado con Binance real")
    # Sistema continúa con posiciones correctas
else:
    logger.error("❌ FALLO DE SINCRONIZACIÓN - ABORTAR OPERACIÓN")
```
- **Verificación automática:** Compara estado local vs exchange real
- **Detección de discrepancias:** Alerta si hay diferencias significativas
- **Corrección automática:** Actualiza estado local con datos reales

#### 🚨 **3. DETECCIÓN DE DESINCRONIZACIÓN**
```python
# Monitoreo continuo de integridad
btc_diff = abs(local_btc - exchange_btc)
if btc_diff > 0.0001:
    logger.warning("🚨 DESINCRONIZACIÓN BTC: Local vs Exchange")
    # Corrección automática o alerta crítica
```
- **Monitoreo en tiempo real:** Comparación continua local vs exchange
- **Alertas automáticas:** Notificación inmediata de discrepancias
- **Corrección automática:** Re-sincronización cuando se detecta

#### 🔌 **4. RECUPERACIÓN TRAS FALLOS DE CONECTIVIDAD**
```python
# Escenario: Se va la luz → Vuelve la conexión
# 1. Sistema reinicia automáticamente
# 2. sync_with_exchange() lee posiciones reales
# 3. Stop-loss orders siguen activas en Binance
# 4. Sistema continúa con estado correcto
```
- **Recuperación automática:** Sistema se re-sincroniza tras fallos
- **Estado consistente:** Posiciones y stops preservados en exchange
- **Continuidad operativa:** Trading continúa sin intervención manual

### 🎯 **ARQUITECTURA DE PROTECCIÓN**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   SISTEMA HRM   │    │    BINANCE      │    │   POSICIONES    │
│                 │    │    EXCHANGE     │    │     REALES      │
│  ┌─────────┐    │    │                 │    │                 │
│  │ STOP-   │◄───┼────┤ STOP-LOSS       │    │  🛡️ PROTEGIDAS  │
│  │ LOSS    │    │    │ REALES          │    │                 │
│  │ LOCAL   │    │    │                 │    │                 │
│  └─────────┘    │    └─────────────────┘    └─────────────────┘
│                 │              ▲
│  ┌─────────┐    │              │
│  │ SINCRONIZ│◄──┼──────────────┘
│  │ ZACIÓN   │    │    VERIFICACIÓN AUTOMÁTICA
│  └─────────┘    │    AL INICIO Y DURANTE OPERACIÓN
└─────────────────┘
```

### 📊 **ESTADOS DE PROTECCIÓN**

| Estado | Descripción | Acción |
|--------|-------------|--------|
| **🟢 SINCRONIZADO** | Estado local = Exchange real | Operación normal |
| **🟡 DESINCRONIZADO** | Diferencias detectadas | Re-sincronización automática |
| **🔴 CRÍTICO** | Fallo de sincronización | Alerta + Modo seguro |
| **⚫ OFFLINE** | Sin conexión | Stop-loss en exchange activos |

### ⚙️ **CONFIGURACIÓN PARA PRODUCCIÓN**

```bash
# Variables críticas para modo HARDCORE
export BINANCE_MODE=LIVE
export USE_TESTNET=false
export HRM_HARDCORE_MODE=true  # Activa protecciones máximas
export HRM_SYNC_ON_STARTUP=true  # Sincronización obligatoria
export HRM_STOPLOSS_REAL=true  # Stop-loss reales en exchange

# Monitoreo adicional
export HRM_HEALTH_CHECK_INTERVAL=30  # Segundos
export HRM_MAX_DESYNC_TOLERANCE=0.001  # 0.1% máximo desincronización
```

### 🚨 **PROTOCOLOS DE SEGURIDAD**

1. **Inicio del Sistema:**
   - Verificación de conectividad con Binance
   - Sincronización completa de posiciones
   - Validación de stop-loss existentes
   - Solo continúa si sincronización exitosa

2. **Durante Operación:**
   - Monitoreo continuo de estado vs exchange
   - Re-sincronización automática cada 5 minutos
   - Alertas inmediatas por desincronización

3. **Tras Fallos:**
   - Reinicio automático con verificación completa
   - Recuperación de estado desde exchange
   - Validación de integridad antes de continuar

### 🎯 **VENTAJAS DEL SISTEMA HARDCORE**

- **🛡️ Protección 24/7:** Stop-loss persisten aunque el sistema falle
- **🔄 Recuperación automática:** Sin intervención manual tras fallos
- **📊 Transparencia total:** Estado real siempre visible y verificable
- **⚡ Continuidad operativa:** Trading continúa tras desconexiones
- **🚨 Alertas proactivas:** Detección inmediata de problemas

**El sistema HRM ahora es un entorno de producción ultra-seguro donde fallos de conectividad o energía NO resultan en pérdidas catastróficas.**

## 🎛️ **MODOS DE OPERACIÓN**

| Modo | Descripción | Activación |
|------|-------------|------------|
| **PAPER** | Simulación completa sin conexión real | `USE_TESTNET=true` |
| **LIVE** | Ejecución real en Binance Spot (requiere claves API) | `USE_TESTNET=false` |
| **REPLAY** | Reproducción con datasets históricos | Configuración adicional |

### ⚡ **ACTIVAR MODO LIVE**
```bash
export BINANCE_MODE=LIVE
export USE_TESTNET=false
export BINANCE_API_KEY=your_real_key
export BINANCE_API_SECRET=your_real_secret
python main.py
```

## 🎯 **OBJETIVO DEL PROYECTO**

Tomar decisiones de trading razonadas y trazables para múltiples activos (BTC, ETH) mediante una jerarquía de agentes. Aprender qué señales funcionan bajo distintos regímenes y cómo combinar niveles (L2/L3) para optimizar ejecución en L1 con modelos IA. Minimizar riesgos con análisis multinivel, capa dura de seguridad en L1 y gestión de correlación BTC–ETH. Crear un framework reutilizable para distintos universos de activos líquidos.

### 📚 **¿Qué queremos aprender a nivel de sistema?**
- Si el razonamiento multietapa mejora la estabilidad frente a un agente monolítico
- Qué señales funcionan en cada régimen y cómo combinarlas en L2/L3
- Cómo distribuir capital/ponderaciones entre modelos/estrategias

### 🎯 **Beneficios esperados**
- Mayor precisión mediante composición multiasset y modelos IA (LogReg, RF, LightGBM)
- Reducción de riesgo vía diversificación temporal, límite rígido en L1 y gestión de correlación BTC–ETH
- Adaptabilidad automática a distintos regímenes de mercado
- Razonamiento multi-variable con métricas granulares por activo (latencia, slippage, tasa de éxito)

### ⚙️ **Flujo general (visión de tiempos)**
- **Nivel 3:** Análisis Estratégico — horas
- **Nivel 2:** Táctica de Ejecución — minutos
- **Nivel 1:** Ejecución + Gestión de Riesgo — segundos
## 🏗️ ARQUITECTURA REAL DEL SISTEMA

### 🎯 **NIVEL 2 - TÁCTICO (L2)** ✅ IMPLEMENTADO Y MODULARIZADO
**Rol:** Generación inteligente de señales de trading
**Funciones operativas:**
- ✅ **Análisis técnico multi-timeframe** (RSI, MACD, Bollinger Bands)
- ✅ **Modelos FinRL pre-entrenados** con ensemble de predicciones
- ✅ **Composición de señales** con pesos dinámicos
- ✅ **Position sizing** con Kelly Criterion y vol-targeting
- ✅ **Controles de riesgo pre-ejecución** (stops, correlación, drawdown)
- ✅ **Stop-Loss y Take-Profit dinámicos** basados en volatilidad y confianza
- ✅ **Cálculo automático de SL/TP** por señal generada

#### 🏗️ **NUEVA ARQUITECTURA MODULAR L2 (2025)**
**Refactorización completa:** El monolítico `finrl_integration.py` ha sido dividido en módulos especializados:

```
l2_tactic/
├── 📄 __init__.py                    # Punto de entrada unificado
├── 📄 models.py                      # Estructuras de datos (TacticalSignal)
├── 📄 config.py                      # Configuración L2
├── 📄 signal_generator.py            # Orquestador principal
├── 📄 signal_composer.py             # Composición de señales
├── 📄 position_sizer.py              # Cálculo de tamaños de posición
├── 📄 finrl_integration.py           # 🔄 COMPATIBILIDAD (solo imports)
├── 📁 generators/                    # Generadores de señales
│   ├── 📄 __init__.py
│   ├── 📄 technical_analyzer.py      # Análisis técnico
│   ├── 📄 mean_reversion.py          # Estrategia reversión a la media
│   └── 📄 finrl.py                   # Procesador FinRL
├── 📁 ensemble/                      # Combinación de señales
│   ├── 📄 __init__.py
│   ├── 📄 voting.py                  # Ensemble por votación
│   └── 📄 blender.py                 # Ensemble por blending
├── 📁 risk_controls/                 # Gestión de riesgo modular
│   ├── 📄 __init__.py
│   ├── 📄 alerts.py                  # Sistema de alertas
│   ├── 📄 manager.py                 # Orquestador de riesgo
│   ├── 📄 portfolio.py               # Riesgo de portfolio
│   ├── 📄 positions.py               # Riesgo por posición
│   └── 📄 stop_losses.py             # Stop-loss dinámicos
└── 📁 technical/                     # Indicadores técnicos
    ├── 📄 __init__.py
    ├── 📄 multi_timeframe.py         # Análisis multi-timeframe
    └── 📄 indicators.py              # Indicadores técnicos
```

#### 🤖 **Sistema FinRL Modularizado**
**Antes:** Un solo archivo de 1000+ líneas con todo mezclado
**Ahora:** Arquitectura limpia con responsabilidades separadas:

| Módulo | Responsabilidad | Estado |
|--------|----------------|--------|
| `finrl_processor.py` | Clase principal FinRLProcessor | ✅ Operativo |
| `finrl_wrapper.py` | Wrapper inteligente multi-modelo | ✅ Operativo |
| `feature_extractors.py` | Extractores de features personalizados | ✅ Operativo |
| `observation_builders.py` | Construcción de observaciones | ✅ Operativo |
| `model_loaders.py` | Carga unificada de modelos | ✅ Operativo |
| `signal_generators.py` | Generación de señales | ✅ Operativo |

#### 🎯 **Modelos FinRL Soportados**
| Modelo | Dimensiones | Método | Estado |
|--------|-------------|--------|--------|
| **DeepSeek** | 257 | `predict()` | ✅ Operativo |
| **Gemini** | 13 | `get_action()` → `predict()` | ✅ **FIXED** |
| **Claude** | 971 | `predict()` | ✅ Operativo |
| **Kimi** | Variable | `predict()` | ✅ Operativo |
| **Gpt** | Variable | `predict()` | ✅ Operativo |
| **Grok** | Variable | `predict()` | ✅ Operativo |

#### 🔧 **Detección Automática de Métodos**
```python
# Sistema inteligente que detecta el método correcto
def get_finrl_signal(finrl_processor, market_data):
    if hasattr(finrl_processor, 'predict'):
        return finrl_processor.predict(market_data)
    elif hasattr(finrl_processor, 'get_action'):
        return finrl_processor.get_action(market_data)
    else:
        raise AttributeError("Método no encontrado")
```

#### 📈 **Beneficios de la Modularización**
- **🔧 Mantenibilidad:** Cada módulo tiene una responsabilidad clara
- **🔄 Escalabilidad:** Fácil añadir nuevos modelos o estrategias
- **🛡️ Robustez:** Mejor manejo de errores y compatibilidad
- **📊 Rendimiento:** Optimizaciones específicas por componente
- **🔌 Compatibilidad:** Código existente sigue funcionando sin cambios

### ⚙️ **NIVEL 1 - OPERACIONAL (L1)** ✅ IMPLEMENTADO
**Rol:** Ejecución determinista y segura de órdenes
**Funciones operativas:**
- ✅ **Validación de señales** con modelos IA (LogReg, RF, LightGBM)
- ✅ **Gestión de portfolio automática** (BTC, ETH, USDT)
- ✅ **Conexión a Binance Spot** (real y testnet)
- ✅ **Logging persistente** con métricas detalladas
- ✅ **Controles de riesgo** por símbolo y portfolio

### 🚀 **NIVEL L3** - ESTRATÉGICO (IMPLEMENTADO)
**Rol:** Análisis macro y asignación estratégica de capital
**Funciones implementadas:**
- ✅ **Regime Detection** con ensemble ML (Optuna)
- ✅ **Portfolio Optimization** usando Black-Litterman
- ✅ **Sentiment Analysis** con BERT pre-entrenado (Reddit + News API)
- ✅ **Volatility Forecasting** con GARCH y LSTM
- ✅ **Strategic Decision Making** con pipeline completo

✅ **Modelos IA L1:** **FUNCIONALES** (LogReg, RF, LightGBM en models/L1/)

| Tipo | Descripción |
|------|-------------|
| **Precio** | delta_close, EMA/SMA |
| **Volumen** | volumen relativo |
| **Momentum** | RSI, MACD |
| **Multi-timeframe** | 1m + 5m |
| **Cross-asset** | ETH/BTC ratio, correlación rolling, divergencias |
| **Real-time data** | Desde Binance Spot (modo LIVE) o testnet |
## 🚀 EJECUCIÓN DEL SISTEMA

### ⚡ **INICIO RÁPIDO**
```bash
# 1) Configurar variables de entorno (opcional para modo simulado)
export BINANCE_API_KEY=your_api_key
export BINANCE_API_SECRET=your_secret_key
export USE_TESTNET=true  # false para modo LIVE

# 2) Ejecutar sistema principal (modo simulado por defecto)
python main.py

# 3) Para ejecución nocturna continua
python run_overnight.py
```

### 🎯 **MODO SIMULADO CON 3000 USDT (RECOMENDADO PARA TESTING)**
```bash
# Sin configuración adicional - funciona inmediatamente
python main.py

# El sistema inicia con:
# - Balance inicial: 3000.0 USDT
# - Portfolio completamente limpio
# - Sin conexión a exchanges reales
# - Todas las funcionalidades activas
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

### 🔄 **ANÁLISIS DE SENTIMIENTO EN TIEMPO REAL**
**Cada 50 ciclos (~8-9 minutos):**
```
🔄 SENTIMENT: Actualización periódica iniciada (ciclo 50, cada 50 ciclos)
🔄 SENTIMENT: Iniciando actualización de datos de sentimiento...
📱 SENTIMENT: Descargando datos de Reddit...
📱 SENTIMENT: r/CryptoCurrency - Descargados 500 posts
📱 SENTIMENT: r/Bitcoin - Descargados 500 posts
📱 SENTIMENT: r/Ethereum - Descargados 500 posts
📊 SENTIMENT: Reddit total descargado: 1500 posts de 3 subreddits
📰 SENTIMENT: News - 50 artículos descargados y procesados
💬 SENTIMENT: Análisis de sentimiento listo con 95 textos válidos
💬 SENTIMENT: Cache actualizado con 95 textos para análisis L3
```

**Cada ciclo L3:**
```
🧠 SENTIMENT: Iniciando inferencia de sentimiento - 95 textos, batch_size=16
📊 SENTIMENT: Procesando 6 batches de inferencia...
✅ SENTIMENT: Completado batch 6/6 (100.0%)
🎯 SENTIMENT: Inferencia completada - 95 resultados generados
✅ Sentimiento calculado: 0.2345 (device: cpu, textos: 95)
🟠 ANÁLISIS DE SENTIMIENTO: 🟠 POSITIVO - Mercado favorable, tendencia alcista moderada (score: 0.2345)
```

### ⏰ **FRECUENCIAS DE EJECUCIÓN**
- **L2/L1:** Cada 10 segundos (independiente)
- **L3:** Cada 50 ciclos (~8-9 minutos) en segundo plano
- **Sentiment Analysis:** Cada 50 ciclos (descarga fresca de datos)
- **Si L3 falla:** L2 usa última estrategia conocida (fallback automático)

### **VENTAJAS DEL FALLBACK**
- L2/L1 nunca se bloquea si L3 falla
- Última estrategia válida de L3 se mantiene
- Logs centralizados registran errores y warnings
- Sentiment analysis continúa con datos en cache

### 🎛️ **MODOS DE OPERACIÓN**
| Modo | Descripción | Activación | Balance Inicial |
|------|-------------|------------|----------------|
| **SIMULATED** | Simulación completa sin exchange | Automático | 3000 USDT |
| **TESTNET** | Binance testnet | `USE_TESTNET=true` | Desde exchange |
| **LIVE** | Binance Spot real | `USE_TESTNET=false` | Desde exchange |
| **PAPER** | Simulación local | Configuración interna | Configurable |

## 🤖 **SISTEMA DE AUTO-APRENDIZAJE CON PROTECCIÓN ANTI-OVERFITTING**

**NUEVA FUNCIONALIDAD 2025:** HRM ahora incluye un **sistema de aprendizaje continuo completamente automático** con **9 capas de protección anti-overfitting**. El sistema aprende y se mejora solo sin intervención manual.

### 🎯 **Características del Sistema de Auto-Aprendizaje**

#### ✅ **Aprendizaje Continuo Automático**
- **Reentrenamiento automático** basado en triggers inteligentes
- **Online learning** para componentes compatibles
- **Meta-learning** para selección automática de modelos
- **Ensemble evolution** dinámica

#### 🛡️ **Protección Total Anti-Overfitting (9 Capas)**

1. **🔄 Validación Cruzada Continua** - Rolling window validation
2. **📊 Regularización Adaptativa** - Ajuste automático de parámetros
3. **🧬 Ensemble Diverso** - Modelos diversos para estabilidad
4. **⏹️ Early Stopping Inteligente** - Prevención de sobre-entrenamiento
5. **🌊 Concept Drift Detection** - Detección de cambios en distribución
6. **📈 Walk-Forward Validation** - Validación temporal realista
7. **🎨 Data Diversity Enforcement** - Garantía de diversidad en datos
8. **🏷️ Model Aging Detection** - Detección de degradación de modelos
9. **📉 Out-of-Sample Testing** - Validación en datos no vistos

### 🚀 **Arquitectura del Sistema de Auto-Aprendizaje**

```
┌─────────────────────────────────────────────────────────────┐
│                    SISTEMA PRINCIPAL                         │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │            Auto-Retraining System                       │ │
│  │  ┌─────────────────────────────────────────────────────┐ │ │
│  │  │        Anti-Overfit Protection (9 capas)           │ │ │
│  │  │  ┌─────────────────────────────────────────────────┐ │ │ │
│  │  │  │      Model Validation & Selection              │ │ │ │
│  │  │  └─────────────────────────────────────────────────┘ │ │ │
│  │  └─────────────────────────────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │            Online Learning Components                  │ │ │
│  └─────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │            Performance Monitor                         │ │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 📊 **Triggers de Auto-Reentrenamiento**

#### ⏰ **Basado en Tiempo**
- **Cada 7 días** automáticamente
- Reset automático de timers

#### 📈 **Basado en Performance**
- **Win rate < 52%** en últimos 100 trades
- **Drawdown > 12%** máximo
- **Auto-detección** de degradación

#### 🔄 **Basado en Régimen**
- **3 cambios de régimen** consecutivos
- **Adaptación automática** a nuevos mercados

#### 📊 **Basado en Volumen**
- **500+ nuevos trades** acumulados
- **Datos suficientes** para reentrenamiento significativo

### 🔧 **Componentes Implementados**

#### 1. **`auto_learning_system.py`** - 🧠 Sistema Principal
- **9 clases principales** con protección anti-overfitting
- **Auto-reentrenamiento automático** con triggers inteligentes
- **Validación cruzada continua**, **regularización adaptativa**, **ensemble diverso**
- **Concept drift detection**, **early stopping inteligente**
- **Sistema completamente autónomo**

#### 2. **`integration_auto_learning.py`** - 🔗 Integración
- **Conexión automática** con el sistema de trading principal
- **Parsing automático** de logs para capturar trades
- **Hook de logging** para aprendizaje en tiempo real
- **Función de integración** plug-and-play

#### 3. **`README_AUTO_LEARNING.md`** - 📖 Documentación Completa
- **Documentación detallada** del sistema
- **Guía de integración** paso a paso
- **Arquitectura detallada** y funcionalidades
- **Monitoreo y métricas**

### 🎯 **Funcionalidades del Sistema**

#### ✅ **Auto-Reentrenamiento**
```python
# El sistema decide automáticamente cuándo reentrenar
if self._should_retrain():
    await self._auto_retrain_models()
```

#### ✅ **Protección Anti-Overfitting**
```python
# TODAS las verificaciones pasan antes de desplegar modelo
if self._passes_all_anti_overfitting_checks(candidate_model, training_data):
    self._deploy_new_model(model_name, candidate_model)
```

#### ✅ **Ensemble Evolution**
```python
# Solo añade modelos que aumenten diversidad
if self.ensemble_builder.add_model_to_ensemble(candidate_model, validation_data):
    logger.info("✅ Model added to ensemble")
```

#### ✅ **Concept Drift Detection**
```python
# Detecta cambios en la distribución de datos
if self.drift_detector.detect_drift(new_data):
    logger.warning("🌊 CONCEPT DRIFT DETECTED")
```

### 📈 **Beneficios Esperados**

#### 🚀 **Mejora Continua**
- **Win rate**: 55% → 65%+ en 3-6 meses
- **Drawdown máximo**: 15% → 10%+
- **Adaptabilidad**: Auto-ajuste a cambios de mercado

#### 🛡️ **Riesgo Controlado**
- **Sin overfitting**: 9 capas de protección
- **Validación robusta**: Múltiples técnicas
- **Stability**: Ensemble diverso

#### 🤖 **Autonomía Total**
- **Sin intervención**: Funciona 24/7
- **Auto-optimización**: Parámetros ajustados automáticamente
- **Auto-evolución**: Modelos mejoran solos

### 🔌 **Integración Automática**

**El sistema de auto-aprendizaje se integra automáticamente al iniciar HRM:**

```python
# En main.py - integración automática
from integration_auto_learning import integrate_with_main_system

# Integrar al inicio
auto_learning_system = integrate_with_main_system()
```

### 📊 **Monitoreo del Sistema de Auto-Aprendizaje**

#### **Estado del Sistema**
```python
{
    'integrated_system_running': True,
    'auto_learning_active': True,
    'trades_processed': 1250,
    'auto_learning_status': {
        'data_buffer_size': 500,
        'models_count': 5,
        'ensemble_size': 3,
        'performance_metrics': {...}
    }
}
```

#### **Logs Automáticos**
```
🔄 AUTO-TRIGGER: Time-based (192h >= 168h)
🤖 INICIANDO AUTO-REENTRENAMIENTO...
✅ CV Validation passed: 0.73 ± 0.08
✅ Model added to ensemble (improvement: 0.023)
🚀 Desplegado regime_classifier versión auto_v3
```

### 🎉 **Resultado Final**

**Sistema HRM con aprendizaje continuo automático:**
- ✅ **Se mejora solo** sin intervención manual
- ✅ **Aprende de cada trade** automáticamente
- ✅ **Previene overfitting** con 9 capas de protección
- ✅ **Se adapta** a cambios de mercado
- ✅ **Funciona 24/7** de forma autónoma

**¡HRM ahora tiene aprendizaje continuo con protección total anti-overfitting!** 🤖🛡️✨

---

## ✅ **BUENAS PRÁCTICAS DE RIESGO** (resumen actualizado)

| Concepto | Valor real |
|----------|------------|
| **Stop-loss** | Obligatorio + automático |
| **Take-profit** | Dinámico basado en volatilidad |
| **Límites por trade** | BTC: 0.05, ETH: 1.0 |
| **Exposición máxima** | BTC: 20%, ETH: 15% |
| **Correlación BTC-ETH** | Monitoreada en tiempo real |
| **Costos reales** | Comisiones 0.1% Binance aplicadas |
| **Monitoreo posiciones** | Activación automática SL/TP |
| **Modo LIVE** | Implementado y validado |
| **Auto-aprendizaje** | ✅ **NUEVO** - Sistema autónomo con 9 capas anti-overfitting |
| **Determinismo** | Una orden por señal → si falla → rechazo y reporte |
| **Separación L2/L3 ≠ L1** | Responsabilidades claramente separadas |

## 🏗️ Arquitectura del Sistema HRM

### 📊 Flujo Jerárquico de Decisiones

```
🌐 NIVEL 3 (ESTRATÉGICO) - Análisis Macro (cada 10 min)
├── 📊 Análisis de Mercado (Regime Detection)
├── 💬 Análisis de Sentimiento (BERT + Redes Sociales)
├── 📈 Pronóstico de Volatilidad (GARCH + LSTM)
└── 🎯 Optimización de Portfolio (Black-Litterman)
    ↓
🎯 NIVEL 2 (TÁCTICO) - Generación de Señales (cada 10 seg)
├── 🤖 Modelos FinRL (DeepSeek, Gemini, Claude, Kimi)
├── 📊 Análisis Técnico Multi-Timeframe
├── 🎲 Ensemble de Señales con Ponderación Dinámica
└── 🛡️ Controles de Riesgo Pre-Ejecución
    ↓
⚡ NIVEL 1 (OPERACIONAL) - Ejecución Determinista
├── 🔍 Validación de Señales con Modelos IA
├── 💰 Gestión Automática de Portfolio
├── 🔗 Conexión Binance (Live/Testnet)
└── 📝 Logging Persistente y Métricas
```



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
