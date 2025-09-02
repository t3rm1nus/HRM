# 🤖 README PARA IA - Sistema HRM Completo

## 📋 INFORMACIÓN GENERAL DEL PROYECTO

**Nombre:** HRM (Hierarchical Reasoning Model)  
**Tipo:** Sistema de Trading Algorítmico  
**Estado:** PRODUCCIÓN - Funcionando en tiempo real  
**Lenguaje:** Python 3.10+  
**Arquitectura:** Multi-nivel jerárquico (L2 + L1 implementados)  
**Exchange:** Binance Spot (LIVE y TESTNET)  
**Activos:** BTC/USDT, ETH/USDT  

## 🎯 OBJETIVO PRINCIPAL

Sistema de trading algorítmico que combina **análisis técnico avanzado**, **modelos FinRL pre-entrenados**, **gestión dinámica de riesgo** y **ejecución determinista**. Opera cada 10 segundos generando señales inteligentes y ejecutando órdenes con controles de seguridad multi-nivel.

## 🏗️ ARQUITECTURA REAL IMPLEMENTADA

### ✅ **NIVEL 2 - TÁCTICO (L2)** - IMPLEMENTADO Y FUNCIONAL
**Rol:** Generación inteligente de señales de trading  
**Componentes principales:**
- `L2TacticProcessor` - Orchestrador principal
- `FinRL Integration` - Modelos PPO desde `models/L2/`
- `Signal Composer` - Ensemble con pesos dinámicos
- `Risk Overlay` - Controles de riesgo pre-ejecución
- `Technical Analysis` - RSI, MACD, Bollinger Bands multi-timeframe

**Flujo operacional:**
1. Recibe `market_data` y `features` desde `main.py`
2. Procesa con modelos FinRL (PPO) 
3. Combina con análisis técnico multi-timeframe
4. Genera señales con `BlenderEnsemble`
5. Aplica controles de riesgo y position sizing
6. Entrega `TacticalSignals` a L1

### ✅ **NIVEL 1 - OPERACIONAL (L1)** - IMPLEMENTADO Y FUNCIONAL
**Rol:** Ejecución determinista y segura de órdenes  
**Componentes principales:**
- `OrderManager` - Procesa señales L2 con validación completa
- `AI Models` - 3 modelos IA funcionales (LogReg, RF, LightGBM)
- `Trend AI` - Filtrado inteligente con ensemble de modelos
- `DataFeed` - Conexión real a Binance Spot funcionando
- `Portfolio Management` - Tracking automático BTC/ETH/USDT
- `BinanceClient` - Configurado para LIVE y TESTNET

**Modelos IA L1 (funcionales):**
- `models/L1/modelo1_lr.pkl` - Logistic Regression (4KB)
- `models/L1/modelo2_rf.pkl` - Random Forest (3.6GB)
- `models/L1/modelo3_lgbm.pkl` - LightGBM (1.4MB)

**Flujo operacional:**
1. Recibe señales desde `L2TacticProcessor`
2. **Valida con 3 modelos IA** (LogReg, RF, LightGBM) + Trend AI
3. Valida parámetros (symbol, side, qty, stop_loss)
4. Simula ejecución de orden (por seguridad)
5. Actualiza portfolio automáticamente
6. Registra métricas en logs persistentes

### 🚧 **NIVELES L3/L4** - NO IMPLEMENTADOS
- **L3 Estratégico:** Solo placeholders sin funcionalidad
- **L4 Meta:** Solo placeholders sin funcionalidad
- **Nota:** El sistema actual opera efectivamente con L2+L1

## 🔄 FLUJO DE EJECUCIÓN EN TIEMPO REAL

### **Ciclo Principal (cada 10 segundos):**

```python
# 1. Recolección de datos
data_feed.fetch_data(symbol, limit=100)  # OHLCV desde Binance

# 2. Cálculo de indicadores técnicos
calculate_technical_indicators(df)  # RSI, MACD, Bollinger Bands

# 3. Preparación de features para L2
prepare_features_for_l2(state)  # 52 features por símbolo

# 4. Procesamiento L2 - Generación de señales
l2_processor.process(state, market_data, features_by_symbol)

# 5. Procesamiento L1 - Validación y ejecución
l1_order_manager.handle_signal(signal)  # Con modelos IA

# 6. Actualización de portfolio
update_portfolio_from_orders(state, orders)

# 7. Logging persistente
log_cycle_data(state, cycle_id, start_time)
```

### **Estructura de State Global:**
```python
state = {
    "mercado": {symbol: {} for symbol in SYMBOLS},  # Datos de mercado
    "estrategia": "neutral",
    "portfolio": {'BTCUSDT': 0.0, 'ETHUSDT': 0.0, 'USDT': 3000.0},
    "universo": SYMBOLS,
    "exposicion": {symbol: 0.0 for symbol in SYMBOLS},
    "senales": {},
    "ordenes": [],
    "riesgo": {},
    "deriva": False,
    "ciclo_id": 0,
    "features": {},  # Features para L2
    "indicadores_tecnicos": {}  # Indicadores calculados
}
```

## 📁 ESTRUCTURA DE ARCHIVOS CRÍTICOS

### **Puntos de Entrada:**
- `main.py` - Sistema principal (ciclo cada 10s)
- `run_overnight.py` - Ejecución nocturna continua

### **L2 Táctico (`l2_tactic/`):**
- `signal_generator.py` - L2TacticProcessor principal
- `finrl_integration.py` - Modelos FinRL PPO
- `signal_composer.py` - BlenderEnsemble
- `risk_overlay.py` - Controles de riesgo
- `technical/multi_timeframe.py` - Análisis multi-timeframe
- `models.py` - TacticalSignal, MarketFeatures

### **L1 Operacional (`l1_operational/`):**
- `order_manager.py` - OrderManager con modelos IA
- `data_feed.py` - DataFeed para Binance
- `binance_client.py` - Cliente Binance Spot
- `trend_ai.py` - Trend AI con ensemble
- `models.py` - Signal, ExecutionReport
- `config.py` - Configuración L1

### **Modelos IA:**
- `models/L1/modelo1_lr.pkl` - Logistic Regression
- `models/L1/modelo2_rf.pkl` - Random Forest  
- `models/L1/modelo3_lgbm.pkl` - LightGBM
- `models/L2/` - Modelos FinRL PPO

### **Infraestructura:**
- `core/logging.py` - Sistema de logs
- `core/persistent_logger.py` - Logging persistente
- `comms/config.py` - Configuración global
- `data/loaders.py` - Generación de features

### **Datos y Logs:**
- `data/logs/` - Logs del sistema
- `data/portfolio/` - Tracking portfolio CSV
- `data/btc_1m_2.csv`, `data/eth_1m_2.csv` - Datos históricos

## 🔧 CONFIGURACIÓN Y EJECUCIÓN

### **Variables de Entorno Requeridas:**
```bash
export BINANCE_API_KEY=your_api_key
export BINANCE_API_SECRET=your_secret_key
export USE_TESTNET=true  # false para modo LIVE
```

### **Comandos de Ejecución:**
```bash
# Sistema principal
python main.py

# Ejecución nocturna
python run_overnight.py
```

### **Modos de Operación:**
- **TESTNET:** Binance testnet (recomendado)
- **LIVE:** Binance Spot real
- **PAPER:** Simulación local

## 📊 FUNCIONALIDADES TÉCNICAS DETALLADAS

### **Análisis Técnico (L2):**
- **Indicadores:** RSI, MACD, Bollinger Bands, EMA/SMA
- **Multi-timeframe:** 1m, 5m, 15m, 1h
- **Features:** 52 features por símbolo
- **Ensemble:** BlenderEnsemble con pesos dinámicos

### **Modelos IA (L2):**
- **FinRL PPO:** Modelos pre-entrenados desde `models/L2/`
- **Predicciones:** Ensemble de modelos con cache
- **Performance:** Latencia ~100-200ms por ciclo

### **Validación IA (L1):**
- **3 Modelos:** LogReg, Random Forest, LightGBM
- **Trend AI:** Ensemble ponderado (RF: 30%, LGBM: 50%, LR: 20%)
- **Validación:** Cada señal pasa por filtros IA antes de ejecución

### **Gestión de Riesgo:**
- **Stop-loss:** Obligatorio por orden
- **Límites:** BTC (0.05 max), ETH (1.0 max)
- **Exposición:** BTC (20%), ETH (15%)
- **Correlación:** Monitoreo BTC-ETH en tiempo real

### **Portfolio Management:**
- **Tracking automático:** BTC, ETH, USDT balances
- **CSV persistence:** `data/portfolio/portfolio_history_YYYYMMDD.csv`
- **Actualización:** Cada ciclo basado en órdenes "ejecutadas"

## 📈 MÉTRICAS Y LOGGING

### **Logs Persistentes:**
- **Ciclos:** `data/logs/overnight_YYYYMMDD_HHMMSS.log`
- **Portfolio:** `data/portfolio/portfolio_history_YYYYMMDD.csv`
- **Métricas:** Latencia, señales generadas, órdenes ejecutadas

### **Métricas en Tiempo Real:**
```
[TICK] Ciclo 1234 completado en 2.34s
✅ Features preparadas: 52 total para 2 símbolos
[L2] Señales generadas: 1 BTC_buy, 0 ETH
[L1] Órdenes procesadas: 1 de 1 señales
PORTFOLIO TOTAL: 3000.00 USDT | BTC: 0.00000 (0.00$) | ETH: 0.000 (0.00$) | USDT: 3000.00$
```

## 🔍 ARCHIVOS NO UTILIZADOS (POTENCIALMENTE OBSOLETOS)

### **Archivos de Entrenamiento ML:**
- `ml_training/modelo1_train_lgbm_modelo1.py`
- `ml_training/modelo1_train_logreg_modelo1.py`
- `ml_training/modelo1_train_rf_modelo1.py`
- `ml_training/train_lgbm_modelo3.py`
- `ml_training/train_rf_modelo2.py`

### **Archivos de Test:**
- `test_binance.py` - Test simple de conexión
- `tests/backtester.py` - Backtester que usa módulo inexistente

### **Módulos sin uso:**
- `core/scheduler.py` - No importado
- `core/utils.py` - No importado
- `storage/` - Solo referenciado en documentación
- `monitoring/` - Solo auto-importación

### **Niveles no implementados:**
- `l3_strategy/` - Solo placeholders
- `l4_meta/` - Solo placeholders

## ⚠️ LIMITACIONES ACTUALES

### **Funcionalidad Simulada:**
- **Ejecución de órdenes:** Solo simulación (por seguridad)
- **No envía órdenes reales** a Binance (modo PAPER)

### **Niveles Faltantes:**
- **L3 Estratégico:** No implementado
- **L4 Meta:** No implementado

### **Backtesting:**
- **Sistema de backtesting:** No completamente funcional
- **Tests automatizados:** Limitados

## 🚀 PRÓXIMOS PASOS RECOMENDADOS

1. **Activar ejecución real** en L1 (actualmente simulada)
2. **Implementar L3** para decisiones estratégicas
3. **Desarrollar backtesting** completo
4. **Optimizar performance** de modelos FinRL
5. **Añadir más activos** (ADA, SOL, etc.)

## 📋 RESUMEN EJECUTIVO PARA IA

**HRM es un sistema de trading algorítmico REAL Y FUNCIONAL** que opera con BTC y ETH en Binance Spot. El sistema combina análisis técnico avanzado, modelos FinRL pre-entrenados, gestión dinámica de riesgo y ejecución determinista.

**Arquitectura actual:** L2 (generación de señales) + L1 (ejecución operacional)  
**Estado:** PRODUCCIÓN - Ejecutándose cada 10 segundos  
**Funcionalidad:** Conexión real a Binance, modelos IA integrados, portfolio tracking automático  
**Limitación principal:** Ejecución simulada (no envía órdenes reales por seguridad)

El sistema está **completamente operativo** y puede ser ejecutado inmediatamente con `python main.py` o `python run_overnight.py` para ejecución continua.
