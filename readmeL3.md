🌟 L3_Strategic - Nivel Estratégico de Decisión
🎯 Objetivo

L3_Strategic es el nivel superior de toma de decisiones que define la estrategia global del sistema de trading. Analiza condiciones macroeconómicas, tendencias de mercado y patrones a largo plazo para establecer el régimen de mercado, asignación de activos y apetito de riesgo que guiarán las decisiones tácticas de L2.

## ✅ ESTADO ACTUAL: COMPLETAMENTE IMPLEMENTADO Y FUNCIONAL
**L3 está completamente desarrollado con modelos entrenados y pipeline operativo. El sistema HRM incluye L3+L2+L1 funcionando en producción con análisis estratégico avanzado.**

🚫 Lo que L3 NO hace
❌ No hace
No genera señales de trading específicas (responsabilidad de L2)
No ejecuta órdenes (responsabilidad de L1)
No analiza datos técnicos en tiempo real
No gestiona el riesgo operacional por trade
No interactúa directamente con los exchanges
✅ Lo que L3 SÍ hace
✅ Funcionalidad	Descripción
Análisis Macro	Evalúa condiciones económicas globales y tendencias del mercado
Regime Detection	Identifica el régimen de mercado actual (bull, bear, range, volatile)
Asset Allocation	Define la asignación óptima de capital entre diferentes activos
Risk Appetite	Establece el nivel de riesgo permitido según condiciones del mercado
Strategic Signals	Genera directrices estratégicas para guiar a L2
Portfolio Optimization	Optimiza la cartera global basado en modelos de Markowitz y Black-Litterman
Market Sentiment	Analiza el sentimiento del mercado mediante NLP y redes sociales
Volatility Forecasting	Predice volatilidad futura usando GARCH / LSTM

🏗️ Arquitectura del Sistema
┌─────────────────────────────────────────────────┐
│                   L3_Strategic                  │
│                                                 │
│  ┌─────────────┐    ┌─────────────────────┐    │
│  │  Macro      │    │   Regime            │    │
│  │  Analysis   │───▶│   Detection         │    │
│  └─────────────┘    └─────────────────────┘    │
│                          │                     │
│  ┌─────────────┐    ┌────▼─────────────────┐   │
│  │  Sentiment  │    │   Portfolio          │   │
│  │  Analysis   │───▶│   Optimization       │   │
│  └─────────────┘    └─────────────────────┘    │
│                          │                     │
│                   ┌──────▼──────────────┐      │
│                   │  Risk Appetite      │      │
│                   │  Calculator         │      │
│                   └─────────────────────┘      │
│                          │                     │
│                   ┌──────▼──────────────┐      │
│                   │  Strategic          │      │
│                   │  Decision Maker     │      │
│                   └─────────────────────┘      │
└─────────────────────────▼──────────────────────┘
                          │
                  Strategic Guidelines → L2


Diagrama resumido del flujo con fallback:

┌──────────────┐
│   L3_Strategic│
│  periodic     │
└───────┬──────┘
        │ strategic_guidelines
        ▼
┌──────────────┐
│   L2_Tactic  │ <──── fallback si L3 falla
└───────┬──────┘
        │ signals
        ▼
┌──────────────┐
│   L1_Operational │
└──────────────┘


🔧 Componentes Principales

macro_analyzer.py - Análisis de condiciones macroeconómicas
regime_detector.py - Detección de régimen de mercado mediante ML
portfolio_optimizer.py - Optimización de cartera con modelos avanzados
sentiment_analyzer.py - Análisis de sentimiento del mercado
risk_manager.py - Gestión estratégica del riesgo
decision_maker.py - Tomador final de decisiones estratégicas
l3_processor.py - Pipeline de inferencia y consolidación L3 → L2
data_provider.py - Proveedor de datos macro y de mercado
config.py - Configuración de parámetros estratégicos
combine_sentiment.py - Consolida inputs de Twitter, Reddit, News en JSON usable
run_pipeline.py - Orquestador general HRM L3→L1

📊 Flujo de Decisión Estratégica (Pipeline HRM L3→L1)
1. 📈 Recolección de Datos (L3)
   ├─ Indicadores macroeconómicos (GDP, inflación, tasas de interés)
   ├─ Datos de mercado (precios, volúmenes, volatilidad)
   ├─ Datos de sentimiento (redes sociales, noticias)
   └─ Datos de flujos (institucionales, retail)

2. 🧠 Procesamiento y Análisis (L3)
   ├─ Regime Detection (RF/LSTM)
   ├─ Sentiment Analysis (BERT)
   ├─ Volatility Forecasting (GARCH/LSTM)
   ├─ Portfolio Optimization (Black-Litterman)
   ├─ Risk Appetite Calculation (VaR/Expected Shortfall)
   └─ Consolidación de outputs → data/datos_inferencia/

3. 🎯 Toma de Decisiones
   ├─ Definición de régimen de mercado actual
   ├─ Asignación óptima de capital por activo
   ├─ Establecimiento de apetito de riesgo
   ├─ Definición de directrices estratégicas
   └─ Generación de señales para L2

4. 📤 Salida a L2
   ├─ Regime: "bull_market"
   ├─ Asset allocation: {"BTC":0.6,"ETH":0.3,"CASH":0.1}
   ├─ Risk appetite: "moderate"
   └─ Strategic context: {correlation_matrix, volatility_forecast}


# 📊 L3 Strategic Layer – Inference Pipeline

El **Layer 3 (L3) - Strategic** del sistema **HRM** se encarga de consolidar la visión macro y estratégica para alimentar los niveles **L2 (Táctico)** y **L1 (Operacional)**.  
Este pipeline integra modelos de **Regime Detection**, **Volatilidad**, **Sentimiento** y **Optimización de Portafolio (Black-Litterman)**, generando un output jerárquico con los **pesos estratégicos** que guían la toma de decisiones descendente.

---

### ⚡ Ejecución Periódica L3
- L3 se ejecuta en segundo plano cada **10 minutos** (configurable via `L3_UPDATE_INTERVAL`).
- Si L3 tarda más de **30 segundos** (`L3_TIMEOUT`), se usa la última estrategia conocida.
- Esto asegura que L2/L1 siga funcionando sin bloquearse.


## 🚀 Componentes del Pipeline

### 🔹 1. Regime Detection
- Modelo: `models/L3/regime_detection_model_ensemble_optuna.pkl`
- Objetivo: Clasificar el mercado en diferentes regímenes (alcista, bajista, lateral, alta volatilidad, etc.).

### 🔹 2. Sentiment Analysis
- Carpeta modelo: `models/L3/sentiment/`
- Archivos incluidos: `config.json`, `model.safetensors`, `special_tokens_map.json`, `tokenizer_config.json`, `training_args.bin`, `vocab.txt`
- Objetivo: Extraer el sentimiento agregado desde **Reddit, Twitter y News**.

### 🔹 3. Volatility Forecasting
- Modelos disponibles en `models/L3/volatility/`
  - `BTC-USD_volatility_garch.pkl`
  - `BTC-USD_volatility_lstm.h5`
- Objetivo: Proyectar volatilidad futura.  
- **Fallback**: si no hay modelo entrenado para un activo, se utiliza volatilidad histórica.

### 🔹 4. Portfolio Optimization – Black-Litterman
- Carpeta: `models/L3/portfolio/`
  - `bl_cov.csv` (matriz de covarianzas)
  - `bl_weights.csv` (pesos óptimos)
- Objetivo: Integrar inputs anteriores para producir **pesos de asignación estratégica de activos**.

---

## 🔄 Flujo de Inferencia (actualizado)
1. Carga de outputs L3 diarios desde `data/datos_inferencia/`.
2. Fallback automático: si falta algún archivo, se usa última estrategia válida o valores por defecto.
3. Black-Litterman Optimization combina señales de régimen, volatilidad y sentimiento.
4. Output consolidado → `l3_output.json` para uso directo de L2/L1.

---

### 🔗 Integración con L2/L1
- L3 proporciona **estrategia consolidada** (guidelines) a L2 cada ciclo.
- L2 genera señales tácticas usando estrategia L3.
- L1 ejecuta órdenes deterministas validando límites de riesgo.
- El sistema mantiene **loop principal L2/L1 cada 10s**, independiente de L3.

---


🛠️ Estructura de Archivos / Carpetas
project_root/
 ├── models/
 │    └── L3/
 │        ├── regime_detection_model.pkl
 │        ├── sentiment_bert_model/
 │        ├── portfolio/
 │        │    ├── bl_weights.csv
 │        │    └── bl_cov.csv
 │        └── volatility/
 │             ├── BTC-USD_volatility_garch.pkl
 │             └── BTC-USD_volatility_lstm.h5
 ├── data/
 │    ├── datos_para_modelos_l3/     # históricos para entrenamiento
 │    │    ├── sentiment/
 │    │    └── volatility/
 │    └── datos_inferencia/          # outputs recientes L3 → L2
 │         ├── regime_detection.json
 │         ├── sentiment.json
 │         ├── volatility.json
 │         └── portfolio.json
 ├── l3_processor.py                # Pipeline de inferencia consolidado
 ├── combine_sentiment.py           # Combina inputs sociales/noticias
 ├── macro_analyzer.py
 ├── regime_detector.py
 ├── sentiment_analyzer.py
 ├── portfolio_optimizer.py
 ├── risk_manager.py
 ├── decision_maker.py
 ├── data_provider.py
 ├── config.py
 └── run_pipeline.py

🎯 Beneficios del Nuevo Pipeline
- Modularidad y escalabilidad: L3 puede fallar sin interrumpir L2/L1.
- Producción confiable: fallback automático evita bloqueos.
- Logging centralizado: errores y warnings quedan registrados en `core/logging_utils`.

## 🚀 FUNCIONALIDADES IMPLEMENTADAS - L3 COMPLETO

**Componentes operativos en L3:**
- ✅ **Regime Detection** con ensemble ML Optuna entrenado
- ✅ **Portfolio Optimization** usando Black-Litterman con matrices reales
- ✅ **Sentiment Analysis** con BERT pre-entrenado para redes sociales
- ✅ **Volatility Forecasting** con GARCH y LSTM para BTC/ETH
- ✅ **Strategic Decision Making** con pipeline completo L3→L2→L1

**Modelos entrenados disponibles:**
- `regime_detection_model_ensemble_optuna.pkl` - Ensemble Optuna para clasificación de régimen
- `sentiment/` - BERT model completo con tokenizer y configuración
- `volatility/` - GARCH y LSTM models para BTC y ETH
- `portfolio/` - Matrices Black-Litterman (covarianzas y pesos óptimos)

**Integración completa:**
- ✅ L3 ejecuta cada 10 minutos con fallback automático
- ✅ Proporciona directrices estratégicas a L2 en tiempo real
- ✅ L2 genera señales usando contexto estratégico de L3
- ✅ L1 ejecuta órdenes con validación de límites estratégicos
- ✅ Sistema mantiene independencia entre niveles con recuperación automática

## 🎉 Conclusión

L3_Strategic es el **cerebro estratégico completamente operativo** del sistema HRM, combinando:

- ✅ **Análisis macroeconómico avanzado** con datos económicos globales
- ✅ **Modelos de ML sofisticados** (Ensemble Optuna, BERT, GARCH, LSTM, Black-Litterman)
- ✅ **Principios modernos de teoría de portafolio** con optimización Black-Litterman
- ✅ **Integración jerárquica completa** L3→L2→L1 con fallback automático

**Estado actual:** El sistema HRM funciona perfectamente con **L3+L2+L1 en producción completa**, proporcionando trading algorítmico de nivel institucional con análisis estratégico avanzado.

🚀 **Sistema HRM: Arquitectura de 3 niveles completamente implementada y operativa** 🚀
