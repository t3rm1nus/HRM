# 🧠 Listado de las 9 IAs (Capas Anti-Overfitting) y Archivos de Modelos

## Las 9 IAs del Sistema de Auto-Learning

Estas son las 9 capas de protección anti-overfitting implementadas en `auto_learning_system.py`:

| # | IA / Capa | Tipo | Función | Configuración |
|---|-----------|------|---------|---------------|
| 1️⃣ | **AntiOverfitValidator** | Clase | Validación cruzada continua | 5 ventanas, min score 0.55 |
| 2️⃣ | **AdaptiveRegularizer** | Clase | Regularización adaptativa | L2: 0.01, Dropout: 0.20 |
| 3️⃣ | **DiverseEnsembleBuilder** | Clase | Ensemble diverso | Max 10 modelos, sim threshold 0.85 |
| 4️⃣ | **ConceptDriftDetector** | Clase | Detección de drift | Threshold: 0.10 (Jensen-Shannon) |
| 5️⃣ | **SmartEarlyStopper** | Clase | Early stopping inteligente | Patience: 15 epochs |
| 6️⃣ | **TimeBasedTrigger** | Dict | Trigger por tiempo | 168h (7 días) |
| 7️⃣ | **PerformanceBasedTrigger** | Dict | Trigger por performance | Win rate < 52%, Drawdown > 12% |
| 8️⃣ | **RegimeChangeTrigger** | Dict | Trigger por cambio de régimen | 3 cambios detectados |
| 9️⃣ | **DataVolumeTrigger** | Dict | Trigger por volumen | 500+ trades |

---

## 📁 Archivos de Modelos por Capa

### **L1 - Modelos Operacionales** (`models/L1/`)
Cargados desde `auto_learning_system.py` -> `_load_base_models()`

```
models/L1/
├── modelo1_lr.pkl                    # Logistic Regression
├── modelo2_rf.pkl                    # Random Forest
├── modelo3_lgbm.pkl                  # LightGBM
├── modelo3_lgbm.meta.json            # Metadatos LGBM
├── metadata.json                     # Configuración general
├── base_model.py                     # Clase base
├── ensemble_model.py                 # Modelo ensemble
├── lightgbm_model.py                 # Wrapper LightGBM
├── logistic_regression_model.py      # Wrapper LR
└── random_forest_model.py            # Wrapper RF
```

**Clases en código:**
- `BaseL1Model` (base)
- `MomentumModel` - Tendencias corto/medio plazo
- `TechnicalIndicatorsModel` - RSI, MACD, Bollinger
- `VolumeSignalsModel` - Flujos de capital
- `L1Model` - Modelo principal combinado

---

### **L2 - Modelos de IA Táctica** (`models/L2/`)
Cargados desde `l2_tactic/model_loaders.py` -> `load_model_by_type()`

```
models/L2/
├── claude.zip                        # Modelo Claude (Anthropic)
├── deepseek.zip                      # Modelo DeepSeek v1
├── deepseek2.zip                     # Modelo DeepSeek v2
├── gemini.zip                        # Modelo Gemini (Google)
├── gpt.zip                           # Modelo GPT (OpenAI)
├── grok.zip                          # Modelo Grok (xAI)
├── kimi.zip                          # Modelo Kimi (Moonshot)
├── deepseek.py                       # Wrapper DeepSeek
└── wrapper_deepseek.py               # Wrapper mejorado
```

**Métodos de carga en `ModelLoaders`:**
- `load_deepseek_model()` - Carga DeepSeek con wrapper
- `load_claude_model()` - Carga Claude
- `load_kimi_model()` - Carga Kimi
- `load_gpt_model()` - Carga GPT
- `load_stable_baselines3_model()` - Carga SB3/PPO

---

### **L3 - Modelos Estratégicos** (`models/L3/`)
Cargados desde `l3_strategy/l3_processor.py`

```
models/L3/
├── regime_detection_model_ensemble_optuna.pkl    # Detector de régimen (ensemble)
│
├── sentiment/                                     # Modelo BERT de sentimiento
│   ├── model.safetensors
│   ├── tokenizer_config.json
│   ├── vocab.txt
│   ├── special_tokens_map.json
│   └── training_args.bin
│
├── volatility/                                    # Modelos de volatilidad
│   ├── BTC-USD_volatility_garch.pkl              # GARCH BTC
│   ├── BTC-USD_volatility_lstm.h5                # LSTM BTC
│   ├── ETH-USD_volatility_garch.pkl              # GARCH ETH
│   └── ETH-USD_volatility_lstm.h5                # LSTM ETH
│
├── portfolio/                                     # Black-Litterman
│   ├── bl_cov.csv                                # Matriz de covarianza
│   └── bl_weights.csv                            # Pesos óptimos
│
└── regime/                                        # Modelos por régimen
    # (cargados dinámicamente según el régimen detectado)
```

**Clases en código (`l3_strategy/regime_specific_models.py`):**
- `BullMarketModel` - Régimen alcista
- `BearMarketModel` - Régimen bajista
- `RangeMarketModel` - Régimen lateral
- `VolatileMarketModel` - Régimen volátil
- `CrisisMarketModel` - Régimen de crisis

---

## 🔧 Pipeline de Carga de Modelos

### Flujo de Inicialización:

1. **`auto_learning_system.py`** inicia:
   ```python
   AutoRetrainingSystem.__init__()
   └── Carga las 9 IAs de protección
   └── Llama a _load_base_models()
       └── models/L1/modelo*.pkl
   ```

2. **`l2_tactic/tactical_signal_processor.py`** carga:
   ```python
   L2TacticProcessor usa ModelLoaders
   └── Carga modelos desde models/L2/*.zip
   └── Claude, DeepSeek, GPT, Grok, Kimi, Gemini
   ```

3. **`l3_strategy/l3_processor.py`** carga:
   ```python
   load_regime_model() -> regime_detection_model_ensemble_optuna.pkl
   load_sentiment_model() -> models/L3/sentiment/ (BERT)
   load_vol_models() -> models/L3/volatility/* (GARCH/LSTM)
   ```

---

## 📊 Resumen Total de Modelos

| Capa | # Modelos | Archivos Principales |
|------|-----------|---------------------|
| L1 | 3 modelos + ensemble | modelo1_lr.pkl, modelo2_rf.pkl, modelo3_lgbm.pkl |
| L2 | 7 modelos de IA | claude.zip, deepseek.zip, gpt.zip, grok.zip, kimi.zip, gemini.zip, deepseek2.zip |
| L3 | 5+ modelos | Regime, Sentiment (BERT), Volatility (GARCH/LSTM), Black-Litterman |
| **Total** | **15+ modelos** | Más modelos ensemble y especializados |

---

## 🎯 Uso en Auto-Learning

Las **9 IAs** protegen el reentrenamiento de todos estos modelos:

```python
# Cuando se acumulan suficientes trades (500+):
AutoRetrainingSystem._auto_retrain_models()
├── 1. Prepara datos de entrenamiento
├── 2. ConceptDriftDetector.detect_drift()  # Capa 4
├── 3. Para cada modelo candidato:
│   ├── AntiOverfitValidator.validate_new_model()  # Capa 1
│   ├── AdaptiveRegularizer.adjust_regularization()  # Capa 2
│   ├── DiverseEnsembleBuilder.add_model_to_ensemble()  # Capa 3
│   └── SmartEarlyStopper.should_stop_training()  # Capa 5
└── 4. Despliega solo si pasa todas las capas
```

---

*Generado automáticamente el 2026-02-09*
