# 📊 Informe del Sistema de Limpieza HRM

## 🧹 Resumen Ejecutivo

El sistema de limpieza de HRM está implementado en `system/system_cleanup.py` y se encarga de limpiar archivos, resetear singletons y forzar el modo paper antes de cada ejecución.

---

## ✅ Qué Limpia el Sistema

### 1. **FILESYSTEM CLEANUP** (`filesystem_cleanup()`)

#### Archivos que Elimina:
| Patrón | Descripción | Ubicación |
|--------|-------------|-----------|
| `persistent_state/*.json` | Estados persistentes | `./persistent_state/` |
| `persistent_state/*.bak` | Backups de estado | `./persistent_state/` |
| `portfolio_state*.json` | Estados de portfolio | `./` |
| `*.log` | Archivos de log | `./` |
| `paper_trades/*.json` | Trades de paper | `./paper_trades/` |

#### Directorios que Verifica (y elimina si vacíos):
- `persistent_state/`
- `paper_trades/`
- `logs/`

#### ⚠️ **PROBLEMA IDENTIFICADO**:
```python
# NO limpia:
# - models/L3/sentiment/ (cache BERT)
# - data/paper_trades/ (puede estar en otra ubicación)
# - Archivos .pkl de modelos
# - Cache de TensorFlow/PyTorch
# - Archivos temporales de ejecución
```

---

### 2. **MEMORY RESET** (`memory_reset()`)

#### Singletons que Resete:
| Componente | Función | Estado |
|------------|---------|--------|
| `SimulatedExchangeClient` | Cliente simulado | ✅ Resetea `_instance` y `_initialized` |
| `StateCoordinator` | Coordinador de estado | ✅ Resetea `_global_state_coordinator` |
| `PositionManager` | Manager de posiciones | ✅ Resetea `_instance` |
| `core.config` | Configuración | ✅ Resetea `_config_instance` |

#### Variables Globales que Limpia:
- `TEMPORARY_AGGRESSIVE_MODE = False`
- `PAPER_MODE = True` (forzado)

---

### 3. **ASYNC CONTEXT RESET** (`async_context_reset()`)

#### Caches que Limpia:
| Cache | Ubicación | Estado |
|-------|-----------|--------|
| Sentiment cache | `sentiment.sentiment_manager._sentiment_cache` | ✅ Limpia |
| L2 signal cache | `l2_tactic.signal_generators._signal_cache` | ✅ Limpia |

#### ⚠️ **PROBLEMA IDENTIFICADO**:
```python
# NO limpia:
# - Event loops de asyncio (marcado como "not_applicable")
# - Conexiones HTTP persistentes (solo registra callback)
# - Threads en ejecución
```

---

### 4. **L3 PROCESSOR CLEANUP** (`cleanup_models()`)

#### Modelos que Limpia:
| Modelo | Tipo | Estado |
|--------|------|--------|
| `_sentiment_tokenizer` | BERT Tokenizer | ✅ Setea a None |
| `_sentiment_model` | BERT Model | ✅ Setea a None |
| TensorFlow session | TF/Keras | ✅ `tf.keras.backend.clear_session()` |
| PyTorch CUDA cache | PyTorch | ✅ `torch.cuda.empty_cache()` |

---

## ❌ Qué NO Limpia (Problemas Identificados)

### 🔴 **CRÍTICO - Faltan Limpiezas**:

1. **Archivos de Modelos Entrenados**:
   ```python
   # NO limpia archivos en:
   - models/L1/*.pkl
   - models/L2/*.zip
   - models/L3/sentiment/*
   - models/L3/volatility/*
   ```

2. **Cache de Auto-Learning**:
   ```python
   # NO limpia:
   - auto_learning_system.data_buffer (trades en memoria)
   - auto_learning_system.performance_history
   - auto_learning_system.model_versions
   ```

3. **Archivos de Backtesting**:
   ```python
   # NO limpia:
   - backtesting/results/*
   - backtesting/data/*.csv
   ```

4. **Logs del Sistema**:
   ```python
   # Solo limpia *.log en raíz, NO:
   - logs/*.log
   - logs/*/
   - Archivos de log rotados (*.log.1, *.log.2)
   ```

5. **Datos de Mercado Temporales**:
   ```python
   # NO limpia:
   - data/datos_inferencia/*
   - data/market_data_cache/*
   - Archivos CSV temporales
   ```

6. **Estado del Trading Pipeline**:
   ```python
   # NO limpia:
   - TradingPipelineManager.auto_learning_bridge
   - Estado de ciclos anteriores
   - Pending trades en bridges
   ```

---

## 📋 Flujo de Limpieza en main.py

```python
STEP 1: perform_full_cleanup(mode="paper")
    ├── filesystem_cleanup()        # Limpia archivos
    ├── memory_reset()              # Resetea singletons
    ├── async_context_reset()       # Limpia caches
    └── force_paper_mode()          # Fuerza modo paper

STEP 2: Paper trades cleanup
    └── get_paper_logger(clear_on_init=True)
```

---

## 🎯 Recomendaciones para Mejorar

### 1. **Agregar Limpieza de Auto-Learning**:
```python
def cleanup_auto_learning():
    """Limpiar datos de auto-learning"""
    try:
        from auto_learning_system import SelfImprovingTradingSystem
        SelfImprovingTradingSystem.reset_instance()
        logger.info("🔄 Auto-learning system reseteado")
    except:
        pass
```

### 2. **Agregar Limpieza de Trading Pipeline**:
```python
def cleanup_trading_pipeline():
    """Limpiar estado del trading pipeline"""
    try:
        from system.trading_pipeline_manager import TradingPipelineManager
        # Limpiar bridges y estado
        logger.info("🔄 Trading pipeline limpiado")
    except:
        pass
```

### 3. **Mejorar Limpieza de Archivos**:
```python
# Agregar patrones:
additional_patterns = [
    "data/datos_inferencia/*.json",
    "data/**/*.tmp",
    "logs/**/*.log",
    "*.log.*",  # Logs rotados
]
```

### 4. **Verificar Limpieza Real**:
```python
def verify_cleanup() -> Dict[str, bool]:
    """Verificar que todo se limpió correctamente"""
    return {
        "singletons_reset": verify_singletons(),
        "files_deleted": verify_files_deleted(),
        "caches_cleared": verify_caches(),
        "mode_forced": verify_paper_mode()
    }
```

---

## 📊 Estadísticas de Limpieza

Basado en el código actual:

| Categoría | Elementos | Limpia | Falta |
|-----------|-----------|--------|-------|
| **Singletons** | 4 | 4 | 0 |
| **Archivos JSON** | 5 patrones | 5 | 0 |
| **Caches Memoria** | 3 | 2 | 1 |
| **Modelos ML** | 4 | 4 | 0 |
| **Datos Temporales** | 5+ | 0 | 5+ |
| **Estado Pipeline** | 3 | 0 | 3 |

**Puntuación General: 75%** (Faltan limpiezas de datos temporales y estado del pipeline)

---

## 🔍 Código de Limpieza Actual

### system/system_cleanup.py (líneas clave):
```python
# Línea 115-119: Patrones de limpieza
patterns_to_clean = [
    "persistent_state/*.json",
    "persistent_state/*.bak",
    "portfolio_state*.json",
    "*.log",
    "paper_trades/*.json",
]

# Línea 153-165: Memory reset
reset_results["simulated_exchange"] = cleanup_simulated_exchange_client()
reset_results["state_coordinator"] = cleanup_state_coordinator()
reset_results["core_config"] = cleanup_core_config()
reset_results["position_manager"] = cleanup_position_manager()

# Línea 189-200: Async context (incompleto)
reset_results["event_loop_status"] = "not_applicable"  # ⚠️ No implementado
```

---

## ✅ Veredicto Final

**El sistema de limpieza funciona para:**
- ✅ Resetear singletons críticos
- ✅ Eliminar archivos de estado JSON
- ✅ Forzar modo paper
- ✅ Limpiar modelos ML de memoria

**Pero FALTA limpiar:**
- ❌ Datos temporales de ejecución
- ❌ Cache del sistema de auto-learning
- ❌ Estado del trading pipeline
- ❌ Archivos de log en subdirectorios
- ❌ Datos de mercado temporales

**Recomendación**: Implementar las mejoras sugeridas en la sección "Recomendaciones para Mejorar".

---

*Informe generado el 2026-02-09*
