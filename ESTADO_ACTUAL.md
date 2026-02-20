# 📋 ESTADO ACTUAL DEL SISTEMA HRM

> **Fecha de verificación:** Febrero 2026  
> **Documento generado:** Corrección de inconsistencias críticas de documentación

---

## ⚠️ DUPLICACIÓN DE SimulatedExchangeClient

### Ficheros involucrados
Existen **tres** ficheros relacionados con el exchange client simulado:

| Fichero | Ruta | Estado |
|---------|------|--------|
| `simulated_exchange_client.py` | `core/simulated_exchange_client.py` | ⚠️ **LEGADO - No usar** |
| `simulated_exchange_client.py` | `l1_operational/simulated_exchange_client.py` | ✅ **ACTIVO** |
| `paper_exchange_adapter.py` | `core/paper_exchange_adapter.py` | ✅ **ACTIVO (alternativa)** |

### ¿Cuál está ACTIVO en main.py?

El fichero **ACTIVO** y usado por el sistema es:

```python
# En main.py y system/bootstrap.py
from l1_operational.simulated_exchange_client import SimulatedExchangeClient
```

**Razón:** El `l1_operational/simulated_exchange_client.py` implementa un patrón Singleton que mantiene el estado entre ciclos de trading, lo cual es crítico para el modo paper trading.

### ¿Por qué existen duplicados?

1. **`core/simulated_exchange_client.py`** - Versión original más compleja con:
   - Soporte para múltiples tipos de órdenes (market, limit, stop_loss, take_profit)
   - Simulación de order book
   - Historial de precios
   - **Estado:** DEPRECATED - mantenido por compatibilidad con tests antiguos

2. **`l1_operational/simulated_exchange_client.py`** - Versión simplificada y activa:
   - Enfoque en paper trading realista
   - Singleton pattern para mantener estado entre ciclos
   - Compatible con `BinanceClient`
   - **Estado:** ACTIVO - usado en producción

3. **`core/paper_exchange_adapter.py`** - Adapter alternativo:
   - Usa Binance testnet para datos reales
   - Simula ejecución de órdenes
   - **Estado:** Disponible pero no usado actualmente en main.py

### Jerarquía de configuración

```
1. l1_operational/simulated_exchange_client.py  ← ACTIVO (main.py)
   └── Usado por: PortfolioManager, OrderManager
   
2. core/paper_exchange_adapter.py  ← ALTERNATIVA
   └── Podría usarse para paper trading con datos reales de testnet
   
3. core/simulated_exchange_client.py  ← LEGADO
   └── Usado solo en: tests/test_simulated_client.py (tests antiguos)
```

### Modo de operación actual verificado

```bash
# El sistema opera actualmente con:
PAPER_MODE=True (forzado en código)
BINANCE_MODE=paper
USE_TESTNET=true
```

**Verificación en main.py:**
```python
# Líneas 47-62 de main.py
if binance_mode == 'live' and paper_mode_env != 'false':
    logger.warning("⚠️ BINANCE_MODE=live detectado pero PAPER_MODE no es 'false' explícito...")
    os.environ['PAPER_MODE'] = 'true'
    os.environ['BINANCE_MODE'] = 'paper'
```

El sistema **SIEMPRE** fuerza el modo paper si no se establece explícitamente `PAPER_MODE=false`, independientemente de las variables de entorno.

---

## 🏗️ ARQUITECTURA REAL DE MÓDULOS

### Ubicación de procesadores L1/L2

Aunque los módulos L1 y L2 deberían estar en sus respectivas carpetas, existen ficheros en `l3_strategy/`:

- `l3_strategy/l1_processor.py` - Procesador de modelos L1 (legacy)
- `l3_strategy/l2_processor.py` - Procesador de señales L2 (legacy)

**Contexto histórico:** Estos ficheros fueron creados durante una refactorización temprana cuando L3 Strategy era el punto central de procesamiento. Aunque ahora existen `core/l3_processor.py` y `l2_tactic/` como implementaciones principales, estos ficheros se mantienen por:

1. **Referencias en scripts de backtesting** antiguos
2. **Compatibilidad con notebooks** de investigación
3. **Documentación histórica** del flujo de procesamiento

**Estado:** No son usados por `main.py` actual, pero no se eliminan para preservar compatibilidad con herramientas de investigación.

---

## 📊 MÓDULOS AUXILIARES

### hacienda/ - Gestión Fiscal
- **Propósito:** Seguimiento fiscal español, cálculo FIFO, informes de ganancias/pérdidas
- **Estado:** Operativo pero **NO** integrado en el ciclo de trading activo
- **Uso:** Post-trading, generación de informes fiscales anuales

### ml_training/ - Entrenamiento de Modelos
- **Propósito:** Scripts de entrenamiento offline para modelos L1, L2 y L3
- **Estado:** Scripts de utilidad para reentrenamiento manual
- **Uso:** Fuera del ciclo de trading, ejecutados bajo demanda

---

## ⚠️ NOTAS IMPORTANTES

1. **PAPER_MODE está forzado a True** en el código de main.py como medida de seguridad
2. Para activar LIVE trading se requiere:
   - Establecer explícitamente `PAPER_MODE=false` en variables de entorno
   - Confirmación manual con espera de 10 segundos
   - Verificación de capital en riesgo

3. **Los duplicados de SimulatedExchangeClient** son un legado de refactorizaciones previas. El sistema usa consistentemente el de `l1_operational/`.

---

*Documento generado como parte de la corrección de inconsistencias críticas - Febrero 2026*
