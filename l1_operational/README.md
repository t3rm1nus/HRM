# L1_operational - Nivel de Ejecución de Órdenes

## 🎯 Objetivo

L1 es el nivel de ejecución de órdenes que **SOLO ejecuta órdenes seguras, sin tomar decisiones estratégicas ni tácticas**. Recibe señales consolidadas de L2/L3 y las ejecuta de forma determinista.

## 🚫 Lo que L1 NO hace

- ❌ **No modifica cantidades** de órdenes
- ❌ **No ajusta precios** de órdenes
- ❌ **No toma decisiones** de timing de ejecución
- ❌ **No actualiza portfolio** (responsabilidad de L2/L3)
- ❌ **No actualiza datos** de mercado (responsabilidad de L2/L3)
- ❌ **No implementa estrategias** de trading
- ❌ **No calcula posiciones** o exposición

## ✅ Lo que L1 SÍ hace

- ✅ **Valida límites de riesgo** antes de ejecutar
- ✅ **Ejecuta órdenes** pre-validadas en el exchange
- ✅ **Genera reportes** de ejecución detallados
- ✅ **Mantiene trazabilidad** completa de todas las operaciones
- ✅ **Aplica validaciones** de seguridad y compliance
- ✅ **Gestiona errores** de ejecución de forma robusta

## 🏗️ Arquitectura

```
L2/L3 (Señales) → Bus Adapter → Order Manager → Risk Guard → Executor → Exchange
                                    ↓
                              Execution Report → Bus Adapter → L2/L3
```

### Componentes Principales

1. **`models.py`** - Estructuras de datos (Signal, ExecutionReport, RiskAlert)
2. **`bus_adapter.py`** - Interfaz con el bus de mensajes del sistema
3. **`order_manager.py`** - Orquesta el proceso completo de ejecución
4. **`risk_guard.py`** - Valida límites de riesgo (sin modificar órdenes)
5. **`executor.py`** - Ejecuta órdenes en el exchange
6. **`config.py`** - Configuración centralizada de límites y parámetros

## 🔒 Validaciones de Riesgo

### Por Operación
- Tamaño mínimo/máximo por orden
- Límites específicos por símbolo (BTC, ETH, etc.)
- Validación de parámetros básicos

### Por Portafolio
- Exposición máxima por activo
- Drawdown diario máximo
- Saldo mínimo requerido

### Por Ejecución
- Validación de saldo disponible
- Verificación de conexión al exchange
- Timeout de órdenes

## 📊 Flujo de Ejecución

1. **Recepción de Señal** desde L2/L3 vía bus
2. **Validación de Riesgo** (sin modificar la señal)
3. **Creación de Orden** basada en la señal original
4. **Ejecución** en el exchange
5. **Generación de Reporte** con métricas completas
6. **Publicación** del reporte vía bus

## 🧪 Pruebas

Ejecuta las pruebas para verificar que L1 está limpio:

```bash
cd l1_operational
python test_clean_l1.py
```

Las pruebas verifican:
- L1 no modifica señales
- L1 solo valida y ejecuta
- Comportamiento determinista
- Validaciones de riesgo correctas

## ⚙️ Configuración

Los límites de riesgo se configuran en `config.py`:

```python
RISK_LIMITS = {
    "MAX_ORDER_SIZE_BTC": 0.05,      # máximo BTC por orden
    "MAX_ORDER_SIZE_USDT": 1000,     # máximo valor en USDT
    "MIN_ORDER_SIZE_USDT": 10,       # mínimo valor en USDT
}

PORTFOLIO_LIMITS = {
    "MAX_PORTFOLIO_EXPOSURE_BTC": 0.2,  # máximo 20% en BTC
    "MAX_DAILY_DRAWDOWN": 0.05,         # máximo 5% DD diario
}
```

## 🔄 Integración con L2/L3

L1 espera señales con esta estructura:

```python
Signal(
    signal_id="unique_id",
    strategy_id="strategy_name",
    timestamp=1234567890.0,
    symbol="BTC/USDT",
    side="buy",
    qty=0.01,
    order_type="market",
    price=None,  # para órdenes market
    risk={"max_slippage_bps": 50},
    metadata={"confidence": 0.9}
)
```

Y retorna reportes de ejecución:

```python
ExecutionReport(
    client_order_id="L1_1234567890_1_abc12345",
    status="filled",
    filled_qty=0.01,
    avg_price=50000.0,
    fees=0.1,
    slippage_bps=5,
    latency_ms=150.5
)
```

## 🚀 Uso

```python
from l1_operational import procesar_l1, get_l1_status

# Procesar órdenes
state = {"ordenes": [...]}
new_state = procesar_l1(state)

# Verificar estado de L1
status = get_l1_status()
print(f"Órdenes activas: {status['active_orders']}")
```

## 📈 Métricas

L1 proporciona métricas operativas:
- Órdenes activas
- Reportes pendientes
- Alertas pendientes
- Latencia de ejecución
- Tasa de rechazo

## 🎭 Modo de Operación

- **PAPER**: Simulación sin ejecución real (por defecto)
- **LIVE**: Ejecución real en el exchange
- **REPLAY**: Reproducción de datos históricos

## 🔍 Logging

L1 usa Loguru para logging estructurado:
- Nivel INFO para operaciones normales
- Nivel WARNING para rechazos de órdenes
- Nivel ERROR para fallos de ejecución
- Logs incluyen contexto completo de cada operación
