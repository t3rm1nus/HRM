# 📋 L1_operational - Nivel de Ejecución de Órdenes (LIMPIO Y DETERMINISTA)

## 🎯 Objetivo

L1 es el nivel de ejecución de órdenes que **SOLO ejecuta órdenes seguras, sin tomar decisiones estratégicas ni tácticas**. Recibe señales consolidadas de L2/L3 y las ejecuta de forma determinista y trazable.

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

---

## 🏗️ Nueva Arquitectura Limpia

```
L2/L3 (Señales) → Bus Adapter → Order Manager → Risk Guard → Executor → Exchange
                                    ↓
                              Execution Report → Bus Adapter → L2/L3
```

### Componentes Principales

1. **`models.py`** - Estructuras de datos tipadas (Signal, ExecutionReport, RiskAlert)
2. **`bus_adapter.py`** - Interfaz con el bus de mensajes del sistema
3. **`order_manager.py`** - Orquesta el proceso completo de ejecución
4. **`risk_guard.py`** - Valida límites de riesgo (sin modificar órdenes)
5. **`executor.py`** - Ejecuta órdenes en el exchange
6. **`config.py`** - Configuración centralizada de límites y parámetros

---

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

---

## 📊 Flujo de Ejecución

1. **Recepción de Señal** desde L2/L3 vía bus
2. **Validación de Riesgo** (sin modificar la señal)
3. **Creación de Orden** basada en la señal original
4. **Ejecución** en el exchange
5. **Generación de Reporte** con métricas completas
6. **Publicación** del reporte vía bus

---

## 🧪 Pruebas de Limpieza

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

---

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

---

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

---

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

---

## 📈 Métricas

L1 proporciona métricas operativas:
- Órdenes activas
- Reportes pendientes
- Alertas pendientes
- Latencia de ejecución
- Tasa de rechazo

---

## 🎭 Modo de Operación

- **PAPER**: Simulación sin ejecución real (por defecto)
- **LIVE**: Ejecución real en el exchange
- **REPLAY**: Reproducción de datos históricos

---

## 🔍 Logging

L1 usa Loguru para logging estructurado:
- Nivel INFO para operaciones normales
- Nivel WARNING para rechazos de órdenes
- Nivel ERROR para fallos de ejecución
- Logs incluyen contexto completo de cada operación

---

## 📁 Estructura de Archivos

```
l1_operational/
├── __init__.py              # Interfaz principal de L1
├── models.py                # Estructuras de datos tipadas
├── config.py                # Configuración centralizada
├── bus_adapter.py           # Interfaz con el bus de mensajes
├── order_manager.py         # Gestor principal de órdenes
├── risk_guard.py            # Validaciones de riesgo
├── executor.py              # Ejecutor de órdenes
├── data_feed.py             # Obtención de datos de mercado
├── binance_client.py        # Cliente de Binance
├── test_clean_l1.py         # Pruebas de limpieza
├── README.md                # Documentación específica
└── requirements.txt         # Dependencias
```

---

## 🔄 Flujo de Datos

### Entrada (desde L2/L3)
- Señales de trading con parámetros completos
- Configuración de riesgo y metadata
- Timestamps y identificadores únicos

### Procesamiento Interno
- Validación de límites sin modificación
- Creación de intenciones de orden
- Ejecución en el exchange
- Generación de reportes

### Salida (hacia L2/L3)
- Reportes de ejecución completos
- Alertas de riesgo si aplica
- Métricas de latencia y slippage

---

## 🛡️ Seguridad y Compliance

### Validaciones Obligatorias
- Tamaño de orden dentro de límites
- Saldo disponible suficiente
- Exposición del portafolio controlada
- Drawdown diario bajo control

### Circuit Breakers
- Kill-switch por drawdown excesivo
- Bloqueo por múltiples rechazos
- Validación de conectividad al exchange

---

## 📊 Estado del Sistema

L1 mantiene estado interno para:
- Órdenes activas y su estado
- Reportes pendientes de envío
- Alertas de riesgo generadas
- Métricas de rendimiento

---

## 🔧 Desarrollo y Testing

### Entorno de Desarrollo
- Modo PAPER por defecto
- Sandbox de Binance habilitado
- Logging detallado para debugging

### Pruebas Automatizadas
- Validación de limpieza de L1
- Verificación de comportamiento determinista
- Tests de límites de riesgo
- Simulación de escenarios de error

---

## 📈 Roadmap de Mejoras

### Fase 1: Estabilización
- ✅ Arquitectura limpia implementada
- ✅ Validaciones de riesgo básicas
- ✅ Sistema de reportes funcional

### Fase 2: Robustez
- [ ] Manejo avanzado de errores
- [ ] Métricas de rendimiento detalladas
- [ ] Alertas de riesgo inteligentes

### Fase 3: Escalabilidad
- [ ] Soporte para múltiples exchanges
- [ ] Ejecución de órdenes complejas
- [ ] Integración con sistemas de monitoreo

---

## 🎯 Principios de Diseño

1. **Inmutabilidad**: Las señales nunca se modifican
2. **Determinismo**: Misma entrada = misma salida
3. **Trazabilidad**: Cada operación es completamente rastreable
4. **Seguridad**: Validaciones estrictas antes de cualquier ejecución
5. **Simplicidad**: L1 solo hace una cosa y la hace bien

---

## 📞 Soporte y Mantenimiento

- **Logs**: Todos los eventos se registran con contexto completo
- **Métricas**: Monitoreo en tiempo real del rendimiento
- **Alertas**: Notificaciones automáticas de problemas
- **Documentación**: Código auto-documentado y README detallado

---

✍️ **Autor:** Equipo de desarrollo HRM  
📌 **Versión:** 2.0 (Limpia y Determinista)  
📅 **Última actualización:** 2025  
🔒 **Estado:** L1 completamente limpio y validado
