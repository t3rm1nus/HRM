# L1\_Operational - Nivel de Ejecución de Órdenes

## 🎯 Objetivo

L1 es el nivel de **ejecución y gestión de riesgo en tiempo real**, que combina **IA y reglas hard-coded** para garantizar que solo se ejecuten órdenes seguras. Recibe señales consolidadas de L2/L3 y las ejecuta de forma **determinista**, aplicando validaciones de riesgo, fraccionamiento de órdenes y optimización de ejecución.

---

## 🚫 Lo que L1 NO hace

| ❌ No hace                                                           |
| ------------------------------------------------------------------- |
| No decide estrategias de trading                                    |
| No ajusta precios de señales estratégicas                           |
| No toma decisiones tácticas fuera de seguridad y ejecución          |
| No actualiza portafolio completo (responsabilidad de L2/L3)         |
| No recolecta ni procesa datos de mercado (responsabilidad de L2/L3) |

---

## ✅ Lo que L1 SÍ hace

| ✅ Funcionalidad         | Descripción                                                                               |
| ----------------------- | ----------------------------------------------------------------------------------------- |
| Hard-coded Safety Layer | Bloquea operaciones peligrosas, aplica stop-loss obligatorio y chequeos de liquidez/saldo |
| Trend AI                | Evalúa probabilidad de movimientos del mercado y filtra señales de baja confianza         |
| Execution AI            | Optimiza fraccionamiento de órdenes, timing y reduce slippage                             |
| Risk AI                 | Ajusta tamaño de trade y stops dinámicamente según volatilidad y exposición               |
| Ejecución determinista  | Orden final solo se envía si cumple reglas hard-coded; flujo de 1 intento por señal       |
| Reportes y trazabilidad | Genera reportes detallados de todas las órdenes ejecutadas                                |
| Gestión de errores      | Maneja errores de ejecución de forma robusta                                              |

---

## 🏗️ Arquitectura

```text
L2/L3 (Señales)
      ↓
  Bus Adapter
      ↓
Order Manager
      ↓
[Hard-coded Safety Layer + Trend AI + Execution AI + Risk AI]
      ↓
   Executor → Exchange
      ↓
Execution Report → Bus Adapter → L2/L3
```

### Componentes Principales

* `models.py` - Estructuras de datos (Signal, ExecutionReport, RiskAlert, OrderIntent)
* `bus_adapter.py` - Interfaz asíncrona con el bus de mensajes del sistema (tópicos: `signals`, `reports`, `alerts`)
* `order_manager.py` - Orquesta el flujo de ejecución y validaciones IA/hard-coded
* `risk_guard.py` - Valida límites de riesgo y exposición
* `executor.py` - Ejecuta órdenes en el exchange
* `config.py` - Configuración centralizada de límites y parámetros
* `trend_ai.py` - Filtro de tendencia `filter_signal(signal) -> bool` (umbral `TREND_THRESHOLD`)

---

## 🔒 Validaciones de Riesgo

### Por Operación

* Stop-loss obligatorio (coherente con `side` y `price`)
* Tamaño mínimo/máximo por orden (USDT) y por símbolo (BTC/ETH)
* Límites por símbolo (BTC, ETH, etc.)
* Validación de parámetros básicos

### Por Portafolio

* Exposición máxima por activo
* Drawdown diario máximo
* Saldo mínimo requerido

### Por Ejecución

* Validación de saldo disponible
* Verificación de conexión al exchange
* Timeout de órdenes y reintentos exponenciales

---

## 📊 Flujo de Ejecución (Determinista)

1. **Recepción de Señal** desde L2/L3 vía bus
2. **Validación Hard-coded** (stop-loss, tamaño, liquidez/saldo, exposición, drawdown)
3. **Plan determinista**: se genera un `OrderIntent` 1:1 desde la `Signal`
4. **Ejecución** mediante `executor.py` con timeout/retry y medición de latencia
5. **Generación de `ExecutionReport`** y publicación en el bus (`reports`)

---

## 🔄 Flujo Actualizado de L1 (Nueva Versión)

```text
[Señal L2/L3]
      ↓
[Hard-coded Safety Layer]
      ↓
[Modelo 1 - Logistic Regression] -> Feature 1
      ↓
[Modelo 2 - Random Forest]       -> Feature 2
      ↓
[Modelo 3 - LightGBM/Avanzado]   -> Feature 3
      ↓
[Decision Layer / Risk AI / Execution AI]
      ↓
Executor → Exchange
      ↓
Execution Report → Bus Adapter → L2/L3
```

> Ahora L1 integra múltiples modelos de IA en un flujo jerárquico antes de la ejecución, garantizando seguridad y robustez en decisiones.

---

## 🧪 Pruebas

```bash
cd l1_operational
python test_clean_l1.py
```

Verifican:

* L1 no modifica señales estratégicas
* Validación de riesgo correcta
* Ejecución determinista
* Comportamiento consistente de las IA internas

---

## ⚙️ Configuración

```python
RISK_LIMITS = {
    "MAX_ORDER_SIZE_BTC": 0.05,
    "MAX_ORDER_SIZE_USDT": 1000,
    "MIN_ORDER_SIZE_USDT": 10,
}

PORTFOLIO_LIMITS = {
    "MAX_PORTFOLIO_EXPOSURE_BTC": 0.2,
    "MAX_DAILY_DRAWDOWN": 0.05,
}

TREND_THRESHOLD = 0.6  # Umbral del filtro de tendencia
```

---

## 🚀 Uso

```python
import asyncio
from comms.message_bus import MessageBus
from l1_operational.bus_adapter import BusAdapterAsync
from l1_operational.order_manager import OrderManager
from l1_operational.models import Signal

bus = MessageBus()
bus_adapter = BusAdapterAsync(bus)
om = OrderManager()

async def main():
    # Publicar una señal de ejemplo
    await bus.publish("signals", Signal(
        signal_id="s1", strategy_id="stratA", timestamp=0,
        symbol="BTC/USDT", side="buy", qty=0.01, order_type="market", stop_loss=49000.0
    ).__dict__)

    # Procesar loop
    await asyncio.wait_for(om.handle_signal(await bus_adapter.consume_signal()), timeout=5)

asyncio.run(main())
```

---

## 📈 Métricas

* Órdenes activas
* Reportes pendientes
* Alertas pendientes
* Latencia de ejecución (histograma en memoria)
* Tasa de rechazo / fallas / parciales
* Snapshot de saldos por símbolo tras ejecución

---

## 🎭 Modo de Operación

* **PAPER**: Simulación sin ejecución real (por defecto)
* **LIVE**: Ejecución real en el exchange
* **REPLAY**: Reproducción de datos históricos

---

## 🔍 Logging

* Nivel INFO para operaciones normales
* Nivel WARNING para rechazos de órdenes
* Nivel ERROR para fallos de ejecución
* Logs incluyen contexto completo de cada operación

---

## 🤖 Entrenamiento de Modelos

```bash
# Modelo 1: Logistic Regression
python ml_training/train_logreg_modelo1.py

# Modelo 2: Random Forest (L1, capa 2)
python ml_training/train_rf_modelo2_l1.py

# Modelo 3: LightGBM / Avanzado
python ml_training/train_lgbm_modelo1.py
```

Salida:

* Modelos en `models/` con metadatos `.meta.json`
* Umbral óptimo por F1 para reducir falsas señales

---

## 🧠 Trend AI & Hard-coded Safety Layer

* Hard-coded Safety Layer: stop-loss, chequeos de liquidez, validación de tamaño, reglas deterministas.
* Modelos Trend AI: Logistic Regression, Random Forest, LightGBM/Avanzado.
* Flujo jerárquico garantiza robustez y seguridad antes de enviar órdenes al exchange.
