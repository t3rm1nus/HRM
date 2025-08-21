# L1\_operational - Nivel de Ejecución de Órdenes

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

## 🧪 Pruebas

```bash
cd l1_operational
python test_clean_l1.py
```

Pruebas verifican:

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

# Umbral del filtro de tendencia (Trend AI)
TREND_THRESHOLD = 0.6
```

---

## 🔄 Integración con L2/L3

Se espera que las señales tengan esta estructura:

```python
Signal(
    signal_id="unique_id",
    strategy_id="strategy_name",
    timestamp=1234567890.0,
    symbol="BTC/USDT",
    side="buy",
    qty=0.01,
    order_type="market",
    price=None,
    stop_loss=49000.0,
    risk={"max_slippage_bps": 50},
    metadata={"confidence": 0.9}
)
```

---

## 🧠 Trend AI (Filtro opcional)

Interfaz:

```python
from l1_operational.trend_ai import filter_signal

ok = filter_signal({
    "symbol": "BTC/USDT",
    "timeframe": "5m",
    "price": 50000.0,
    "volume": 123.4,
    "features": {"rsi_trend": 0.7, "macd_trend": 0.65, "price_slope": 0.6}
})  # True si score >= TREND_THRESHOLD
```

Configurable vía `TREND_THRESHOLD` en `config.py`.

Reporte de ejecución devuelto:

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
import asyncio
from comms.message_bus import MessageBus
from l1_operational.bus_adapter import BusAdapterAsync
from l1_operational.order_manager import OrderManager, bus_adapter
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

L1 usa **Loguru** para logging estructurado:

* Nivel INFO para operaciones normales
* Nivel WARNING para rechazos de órdenes
* Nivel ERROR para fallos de ejecución
* Logs incluyen contexto completo de cada operación

---

## 📚 Dataset y Features (BTC/USDT)

Generación con:

```bash
python l1_operational/genera_dataset_modelo1.py --symbol BTC/USDT --output-dir data
```

Salida (CSV):

* `data/btc_1m.csv` (OHLCV crudo)
* `data/btc_features_train.csv` y `data/btc_features_test.csv` (con índice temporal)

Indicadores incluidos (columnas principales):

* trend\_sma\_fast, trend\_sma\_slow
* trend\_ema\_fast, trend\_ema\_slow
* trend\_adx, trend\_macd
* momentum\_rsi, momentum\_stoch, momentum\_stoch\_signal
* volume\_obv
* volatility\_bbw, volatility\_atr

Notas:

* Descarga OHLCV real (endpoint público CCXT) y construye features 1m + 5m.
* Objetivo fijo: \~200k filas de features finales.

---

## 🤖 Entrenamiento de Modelos (ligeros)

Objetivo: clasificar probabilidad de movimiento BTC (up/down) en t+1.

Comandos:

```bash
# Modelo 1: Logistic Regression
python ml_training/train_logreg_modelo1.py

# Modelo 2: Random Forest (L1, capa 2)
python ml_training/train_rf_modelo2_l1.py

# Modelo 3: LightGBM (requiere lightgbm instalado)
python ml_training/train_lgbm_modelo1.py
```

Salida:

* Modelos en `models/` y metadatos `.meta.json` (features usadas, umbral óptimo y métricas).
* Modelo 1 guardado en `models/modelo1_logreg.pkl`

Métricas reportadas:

* Accuracy, F1, AUC. Se calcula además un umbral óptimo por F1 para reducir señales falsas.

---

## 📋 Lista de Features para Entrenamiento del Modelo 2 (L1)

1. trend\_sma\_fast
2. trend\_sma\_slow
3. trend\_ema\_fast
4. trend\_ema\_slow
5. trend\_adx
6. trend\_macd
7. momentum\_rsi
8. momentum\_stoch
9. momentum\_stoch\_signal
10. volume\_obv
11. volatility\_bbw
12. volatility\_atr
13. price\_slope
14. rsi\_trend
15. macd\_trend

> Estos features se usarán para entrenar el segundo modelo de la capa L1 y se integrarán en el flujo determinista junto con el modelo 1.
