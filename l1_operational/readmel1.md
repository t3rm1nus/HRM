# L1_Operational - Nivel de Ejecución de Órdenes (Actualizado)

## 🎯 Objetivo

L1 es el nivel de **ejecución y gestión de riesgo en tiempo real**, que combina **IA multiasset y reglas hard-coded** para garantizar que solo se ejecuten órdenes seguras. Recibe señales consolidadas de L2/L3 y las ejecuta de forma **determinista**, aplicando validaciones de riesgo, fraccionamiento de órdenes y optimización de ejecución **para múltiples activos (BTC, ETH)**.

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
| **Hard-coded Safety Layer** | Bloquea operaciones peligrosas, aplica stop-loss obligatorio y chequeos de liquidez/saldo |
| **Multiasset Trend AI**     | Evalúa probabilidad de movimientos para **BTC y ETH**, filtra señales de baja confianza |
| **Execution AI**            | Optimiza fraccionamiento de órdenes, timing y reduce slippage por símbolo              |
| **Risk AI**                 | Ajusta tamaño de trade y stops dinámicamente según volatilidad y exposición por activo |
| **Ejecución determinista**  | Orden final solo se envía si cumple reglas hard-coded; flujo de 1 intento por señal    |
| **Reportes multiasset**     | Genera reportes detallados de todas las órdenes ejecutadas por símbolo                 |
| **Gestión de errores**      | Maneja errores de ejecución de forma robusta                                           |

---

## 🗏️ Arquitectura Actualizada

```text
L2/L3 (Señales BTC/ETH)
          ↓
    Bus Adapter
          ↓
  Order Manager
          ↓
[Hard-coded Safety Layer]
          ↓
[Modelo 1: LogReg] → Feature 1 (BTC/ETH)
          ↓
[Modelo 2: Random Forest] → Feature 2 (BTC/ETH)
          ↓
[Modelo 3: LightGBM] → Feature 3 (BTC/ETH)
          ↓
[Decision Layer: Risk AI + Execution AI]
          ↓
   Executor → Exchange
          ↓
Execution Report → Bus Adapter → L2/L3
```

### Componentes Principales

* `models.py` - Estructuras de datos (Signal, ExecutionReport, RiskAlert, OrderIntent)
* `bus_adapter.py` - Interfaz asíncrona con el bus de mensajes del sistema (tópicos: `signals`, `reports`, `alerts`)
* `order_manager.py` - Orquesta el flujo de ejecución y validaciones IA/hard-coded **multiasset**
* `risk_guard.py` - Valida límites de riesgo y exposición **por símbolo**
* `executor.py` - Ejecuta órdenes en el exchange
* `config.py` - Configuración centralizada de límites y parámetros **por activo**
* **`ai_models/`** - Modelos IA multiasset entrenados:
  - `modelo1_logreg_multiasset.pkl` - Logistic Regression (BTC/ETH)
  - `modelo1_rf_multiasset.pkl` - Random Forest (BTC/ETH) 
  - `modelo1_lgbm_multiasset.pkl` - LightGBM (BTC/ETH)

---

## 🔑 Validaciones de Riesgo (Multiasset)

### Por Operación
* Stop-loss obligatorio (coherente con `side` y `price`)
* Tamaño mínimo/máximo por orden (USDT) **y por símbolo específico**
* Límites por símbolo (BTC: 0.05 BTC max, ETH: 2.0 ETH max)
* Validación de parámetros básicos

### Por Portafolio
* **Exposición máxima por activo**: BTC (20%), ETH (15%)
* **Drawdown diario máximo por símbolo**
* **Saldo mínimo requerido por par** (BTC/USDT, ETH/USDT)
* **Correlación BTC-ETH**: Límites de exposición cruzada

### Por Ejecución
* Validación de saldo disponible **por base asset**
* Verificación de conexión al exchange
* Timeout de órdenes y reintentos exponenciales
* **Slippage protection por símbolo**

---

## 📊 Flujo de Ejecución (Determinista Multiasset)

1. **Recepción de Señal** desde L2/L3 vía bus (BTC/USDT o ETH/USDT)
2. **Validación Hard-coded** por símbolo (stop-loss, tamaño, liquidez/saldo, exposición, drawdown)
3. **Filtros IA multiasset**:
   - LogReg: Probabilidad de tendencia (threshold específico por asset)
   - Random Forest: Confirmación con features cruzadas BTC-ETH
   - LightGBM: Decisión final con regularización avanzada
4. **Plan determinista**: se genera un `OrderIntent` 1:1 desde la `Signal`
5. **Ejecución** mediante `executor.py` con timeout/retry y medición de latencia
6. **Generación de `ExecutionReport`** y publicación en el bus (`reports`)

---

## 🤖 Modelos IA Multiasset (Nuevos)

### Modelo 1: Logistic Regression
```python
# Carga: joblib.load('models/modelo1_logreg_multiasset.pkl')
# Input: Features + is_btc/is_eth + eth_btc_ratio + correlación
# Output: Probabilidad [0-1] de movimiento alcista
# Threshold: Específico por símbolo (BTC: ~0.62, ETH: ~0.58)
```

### Modelo 2: Random Forest  
```python
# Carga: joblib.load('models/modelo1_rf_multiasset.pkl')
# Árboles: 500, max_depth: None, class_weight: balanced_subsample
# Features: Incluye features cruzadas BTC-ETH
# Output: Probabilidad con feature importance por símbolo
```

### Modelo 3: LightGBM (Avanzado)
```python
# Carga: joblib.load('models/modelo1_lgbm_multiasset.pkl')
# Predicción: model.predict(X, num_iteration=model.best_iteration)
# Parámetros: 64 hojas, L1/L2 regularization, early stopping
# Features: Máxima importancia con gain-based ranking
```

---

## 📈 Métricas Multiasset

### Globales
* Órdenes activas **por símbolo**
* Reportes pendientes
* Alertas pendientes
* Latencia de ejecución (histograma en memoria) **por exchange pair**

### Por Símbolo
* **BTC/USDT**: Tasa éxito, slippage promedio, volumen ejecutado
* **ETH/USDT**: Tasa éxito, slippage promedio, volumen ejecutado
* **Correlación**: Exposición cruzada BTC-ETH en tiempo real
* **Snapshot de saldos**: BTC, ETH, USDT tras cada ejecución

---

## ⚙️ Configuración Multiasset

```python
RISK_LIMITS = {
    # Límites por símbolo
    "MAX_ORDER_SIZE_BTC": 0.05,      # 0.05 BTC máximo por orden
    "MAX_ORDER_SIZE_ETH": 2.0,       # 2.0 ETH máximo por orden
    "MAX_ORDER_SIZE_USDT": 1000,     # $1000 USDT máximo
    "MIN_ORDER_SIZE_USDT": 10,       # $10 USDT mínimo
    
    # Límites específicos
    "BTC_MIN_SIZE": 0.0001,          # Mínimo técnico BTC
    "ETH_MIN_SIZE": 0.001,           # Mínimo técnico ETH
}

PORTFOLIO_LIMITS = {
    # Exposición máxima por activo
    "MAX_PORTFOLIO_EXPOSURE_BTC": 0.20,  # 20% en BTC
    "MAX_PORTFOLIO_EXPOSURE_ETH": 0.15,  # 15% en ETH
    "MAX_DAILY_DRAWDOWN": 0.05,          # 5% drawdown global
    
    # Límites cruzados
    "MAX_CRYPTO_EXPOSURE": 0.30,         # 30% total cripto
    "MIN_USDT_BALANCE": 100,             # $100 USDT siempre
}

# Thresholds IA por modelo y símbolo
AI_THRESHOLDS = {
    "LOGREG": {"BTC": 0.620, "ETH": 0.580},
    "RF": {"BTC": 0.630, "ETH": 0.590},  
    "LGBM": {"BTC": 0.625, "ETH": 0.585}
}
```

---

## 🚀 Uso Multiasset

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
    # Señal BTC
    await bus.publish("signals", Signal(
        signal_id="btc_s1", strategy_id="stratA", timestamp=0,
        symbol="BTC/USDT", side="buy", qty=0.01, 
        order_type="market", stop_loss=49000.0
    ).__dict__)
    
    # Señal ETH  
    await bus.publish("signals", Signal(
        signal_id="eth_s1", strategy_id="stratA", timestamp=0,
        symbol="ETH/USDT", side="buy", qty=0.5,
        order_type="market", stop_loss=2900.0
    ).__dict__)

    # Procesar ambas señales
    while True:
        signal = await bus_adapter.consume_signal()
        if signal:
            await om.handle_signal(signal)

asyncio.run(main())
```

---

## 🧪 Pruebas Multiasset

```bash
cd l1_operational
python test_clean_l1_multiasset.py
```

Verifican:
* L1 no modifica señales estratégicas **de ningún símbolo**
* Validación de riesgo correcta **por BTC y ETH**
* Ejecución determinista **multiasset**
* Comportamiento consistente de las **3 IA internas con features cruzadas**
* **Límites de correlación BTC-ETH**

---

## 🎭 Modo de Operación

* **PAPER**: Simulación sin ejecución real (por defecto) - **soporta BTC/ETH**
* **LIVE**: Ejecución real en el exchange - **binance BTC/USDT, ETH/USDT**
* **REPLAY**: Reproducción de datos históricos - **datasets multiasset**

---

## 📝 Logging Multiasset

* Nivel INFO para operaciones normales **con etiqueta [BTC] o [ETH]**
* Nivel WARNING para rechazos de órdenes **por símbolo específico**
* Nivel ERROR para fallos de ejecución **con contexto de asset**
* Logs incluyen **contexto completo por símbolo y correlaciones**

---

## 🤖 Entrenamiento de Modelos Multiasset

```bash
# Modelo 1: Logistic Regression (BTC + ETH)
python ml_training/modelo1_train_logreg_multiasset.py

# Modelo 2: Random Forest (BTC + ETH)  
python ml_training/modelo1_train_rf_multiasset.py

# Modelo 3: LightGBM (BTC + ETH)
python ml_training/modelo1_train_lgbm_multiasset.py
```

**Salida por modelo**:
* `models/modelo1_[tipo]_multiasset.pkl` - Modelo entrenado
* `models/modelo1_[tipo]_multiasset.meta.json` - Metadatos con métricas **por símbolo**
* **Threshold óptimo separado para BTC y ETH**
* **Feature importance con correlaciones cruzadas**

---

## 🧠 Sistema IA Jerárquico (Multiasset)

### Flujo de Decisión:
1. **Hard-coded Safety**: Validaciones básicas por símbolo
2. **LogReg**: Filtro rápido de tendencia (BTC/ETH específico)  
3. **Random Forest**: Confirmación con ensemble robusto
4. **LightGBM**: Decisión final con regularización avanzada
5. **Decision Layer**: Combinación ponderada de los 3 modelos

### Features Multiasset:
* **Por símbolo**: RSI, MACD, Bollinger, volumen, etc.
* **Cruzadas**: ETH/BTC ratio, correlación rolling, divergencias
* **Encoding**: is_btc, is_eth para diferenciación
* **Temporales**: Features específicas por timeframe de cada asset

---

## 📊 Dashboard de Métricas (Multiasset)

```
🎯 L1 OPERATIONAL DASHBOARD
├── BTC/USDT
│   ├── Señales procesadas: 45 ✅ | 3 ❌
│   ├── Success rate: 93.8%
│   ├── Slippage promedio: 0.12%
│   └── Exposición actual: 18.5% / 20% max
├── ETH/USDT  
│   ├── Señales procesadas: 32 ✅ | 2 ❌
│   ├── Success rate: 94.1%
│   ├── Slippage promedio: 0.15%
│   └── Exposición actual: 12.3% / 15% max
└── Correlación BTC-ETH: 0.73 (límite: 0.80)
```

---

## 🔄 Integración con Capas Superiores

### L2/L3 → L1 (Input esperado):
```json
{
  "signal_id": "btc_signal_123",
  "symbol": "BTC/USDT",        // O "ETH/USDT"
  "side": "buy",
  "qty": 0.01,                 // Respetando límites por símbolo
  "stop_loss": 49000.0,
  "strategy_context": {
    "regime": "bull_market",
    "correlation_btc_eth": 0.65
  }
}
```

### L1 → L2/L3 (Output generado):
```json
{
  "execution_id": "exec_456", 
  "signal_id": "btc_signal_123",
  "symbol": "BTC/USDT",
  "status": "filled",
  "executed_qty": 0.01,
  "avg_price": 50125.30,
  "slippage": 0.11,
  "ai_scores": {
    "logreg": 0.745,
    "random_forest": 0.821, 
    "lightgbm": 0.798
  },
  "risk_metrics": {
    "portfolio_exposure_btc": 0.185,
    "correlation_impact": 0.023
  }
}
```

---

## ✨ Novedades de la Versión Multiasset

### 🆕 Nuevas características:
- ✅ **Soporte nativo BTC + ETH** en todos los componentes
- ✅ **3 modelos IA entrenados** con features cruzadas
- ✅ **Thresholds optimizados** por F1-score específicos por símbolo  
- ✅ **Gestión de riesgo avanzada** con límites de correlación
- ✅ **Métricas granulares** por activo y globales
- ✅ **Configuración flexible** para añadir más assets

### 🔧 Componentes actualizados:
- `order_manager.py` → Flujo multiasset con 3 IA
- `risk_guard.py` → Límites específicos por símbolo
- `config.py` → Configuración granular BTC/ETH
- `ai_models/` → Modelos entrenados listos para producción

### 📈 Rendimiento esperado:
- **BTC**: Accuracy ~66%, F1 ~64%, AUC ~72%
- **ETH**: Accuracy ~65%, F1 ~61%, AUC ~70%  
- **Latencia**: <50ms por señal (incluyendo 3 modelos IA)
- **Throughput**: >100 señales/segundo

---

## 🎉 Conclusión

**L1 está ahora completamente preparado para operar con múltiples activos**, combinando la robustez de reglas deterministas con la inteligencia de 3 modelos IA especializados en BTC y ETH. El sistema garantiza ejecución segura, eficiente y optimizada para cada símbolo mientras mantiene control de riesgo a nivel de portafolio.

**¿Listo para el trading multiasset inteligente? 🚀**