# Backtesting L1 Standalone

Script independiente para backtesting de modelos L1 (`l1_operational/`) completamente aislado de L2 y L3.

## Características

- ✅ **L1 Aislado**: NO importa nada de `l2_tactic/` ni `l3_strategy/`
- ✅ Carga datos OHLCV desde CSV locales o Binance testnet
- ✅ Usa modelos L1 reales: `L1Model`, `AIModelPipeline`, `filter_signal` (TrendAI)
- ✅ Simulación de ejecución con `SimulatedExchangeClient`
- ✅ Métricas completas: win rate, profit factor, max drawdown, Sharpe ratio
- ✅ Exporta CSV con timestamp, signal, confidence, pnl_trade, cumulative_pnl

## Uso

### Básico (valores por defecto)

```bash
python backtesting/backtest_l1_standalone.py
```

Esto ejecuta un backtest con:
- Símbolos: BTCUSDT, ETHUSDT
- Timeframe: 1m
- Días: 30
- Balance inicial: $3000 USDT
- Tamaño de posición: 10% del capital
- Confianza mínima: 0.6

### Parámetros personalizados

```bash
# Solo BTC, 60 días, balance de $5000
python backtesting/backtest_l1_standalone.py --symbols BTCUSDT --days 60 --balance 5000

# Sin usar caché (descarga siempre de Binance)
python backtesting/backtest_l1_standalone.py --no-cache --days 90

# Mayor tamaño de posición y confianza mínima más baja
python backtesting/backtest_l1_standalone.py --position-size 0.2 --min-confidence 0.55

# Output personalizado
python backtesting/backtest_l1_standalone.py --output-dir ./mis_resultados
```

### Todos los parámetros

```
--symbols        Símbolos a testear (default: BTCUSDT ETHUSDT)
--timeframe      Timeframe de las velas (default: 1m)
--days           Días de datos históricos (default: 30)
--balance        Balance inicial en USDT (default: 3000)
--position-size  Porcentaje del capital por operación (default: 0.1)
--min-confidence Confianza mínima para operar (default: 0.6)
--no-cache       No usar datos en caché, descargar siempre
--output-dir     Directorio de salida (default: backtesting/results/)
```

## Estructura de Archivos Generados

```
backtesting/results/
├── l1_backtest_signals_YYYYMMDD_HHMMSS.csv   # Historial de señales
├── l1_backtest_trades_YYYYMMDD_HHMMSS.csv    # Historial de trades
├── l1_backtest_metrics_YYYYMMDD_HHMMSS.json  # Métricas en JSON
└── l1_backtest_summary_YYYYMMDD_HHMMSS.json  # Resumen completo
```

### Formato CSV de Señales

| Columna | Descripción |
|---------|-------------|
| timestamp | Fecha y hora de la señal |
| symbol | Par de trading (BTCUSDT, ETHUSDT) |
| price | Precio de cierre de la vela |
| signal | Señal generada (BUY, SELL, HOLD) |
| confidence | Nivel de confianza (0.0 - 1.0) |
| action_taken | Si se ejecutó la orden (true/false) |
| cumulative_pnl | PnL acumulado hasta ese momento |

### Formato CSV de Trades

| Columna | Descripción |
|---------|-------------|
| trade_id | ID único del trade |
| timestamp | Fecha de entrada |
| symbol | Par de trading |
| side | Lado (BUY/SELL) |
| entry_price | Precio de entrada |
| exit_price | Precio de salida |
| quantity | Cantidad operada |
| confidence | Confianza de la señal |
| pnl | PnL en USDT |
| pnl_pct | PnL en porcentaje |
| hold_time_minutes | Tiempo de holding |

## Métricas Calculadas

- **Total Trades**: Número total de operaciones
- **Win Rate**: Porcentaje de trades ganadores
- **Profit Factor**: Ganancias brutas / Pérdidas brutas
- **Max Drawdown**: Pérdida máxima desde el peak
- **Sharpe Ratio**: Rendimiento ajustado por riesgo
- **Avg Hold Time**: Tiempo promedio de holding
- **Return %**: Retorno total del período

## Datos Históricos

### Carga desde CSV Local

El script busca archivos CSV en `backtesting/data/` con estos nombres:
- `{symbol}_1m.csv` (ej: `btcusdt_1m.csv`)
- `{symbol}_features_train.csv`
- `{symbol}_features_test.csv`

### Descarga desde Binance

Si no hay datos locales, descarga automáticamente desde Binance testnet usando CCXT.

Los datos descargados se guardan en caché para futuras ejecuciones.

## Modelos L1 Utilizados

1. **L1Model**: Modelo principal que combina:
   - MomentumModel (momentum corto/medio plazo)
   - TechnicalIndicatorsModel (RSI, MACD, Bollinger)
   - VolumeSignalsModel (flujos de volumen, liquidez)

2. **AIModelPipeline**: Pipeline de IA con ensemble de modelos ML

3. **TrendAI (filter_signal)**: Validación de tendencia con fallback a reglas técnicas

## Ejemplo de Salida

```
================================================================================
🚀 INICIANDO BACKTEST L1 STANDALONE
================================================================================
Símbolos: ['BTCUSDT', 'ETHUSDT']
Timeframe: 1m
Días: 30
Balance inicial: $3000.00
Tamaño posición: 10.0%
Confianza mínima: 0.60

📊 Cargando datos históricos...
📁 Cargando datos desde backtesting/data/btcusdt_1m.csv
📁 Cargando datos desde backtesting/data/ethusdt_1m.csv
✅ Datos cargados: ['BTCUSDT', 'ETHUSDT']
  - BTCUSDT: 43200 velas
  - ETHUSDT: 43200 velas

⏳ Procesando velas...
  Progreso: 5.0% (2160/43200)
  ...
  Progreso: 100.0% (43200/43200)

🔚 Cerrando posiciones abiertas...
🔚 Cierre final BTCUSDT @ 64235.50 PnL=125.30

📈 Calculando métricas...

================================================================================
📊 RESULTADOS DEL BACKTEST
================================================================================
  total_trades              : 45
  winning_trades            : 28
  losing_trades             : 17
  win_rate                  : 62.22%
  profit_factor             : 1.85
  max_drawdown              : 234.50
  max_drawdown_pct          : 7.45%
  sharpe_ratio              : 1.42
  total_pnl                 : 456.78
  avg_hold_time_minutes     : 145.3
  avg_trade_pnl             : 10.15
  best_trade                : 89.45
  worst_trade               : -34.20
  start_balance             : 3000.00
  end_balance               : 3456.78
  return_pct                : 15.23%
```

## Notas

- El script usa `SimulatedExchangeClient` de `l1_operational/` para simular ejecución real
- Se aplica slippage del 0.05% y comisiones del 0.1%
- Las posiciones se cierran automáticamente al final del backtest
- El TrendAI puede filtrar señales si la tendencia no es favorable
