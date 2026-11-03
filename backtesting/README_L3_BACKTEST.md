# Backtesting L3 Regime Classifier

Script independiente para backtesting y validación del clasificador de régimen de mercado L3 (`l3_strategy/regime_classifier.py` y `l3_strategy/range_detector.py`).

## Características

- ✅ **L3 Aislado**: NO importa nada de `l1_operational/` ni `l2_tactic/`
- ✅ Carga datos OHLCV desde CSV locales o Binance (mínimo 1 año de datos)
- ✅ Usa clasificadores L3 reales: `MarketRegimeClassifier` y `range_detector`
- ✅ Labels manuales de validación para eventos históricos conocidos (2022-2024)
- ✅ Matriz de confusión del clasificador
- ✅ Medición de lag: ¿cuántas velas tarda en detectar cambios de régimen?
- ✅ Comparación de trading: CON L3 vs SIN L3 → ¿L3 añade alpha real?
- ✅ Exporta CSV con todas las clasificaciones y reporte detallado

## Uso

### Básico (valores por defecto)

```bash
python backtesting/backtest_l3_regime.py
```

Esto ejecuta un backtest con:
- Símbolos: BTCUSDT, ETHUSDT
- Timeframe: 1h (más estable para análisis de régimen)
- Días: 730 (2 años)
- Ventana de análisis: 168 velas (7 días)
- Intervalo de clasificación: 6 horas

### Parámetros personalizados

```bash
# Solo BTC, 1 año de datos
python backtesting/backtest_l3_regime.py --symbols BTCUSDT --days 365

# Timeframe diferente y más días
python backtesting/backtest_l3_regime.py --timeframe 4h --days 500

# Sin usar caché (descarga siempre de Binance)
python backtesting/backtest_l3_regime.py --no-cache --days 730

# Output personalizado
python backtesting/backtest_l3_regime.py --output-dir ./mis_resultados
```

### Todos los parámetros

```
--symbols        Símbolos a testear (default: BTCUSDT ETHUSDT)
--timeframe      Timeframe de las velas (default: 1h)
--days           Días de datos históricos (default: 730, mínimo: 365)
--no-cache       No usar datos en caché, descargar siempre
--output-dir     Directorio de salida (default: backtesting/results/)
```

## Estructura de Archivos Generados

```
backtesting/results/
├── l3_regime_backtest_report_YYYYMMDD_HHMMSS.txt    # Reporte detallado
├── l3_regime_classifications_YYYYMMDD_HHMMSS.csv    # Clasificaciones
└── l3_regime_backtest_results_YYYYMMDD_HHMMSS.json  # Resultados en JSON
```

### Formato CSV de Clasificaciones

| Columna | Descripción |
|---------|-------------|
| timestamp | Fecha y hora de la clasificación |
| symbol | Par de trading (BTCUSDT, ETHUSDT) |
| primary_regime | Régimen primario (TRENDING, RANGE, VOLATILE, BREAKOUT) |
| subtype | Subtipo (STRONG_BULL, TIGHT_RANGE, etc.) |
| simplified_regime | Régimen simplificado (bull, bear, sideways, volatile) |
| confidence | Nivel de confianza (0.0 - 1.0) |
| price | Precio de cierre |
| price_change_24h | Cambio de precio 24h |
| price_change_7d | Cambio de precio 7d |
| volatility | Volatilidad calculada |

## Métricas Calculadas

### 1. Matriz de Confusión
- **Accuracy**: Precisión global del clasificador
- **Precision**: TP / (TP + FP) por régimen
- **Recall**: TP / (TP + FN) por régimen
- **F1 Score**: Media armónica de Precision y Recall

### 2. Análisis de Lag
- **Total Changes**: Cambios de régimen reales detectados
- **Detected Changes**: Cuántos fueron detectados por el clasificador
- **Avg Lag**: Velas promedio de retraso en detección
- **Median Lag**: Mediana del lag
- **Min/Max Lag**: Rangos del lag

### 3. Validación Ground Truth
Comparación contra eventos históricos conocidos:

**BTC/USDT:**
- Early 2022 Correction (bear)
- Terra/UST Collapse (bear)
- Summer 2022 Consolidation (sideways)
- FTX Collapse (bear)
- Q1 2023 Rally (bull)
- ETF Approval Rally (bull)
- Post-Halving Consolidation (sideways)

**ETH/USDT:**
- DeFi Contagion 2022 (bear)
- Merge Sell-off (bear)
- ETF Narrative Rally (bull)
- Post-ETF Consolidation (sideways)

### 4. Comparación de Trading

**Estrategia CON L3:**
- Solo entra en régimen BULL o BREAKOUT
- Confianza mínima > 0.6
- Stop loss del 5%
- Take profit del 10%

**Estrategia SIN L3 (Momentum):**
- Entra en momentum > +2% en 24h
- Sale en momentum < -1% en 24h
- Stop loss del 5%
- Take profit del 10%

**Métricas comparadas:**
- PnL total
- Win rate
- Alpha añadido por L3
- Mejora en win rate

## Clasificador de Régimen L3

### MarketRegimeClassifier

Clasifica el mercado en 4 régimenes principales:

1. **TRENDING**: Tendencia clara
   - STRONG_BULL: Tendencia alcista fuerte
   - MODERATE_BULL: Tendencia alcista moderada
   - WEAK_BULL: Tendencia alcista débil
   - STRONG/MODERATE/WEAK_BEAR: Tendencias bajistas

2. **RANGE**: Mercado lateral
   - TIGHT_RANGE: Rango estrecho
   - NORMAL_RANGE: Rango normal
   - WIDE_RANGE: Rango amplio
   - OVERSOLD_SETUP: Setup de sobreventa
   - OVERBOUGHT_SETUP: Setup de sobrecompra

3. **VOLATILE**: Alta volatilidad
   - HIGH_VOLATILITY: Volatilidad extrema

4. **BREAKOUT**: Ruptura de rango
   - BULL_BREAKOUT: Ruptura alcista
   - BEAR_BREAKOUT: Ruptura bajista

### Mapeo a Régimenes Simplificados

```
TRENDING + BULL* → bull
TRENDING + BEAR* → bear
RANGE → sideways
VOLATILE → volatile
BREAKOUT → breakout
```

## Ejemplo de Salida

```
================================================================================
🚀 INICIANDO BACKTEST L3 RÉGIMEN
================================================================================
Símbolos: ['BTCUSDT', 'ETHUSDT']
Timeframe: 1h
Días: 730

📊 Cargando datos históricos...
📁 Cargando datos desde backtesting/data/btcusdt_1h.csv
📁 Cargando datos desde backtesting/data/ethusdt_1h.csv
✅ Datos cargados: ['BTCUSDT', 'ETHUSDT']
  - BTCUSDT: 17520 velas (730.0 días)
  - ETHUSDT: 17520 velas (730.0 días)

⏳ Ejecutando backtest de clasificación de régimen...
================================================================================
🚀 INICIANDO BACKTEST L3 RÉGIMEN
================================================================================

📊 Procesando BTCUSDT...
   Datos: 17520 velas (730.0 días)
   Cambios de régimen reales detectados: 23
   Validando contra labels manuales...

⏱️ Calculando lag del clasificador...
================================================================================
✅ BACKTEST COMPLETADO
================================================================================

💰 Comparando estrategias de trading...
================================================================================
📈 COMPARACIÓN DE ESTRATEGIAS DE TRADING
================================================================================

🔄 Comparando estrategias para BTCUSDT...
   Trading CON L3: PnL=1245.30, Win Rate=58.3%
   Trading SIN L3: PnL=892.50, Win Rate=48.2%
   Alpha añadido por L3: 352.80 (39.5%)

📄 Generando reportes...
📄 Reporte guardado en: backtesting/results/l3_regime_backtest_report_20240221_120000.txt
💾 Clasificaciones exportadas a: backtesting/results/l3_regime_classifications_20240221_120000.csv
💾 Resultados JSON guardados en: backtesting/results/l3_regime_backtest_results_20240221_120000.json

================================================================================
📊 RESUMEN FINAL
================================================================================
Total clasificaciones: 5840
Accuracy matriz confusión: 73.5%
Lag promedio: 4.2 velas
Alpha total añadido por L3: 587.40

================================================================================
✅ BACKTEST COMPLETADO
================================================================================
📄 Reporte: backtesting/results/l3_regime_backtest_report_20240221_120000.txt
💾 CSV: backtesting/results/l3_regime_classifications_20240221_120000.csv
📊 JSON: backtesting/results/l3_regime_backtest_results_20240221_120000.json
```

## Reporte Detallado

El archivo de reporte incluye:

1. **Resumen de Clasificaciones**: Total y distribución de régimenes
2. **Matriz de Confusión**: Actual vs Predicho con todas las métricas
3. **Análisis de Lag**: Estadísticas completas del lag de detección
4. **Validación Ground Truth**: Evento por evento con ✅/❌
5. **Comparación de Trading**: PnL, win rate y alpha por símbolo

### Ejemplo de Reporte

```
================================================================================
BACKTEST L3 RÉGIMEN - REPORTE DE RESULTADOS
================================================================================

📊 RESUMEN DE CLASIFICACIONES
--------------------------------------------------------------------------------
Total de clasificaciones: 5840
Cambios de régimen detectados: 156

📈 DISTRIBUCIÓN DE RÉGIMENES
--------------------------------------------------------------------------------
  bull        : 1523 (26.1%)
  bear        :  892 (15.3%)
  sideways    : 2845 (48.7%)
  volatile    :  412 (7.1%)
  breakout    :  168 (2.9%)

🎯 MATRIZ DE CONFUSIÓN
--------------------------------------------------------------------------------
Accuracy global: 73.5%

Matriz (Actual vs Predicho):
Actual/Pred       bull      bear   sideways  volatile
bull              18         2          3         0
bear               3        14          2         1
sideways           2         1         22         0
volatile           0         1          2         8

Métricas por régimen:
  bull        : Precision=78.3%, Recall=78.3%, F1=0.78
  bear        : Precision=77.8%, Recall=70.0%, F1=0.74
  sideways    : Precision=75.9%, Recall=88.0%, F1=0.81
  volatile    : Precision=88.9%, Recall=72.7%, F1=0.80

⏱️ ANÁLISIS DE LAG
--------------------------------------------------------------------------------
Cambios de régimen totales: 23
Cambios detectados: 21
Lag promedio: 4.2 velas
Lag mediana: 3.0 velas
Lag mínimo: 1 velas
Lag máximo: 12 velas

✅ VALIDACIÓN CONTRA GROUND TRUTH
--------------------------------------------------------------------------------
Eventos correctamente clasificados: 19/23 (82.6%)

✅ Early 2022 Correction              | Actual: bear     | Pred: bear     | Conf: 0.82
✅ Terra/UST Collapse                 | Actual: bear     | Pred: bear     | Conf: 0.91
✅ Summer Consolidation               | Actual: sideways | Pred: sideways | Conf: 0.76
❌ FTX Collapse                       | Actual: bear     | Pred: volatile | Conf: 0.68
✅ End of 2022 Accumulation           | Actual: sideways | Pred: sideways | Conf: 0.72
...

💰 COMPARACIÓN DE ESTRATEGIAS DE TRADING
--------------------------------------------------------------------------------

BTCUSDT:
  Trading CON L3:     PnL=1245.30
  Trading SIN L3:     PnL=892.50
  Alpha añadido:      352.80 (39.5%)
  Mejora Win Rate:    10.1%
  L3 añade alpha:     SÍ

ETHUSDT:
  Trading CON L3:     PnL=987.60
  Trading SIN L3:     PnL=753.20
  Alpha añadido:      234.40 (31.1%)
  Mejora Win Rate:    8.5%
  L3 añade alpha:     SÍ

================================================================================
FIN DEL REPORTE
================================================================================
```

## Interpretación de Resultados

### Buen Clasificador (targets)
- **Accuracy** > 70%
- **Lag promedio** < 6 velas (1-2 días en timeframe 1h)
- **Recall** > 65% para todos los régimenes
- **Alpha positivo** > 20%

### Casos de Error Comunes

1. **Lag alto**: El clasificador tarda mucho en detectar cambios
   - Solución: Reducir ventana de análisis o ajustar thresholds

2. **Confusión bull/bear**: Falsos positivos en tendencias
   - Revisar calibración de indicadores de momentum

3. **Volatile mal clasificado**: Mercado volátil detectado como rango
   - Ajustar threshold de volatilidad en el clasificador

## Datos Históricos

### Carga desde CSV Local

El script busca archivos CSV en `backtesting/data/` con estos nombres:
- `{symbol}_1h.csv` (ej: `btcusdt_1h.csv`)
- `{symbol}_features_train.csv`
- `{symbol}_features_test.csv`

### Descarga desde Binance

Si no hay datos locales, descarga automáticamente desde Binance usando CCXT.
Requiere al menos 1 año de datos (365 días) para resultados significativos.

Los datos descargados se guardan en caché para futuras ejecuciones.

## Notas

- El timeframe 1h es recomendado para análisis de régimen (más estable que 1m)
- La ventana de análisis de 168 velas (7 días) captura suficiente contexto
- Los labels manuales cubren eventos importantes de 2022-2024
- El lag se calcula comparando cambios detectados vs cambios reales
- El alpha se calcula como diferencia de PnL entre estrategias CON/SIN L3
