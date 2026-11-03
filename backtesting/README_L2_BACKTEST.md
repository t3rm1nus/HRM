# Backtesting L2 Standalone

Script independiente para backtesting de filtros L2 sobre señales L1 generadas previamente.

## Descripción

Este script analiza el valor añadido de los componentes L2 (`l2_tactic/`) sobre las señales L1:

1. **Carga** el CSV de señales L1 generado por `backtest_l1_standalone.py`
2. **Instancia** SOLO los componentes de `l2_tactic/`:
   - `signal_generator_refactored.py` - Procesador de señales
   - `position_sizer.py` - Calculador de tamaño de posición
   - `risk_overlay.py` - Filtros de riesgo
3. **Para cada señal L1**, procesa a través de L2 observando:
   - Timing de entrada modificado
   - Tamaño de posición calculado
   - Stop loss dinámico asignado
4. **Compara** dos equity curves: "L1 solo" vs "L1 + filtro L2"
5. **Métricas adicionales**: señales rechazadas, mejora Sharpe, reducción drawdown
6. **Test de robustez**: 3 configuraciones diferentes

**IMPORTANTE**: No importa nada de `l3_strategy/` - L2 completamente aislado.

## Uso

### Backtest L2 con CSV de L1 más reciente

```bash
python backtesting/backtest_l2_standalone.py
```

### Backtest L2 con CSV específico

```bash
python backtesting/backtest_l2_standalone.py --l1-csv results/l1_backtest_signals_20260221_120000.csv
```

### Test de robustez con 3 configuraciones

```bash
python backtesting/backtest_l2_standalone.py --robustness-test
```

### Configuración personalizada

```bash
python backtesting/backtest_l2_standalone.py --l1-csv signals.csv --balance 5000 --config conservative
```

## Configuraciones Disponibles

### Conservador
- Kelly fracción: 0.3
- Max riesgo por trade: 0.5%
- Confianza mínima: 0.7
- Todos los filtros activados

### Balanceado (default)
- Kelly fracción: 0.5
- Max riesgo por trade: 1.0%
- Confianza mínima: 0.6
- Todos los filtros activados

### Agresivo
- Kelly fracción: 0.7
- Max riesgo por trade: 2.0%
- Confianza mínima: 0.5
- Filtros desactivados

## Salida

### Archivos Generados

1. **l2_decisions_{config}_{timestamp}.csv** - Decisiones L2 para cada señal L1
2. **l2_backtest_{config}_{timestamp}.json** - Métricas y resultados detallados
3. **l2_robustness_summary_{timestamp}.json** - Resumen del test de robustez

### Métricas Reportadas

#### Métricas Comunes (L1 y L1+L2)
- Total trades
- Win rate
- Profit factor
- Max drawdown
- Sharpe ratio
- Total PnL
- Average trade PnL
- Best/worst trade

#### Métricas Específicas L2
- Señales L1 procesadas
- Señales L2 aprobadas
- Señales L2 rechazadas
- Tasa de rechazo L2
- Tamaño promedio de posición L1 vs L2
- Leverage promedio L2

### Comparación L1 vs L1+L2
El script muestra:
- Diferencia de PnL (USDT y %)
- Diferencia de Sharpe ratio
- Diferencia de Max Drawdown
- Análisis de cuántas señales rechaza L2

## Flujo de Procesamiento

```
Señal L1 (CSV)
    ↓
[Risk Overlay] - Verificar drawdown, volatilidad, correlación
    ↓
[Confianza Mínima] - Filtrar por umbral de confianza
    ↓
[Position Sizer] - Calcular tamaño óptimo con Kelly + vol targeting
    ↓
Decisión L2 (aprobar/rechazar + modificaciones)
    ↓
Ejecución Trade (L1 solo vs L1+L2)
    ↓
Equity Curves comparativas
```

## Test de Robustez

Ejecuta el mismo backtest con 3 configuraciones distintas y analiza:

1. **Estabilidad de resultados** - Desviación estándar de métricas
2. **Sensibilidad a parámetros** - Rango de variación PnL y Sharpe
3. **Score de estabilidad** - STABLE vs VARIABLE

### Interpretación

- **Resultados ESTABLES**: Desviación estándar < 50% de la media
- **Resultados VARIABLES**: Alta sensibilidad a cambios de parámetros

## Requisitos

1. Ejecutar primero `backtest_l1_standalone.py` para generar señales L1
2. Python 3.8+
3. Dependencias: pandas, numpy

## Ejemplo de Salida

```
================================================================================
📊 RESULTADOS DEL BACKTEST L2
================================================================================

Configuración: Balanceado
----------------------------------------

📈 ESTRATEGIA L1 (sin filtro):
  total_trades              : 150
  win_rate                  : 52.00%
  sharpe_ratio              : 1.25
  max_drawdown_pct          : 15.20%
  total_pnl                 : $450.50

🛡️ ESTRATEGIA L1 + L2 (con filtro):
  total_trades              : 95
  win_rate                  : 58.00%
  sharpe_ratio              : 1.65
  max_drawdown_pct          : 10.50%
  total_pnl                 : $520.30
  l1_signals_processed      : 150
  l2_signals_rejected       : 55
  l2_rejection_rate         : 36.67%

📊 COMPARACIÓN L1 vs L1+L2:
  Diferencia PnL:          +69.80 USDT (+15.5%)
  Diferencia Sharpe:       +0.40
  Diferencia Max DD:       -4.70%
  Señales L2 rechazadas:   55 (36.67%)
```

## Notas

- Las señales HOLD de L1 no se procesan en L2
- El position sizer puede rechazar señales por tamaño insuficiente
- El risk overlay filtra por drawdown, volatilidad y correlación
- Las modificaciones de L2 incluyen: tamaño, stop loss, take profit, leverage
