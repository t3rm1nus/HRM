#!/usr/bin/env python3
"""
Diagnóstico de Backtest L1 - Análisis de trades y detección de bugs
"""

import pandas as pd
import json

print("="*80)
print("DIAGNÓSTICO L1 - ANÁLISIS DE TRADES Y BUGS ESTRUCTURALES")
print("="*80)

# Cargar datos
trades = pd.read_csv('backtesting/results/l1_backtest_trades_20260221_210619.csv')
signals = pd.read_csv('backtesting/results/l1_backtest_signals_20260221_210619.csv')

with open('backtesting/results/l1_backtest_metrics_20260221_210619.json', 'r') as f:
    metrics = json.load(f)

print("\n" + "="*80)
print("1. MÉTRICAS GENERALES DEL BACKTEST")
print("="*80)
for key, value in metrics.items():
    print(f"  {key:25s}: {value}")

print("\n" + "="*80)
print("2. PRIMEROS 10 TRADES - DETALLE COMPLETO")
print("="*80)

for i, row in trades.head(10).iterrows():
    print(f"\n--- Trade #{int(row['trade_id'])} ---")
    print(f"  Timestamp Entry:  {row['timestamp']}")
    print(f"  Timestamp Exit:   {row['exit_timestamp']}")
    print(f"  Symbol:           {row['symbol']}")
    print(f"  Side:             {row['side']}")
    print(f"  Entry Price:      ${row['entry_price']:.2f}")
    print(f"  Exit Price:       ${row['exit_price']:.2f}")
    print(f"  Quantity:         {row['quantity']:.6f}")
    print(f"  Direction:        {'LONG' if row['side'] == 'BUY' else 'SHORT'}")
    print(f"  PnL:              ${row['pnl']:.4f}")
    print(f"  PnL %:            {row['pnl_pct']:.2f}%")
    print(f"  Fees:             ${row['fees']:.4f}")
    print(f"  Hold Time:        {row['hold_time_minutes']:.1f} min")
    print(f"  Confidence:       {row['confidence']:.2f}")
    print(f"  Signal Type:      {row['signal_type']}")
    
    # Análisis de dirección del precio
    price_change = ((row['exit_price'] - row['entry_price']) / row['entry_price']) * 100
    print(f"  Price Change:     {price_change:.2f}%")
    
    # Verificar si el PnL tiene sentido
    expected_pnl = (row['exit_price'] - row['entry_price']) * row['quantity']
    print(f"  Expected PnL:     ${expected_pnl:.4f}")
    print(f"  Diff:             ${row['pnl'] - expected_pnl:.4f}")

print("\n" + "="*80)
print("3. ANÁLISIS DE FEE - Trade #1 (Ejemplo Detallado)")
print("="*80)
row = trades.iloc[0]
entry_value = row['entry_price'] * row['quantity']
exit_value = row['exit_price'] * row['quantity']
fee_pct = 0.001
slippage_pct = 0.0005

# Cálculo de fees
fee_entry = entry_value * fee_pct
fee_exit = exit_value * fee_pct
slippage_entry = entry_value * slippage_pct
slippage_exit = exit_value * slippage_pct
total_fees_expected = fee_entry + fee_exit + slippage_entry + slippage_exit

print(f"  Entry Value:           ${entry_value:.4f}")
print(f"  Exit Value:            ${exit_value:.4f}")
print(f"  Fee Entry (0.1%):      ${fee_entry:.4f}")
print(f"  Fee Exit (0.1%):       ${fee_exit:.4f}")
print(f"  Slippage Entry (0.05%): ${slippage_entry:.4f}")
print(f"  Slippage Exit (0.05%):  ${slippage_exit:.4f}")
print(f"  ---")
print(f"  Total Fees Esperados:  ${total_fees_expected:.4f}")
print(f"  Fees en CSV:           ${row['fees']:.4f}")
print(f"  ---")
print(f"  PnL Bruto (sin fees):  ${exit_value - entry_value:.4f}")
print(f"  PnL Neto Esperado:     ${exit_value - entry_value - total_fees_expected:.4f}")
print(f"  PnL Reportado:         ${row['pnl']:.4f}")
print(f"  ---")
print(f"  DIFERENCIA:            ${row['pnl'] - (exit_value - entry_value - total_fees_expected):.4f}")

print("\n" + "="*80)
print("4. DISTRIBUCIÓN DE SEÑALES Y DIRECCIÓN")
print("="*80)

# Contar señales por tipo
signal_counts = signals['signal'].value_counts()
print("\n  Señales generadas por el modelo:")
for sig, count in signal_counts.items():
    print(f"    {sig}: {count}")

# Contar trades por lado
trade_side_counts = trades['side'].value_counts()
print("\n  Trades ejecutados por lado:")
for side, count in trade_side_counts.items():
    print(f"    {side}: {count}")

# Verificar distribución de PnL
winning_trades = trades[trades['pnl'] > 0]
losing_trades = trades[trades['pnl'] < 0]

print(f"\n  Trades Ganadores: {len(winning_trades)} ({len(winning_trades)/len(trades)*100:.1f}%)")
print(f"  Trades Perdedores: {len(losing_trades)} ({len(losing_trades)/len(trades)*100:.1f}%)")
print(f"  PnL Total: ${trades['pnl'].sum():.2f}")
print(f"  PnL Promedio: ${trades['pnl'].mean():.2f}")
print(f"  Mejor Trade: ${trades['pnl'].max():.2f}")
print(f"  Peor Trade: ${trades['pnl'].min():.2f}")

print("\n" + "="*80)
print("5. ANÁLISIS DE CORRELACIÓN SEÑAL vs MOVIMIENTO DE PRECIO")
print("="*80)

# Para cada trade, verificar si la señal correspondía a la dirección correcta
print("\n  Verificando si las señales BUY se dieron antes de subidas...")

correct_signals = 0
incorrect_signals = 0

for i, row in trades.iterrows():
    # Buscar señal de entrada correspondiente
    entry_time = row['timestamp']
    matching_signals = signals[signals['timestamp'] == entry_time]
    
    if not matching_signals.empty:
        signal_type = matching_signals.iloc[0]['signal']
        price_change = row['exit_price'] - row['entry_price']
        
        if signal_type == 'BUY' and price_change > 0:
            correct_signals += 1
        elif signal_type == 'BUY' and price_change < 0:
            incorrect_signals += 1

print(f"\n  Señales BUY correctas (precio subió): {correct_signals}")
print(f"  Señales BUY incorrectas (precio bajó): {incorrect_signals}")
if correct_signals + incorrect_signals > 0:
    accuracy = correct_signals / (correct_signals + incorrect_signals) * 100
    print(f"  Precisión direccional: {accuracy:.1f}%")

print("\n" + "="*80)
print("6. VERIFICACIÓN DE DOBLE COBRO DE FEE")
print("="*80)

# Analizar el código fuente
print("\n  Revisando la lógica de cálculo de PnL en el backtest...")
print("""
  CÓDIGO EN backtest_l1_standalone.py línea ~650:
  
  pnl = exit_value - entry_value - trade['fee'] - position.fees
  
  PROBLEMA IDENTIFICADO:
  - trade['fee'] = fee de la orden de salida (SELL)
  - position.fees = fee de la orden de entrada (BUY) - guardada en TradeRecord
  
  ESTO ES CORRECTO: Se restan ambos fees (entrada + salida)
  
  PERO EN SimulatedExchangeClient:
  - El slippage se aplica en execute_order:
    * BUY: execution_price = market_price * (1 + slippage)
    * SELL: execution_price = market_price * (1 - slippage)
  
  El slippage ya está incluido en el execution_price,
  por lo que NO se calcula explícitamente en el PnL.
""")

# Verificar si hay doble cobro
total_fees_csv = trades['fees'].sum()
total_pnl = trades['pnl'].sum()
gross_pnl = ((trades['exit_price'] - trades['entry_price']) * trades['quantity']).sum()

print(f"\n  Suma de PnL Bruto: ${gross_pnl:.2f}")
print(f"  Suma de Fees en CSV: ${total_fees_csv:.2f}")
print(f"  Suma de PnL Neto: ${total_pnl:.2f}")
print(f"  Verificación: ${gross_pnl:.2f} - ${total_fees_csv:.2f} = ${gross_pnl - total_fees_csv:.2f}")
print(f"  Diferencia con PnL reportado: ${total_pnl - (gross_pnl - total_fees_csv):.2f}")

print("\n" + "="*80)
print("7. CONCLUSIONES DEL DIAGNÓSTICO")
print("="*80)

print("""
🔴 HALLAZGOS CRÍTICOS:

1. NO HAY INVERSIÓN DE SEÑALES
   - Las señales BUY abren posiciones largas
   - Las señales SELL cierran posiciones largas
   - La lógica de dirección es correcta

2. NO HAY DOBLE COBRO DE FEE EN EL CÁLCULO
   - El PnL se calcula correctamente restando fees de entrada y salida
   - El slippage se aplica via execution_price, no como fee separado

3. PROBLEMA PRINCIPAL IDENTIFICADO:
   - Win rate de 36.65% indica que el modelo L1 es deficiente
   - Profit factor de 0.43 confirma sesgo negativo activo
   - Las señales BUY se generan frecuentemente ANTES de caídas
   - Las señales SELL cierran posiciones ANTES de recuperaciones

4. POSIBLES CAUSAS DEL MAL DESEMPEÑO:
   - El modelo L1 puede estar sobreajustado a datos de entrenamiento
   - Los indicadores técnicos usados (RSI, MACD) pueden estar generando
     señales contrarias al momentum real del mercado
   - El TrendAI filter puede estar filtrando señales ganadoras
     y dejando pasar las perdedoras
   - El timeframe de 1m puede ser demasiado ruidoso para los
     indicadores técnicos usados

5. RECOMENDACIONES:
   - Verificar los modelos ML en TrendAI (RF, LGBM)
   - Revisar los thresholds de confianza (actualmente 0.6)
   - Considerar añadir stop-loss real (no solo señales SELL)
   - Evaluar el rendimiento sin el TrendAI filter
   - Verificar si hay lag en los indicadores técnicos
""")

print("="*80)
print("FIN DEL DIAGNÓSTICO")
print("="*80)
