# /backtesting/hrm_tester.py

import logging
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from .backtesting_utils import TestMode, TestLevel, TestResult, L1Model
from .report_generator import ReportGenerator
from .getdata import BinanceDataCollector
from core.portfolio_manager import update_portfolio_from_orders
import asyncio
from core.logging import logger

class HRMStrategyTester:
    """Clase principal para ejecutar y evaluar la estrategia HRM"""

    def __init__(self, config: Dict, data_collector: BinanceDataCollector):
        self.config = config
        self.data_collector = data_collector
        self.logger = logger

    async def run_full_backtest(self):
        # Ya no se accede a self.config['binance'], se usa self.data_collector
        data = await self.data_collector.collect_historical_data(
            symbols=self.config['binance']['symbols'],
            intervals=self.config['binance']['intervals'],
            historical_days=self.config['binance']['historical_days']
        )
        
        # 2. Ejecución de la estrategia
        strategy_results = await self.run_hrm_strategy(data)
        
        # 3. Generación de reportes
        report_results = await self.report_generator.generate_complete_report(strategy_results)
        
        self.logger.info("Backtest completed successfully")
        return report_results

    async def run_hrm_strategy(self, data: Dict) -> Dict:
        """Ejecuta la estrategia HRM simulando el flujo real del sistema con SL/TP y P&L realista."""
        self.logger.info("🚀 Ejecutando estrategia HRM con flujo simulado...")
        
        try:
            # Inicializar resultados
            results = {
                'overall': {
                    'total_return': 0.0,
                    'annualized_return': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'win_rate': 0.0,
                    'profit_factor': 0.0,
                    'total_trades': 0
                },
                'l1_models': {},
                'l2_model': {},
                'l3_models': {},
                'trades': []
            }

            # Estado del portfolio simulado usando misma lógica que runtime
            initial_capital = float(self.config.get('initial_capital', 100000.0)) if isinstance(self.config, dict) else 100000.0
            state = {
                'initial_capital': initial_capital,
                'portfolio': {
                    'USDT': initial_capital,
                    'positions': {
                        'BTCUSDT': {'size': 0.0},
                        'ETHUSDT': {'size': 0.0}
                    },
                    'total_fees': 0.0,
                    'drawdown': 0.0,
                    'peak_value': initial_capital
                },
                'mercado': {}
            }

            # Parámetros de control para realismo
            # Perfiles de coste (pueden venir de config.validation.cost_profiles)
            cfg_fee = None
            cfg_slip = None
            try:
                vcfg = getattr(self, 'validation_profile', None)
                if isinstance(vcfg, dict):
                    cfg_fee = float(vcfg.get('fee_rate', 0.001))
                    cfg_slip = float(vcfg.get('slippage_pct', 0.0))
            except Exception:
                pass

            FEE_RATE = cfg_fee if cfg_fee is not None else 0.001
            TP_PCT = 0.015                 # 1.5% take-profit
            SL_PCT = 0.012                 # 1.2% stop-loss
            CONF_THRESHOLD = 0.72          # confianza mínima
            COOLDOWN_BARS = 20             # barras de enfriamiento tras cerrar
            ATR_K = 1.8                    # trailing stop por ATR
            ATR_MIN_RATIO = 0.002          # volatilidad mínima (0.2%)
            ATR_MAX_RATIO = 0.05           # volatilidad máxima (5%)

            # Procesar cada símbolo e intervalo
            for symbol in data:
                for interval in data[symbol]:
                    if isinstance(data[symbol][interval], pd.DataFrame) and not data[symbol][interval].empty:
                        df = data[symbol][interval].copy()
                        self.logger.info(f"    📊 Procesando {symbol} {interval}: {len(df)} filas")
                        
                        # Calcular RSI si no existe
                        if 'rsi' not in df.columns:
                            self.logger.info(f"    🔧 Calculando RSI para {symbol} {interval}")
                            delta = df['close'].diff()
                            up = delta.clip(lower=0).rolling(14, min_periods=5).mean()
                            down = -delta.clip(upper=0).rolling(14, min_periods=5).mean()
                            rs = up / (down.replace(0, np.nan))
                            df['rsi'] = 100 - (100 / (1 + rs.replace({np.inf: np.nan})))

                        # Calcular ATR(14) si no existe
                        if 'atr14' not in df.columns:
                            high = df['high']
                            low = df['low']
                            close = df['close']
                            prev_close = close.shift(1)
                            tr = pd.concat([
                                (high - low),
                                (high - prev_close).abs(),
                                (low - prev_close).abs()
                            ], axis=1).max(axis=1)
                            df['atr14'] = tr.rolling(14, min_periods=5).mean()
                        
                        # Simular el flujo real paso a paso
                        trades_temp = []
                        open_position = None  # {'entry_ts','entry_price','qty','tp','sl','confidence','trail_sl'}
                        cooldown_counter = 0
                        
                        for idx, row in df.iterrows():
                            try:
                                current_price = row['close']
                                # Actualizar mercado para el símbolo actual (por compatibilidad de cálculo)
                                state['mercado'][symbol] = {'close': float(current_price)}
                                
                                # Simular L1: Predicción AI (basada en RSI + volatilidad)
                                rsi = row.get('rsi', 50)
                                volatility = df['close'].rolling(20).std().iloc[-1] if len(df) > 20 else 0.02
                                
                                # L2: Generar señal táctica (simulando lógica real)
                                signal_strength = 0
                                signal_side = None
                                
                                # Estrategia basada en RSI + momentum
                                if rsi < 30 and current_price > df['close'].rolling(5).mean().iloc[-1]:
                                    signal_strength = 0.8  # Fuerte señal de compra
                                    signal_side = 'buy'
                                elif rsi > 70 and current_price < df['close'].rolling(5).mean().iloc[-1]:
                                    signal_strength = 0.8  # Fuerte señal de venta
                                    signal_side = 'sell'
                                elif 30 <= rsi <= 40 and current_price > df['close'].rolling(10).mean().iloc[-1]:
                                    signal_strength = 0.6  # Señal moderada de compra
                                    signal_side = 'buy'
                                elif 60 <= rsi <= 70 and current_price < df['close'].rolling(10).mean().iloc[-1]:
                                    signal_strength = 0.6  # Señal moderada de venta
                                    signal_side = 'sell'
                                
                                # Primero, gestionar salida por SL/TP/Trailing si hay posición abierta
                                if open_position is not None:
                                    ep = open_position['entry_price']
                                    qty = open_position['qty']
                                    tp = open_position['tp']
                                    sl = open_position['sl']
                                    trail_sl = open_position.get('trail_sl', sl)

                                    # Actualizar trailing stop con ATR
                                    atr = float(row.get('atr14', np.nan))
                                    if not np.isnan(atr) and atr > 0:
                                        candidate_trail = current_price - ATR_K * atr
                                        if candidate_trail > trail_sl:
                                            trail_sl = candidate_trail
                                    # Usar el mayor entre SL fijo y trailing
                                    eff_sl = max(sl, trail_sl)

                                    exit_reason = None
                                    if current_price >= tp:
                                        exit_reason = 'TP'
                                    elif current_price <= eff_sl:
                                        exit_reason = 'SL'

                                    # También permitir cierre por señal contraria fuerte
                                    if exit_reason is None and signal_side == 'sell' and signal_strength >= CONF_THRESHOLD:
                                        exit_reason = 'Signal'

                                    if exit_reason is not None:
                                        # Ejecutar orden de salida con la misma lógica que runtime
                                        sell_order = {
                                            'symbol': symbol,
                                            'side': 'sell',
                                            'quantity': float(qty),
                                            'filled_price': float(current_price),
                                            'status': 'filled'
                                        }
                                        await update_portfolio_from_orders(state, [sell_order])

                                        trades_temp.append({
                                            'entry_timestamp': open_position['entry_ts'],
                                            'exit_timestamp': idx,
                                            'symbol': symbol,
                                            'side': 'long',
                                            'quantity': qty,
                                            'entry_price': ep,
                                            'exit_price': float(current_price),
                                            'fees_entry': None,
                                            'fees_exit': None,
                                            'pnl': None,
                                            'confidence': open_position['confidence'],
                                            'exit_reason': exit_reason
                                        })
                                        self.logger.info(f"    ✅ Cierre {exit_reason}: {symbol} {qty:.4f} @ {current_price}")
                                        open_position = None
                                        cooldown_counter = COOLDOWN_BARS
                                        continue  # pasar a la siguiente barra tras cerrar

                                # Reducir cooldown si aplica
                                if cooldown_counter > 0:
                                    cooldown_counter -= 1
                                    continue

                                # Confirmación multi-timeframe: para 5m exigir alineación con 1h (MA20>MA50)
                                mtf_ok = True
                                if interval == '5m':
                                    df_1h = data.get(symbol, {}).get('1h')
                                    if isinstance(df_1h, pd.DataFrame) and not df_1h.empty:
                                        ma_fast = df_1h['close'].ewm(span=20, adjust=False).mean()
                                        ma_slow = df_1h['close'].ewm(span=50, adjust=False).mean()
                                        mtf_ok = bool(ma_fast.iloc[-1] > ma_slow.iloc[-1]) if signal_side == 'buy' else bool(ma_fast.iloc[-1] < ma_slow.iloc[-1])

                                # Filtro de volatilidad por ATR
                                atr = float(row.get('atr14', np.nan))
                                atr_ok = True
                                if not np.isnan(atr) and current_price > 0:
                                    atr_ratio = atr / current_price
                                    atr_ok = (ATR_MIN_RATIO <= atr_ratio <= ATR_MAX_RATIO)

                                # Apertura de posición sólo si no hay posición, confianza mínima, MTF y ATR válidos
                                if open_position is None and signal_side == 'buy' and signal_strength >= CONF_THRESHOLD and mtf_ok and atr_ok:
                                    # USDT disponible del estado
                                    available_capital = float(state.get('portfolio', {}).get('USDT', 0.0))
                                    # Tamaño por riesgo/convicción limitado al 5% del capital
                                    position_value = min(available_capital * 0.05, available_capital * 0.1 * signal_strength)
                                    qty = max(0.0, position_value / current_price)
                                    if qty > 0 and (available_capital >= position_value * (1 + FEE_RATE)):
                                        # Ejecutar orden de entrada con la misma lógica que runtime
                                        # aplicar slippage al precio de entrada
                                        price_in = float(current_price) * (1.0 + (cfg_slip or 0.0))
                                        buy_order = {
                                            'symbol': symbol,
                                            'side': 'buy',
                                            'quantity': float(qty),
                                            'filled_price': price_in,
                                            'status': 'filled'
                                        }
                                        await update_portfolio_from_orders(state, [buy_order])

                                        # Definir SL/TP relativos al precio de entrada
                                        tp = price_in * (1 + TP_PCT)
                                        sl = price_in * (1 - SL_PCT)
                                        open_position = {
                                            'entry_ts': idx,
                                            'entry_price': float(price_in),
                                            'qty': float(qty),
                                            'tp': float(tp),
                                            'sl': float(sl),
                                            'trail_sl': float(sl),
                                            'confidence': float(signal_strength)
                                        }
                                        self.logger.info(f"    ✅ Apertura long: {symbol} {qty:.4f} @ {price_in} TP={tp:.2f} SL={sl:.2f} (conf: {signal_strength:.2f})")
                                
                            except Exception as e:
                                self.logger.error(f"    ❌ Error procesando {symbol} {idx}: {e}")
                                continue

                        # Agregar trades de este símbolo/intervalo
                        if trades_temp:
                            results['trades'].extend(trades_temp)
                            self.logger.info(f"    📈 {symbol} {interval}: {len(trades_temp)} trades generados")

            # Calcular métricas finales usando el mismo P&L del runtime
            results = await self._calculate_final_metrics(results, state)
            
            self.logger.info(f"🎯 Estrategia HRM completada: {len(results['trades'])} trades ejecutados")
            return results

        except Exception as e:
            self.logger.error(f"❌ Error ejecutando estrategia HRM: {e}")
            raise


    async def _calculate_final_metrics(self, results, state):
        """Calcula métricas finales del backtesting usando valores del portfolio runtime-like"""
        try:
            trades = results.get('trades', [])
            portfolio = state.get('portfolio', {})
            initial_capital = float(state.get('initial_capital', 100000.0))
            total_value = float(state.get('total_value', initial_capital))

            total_trades = len(trades)
            winning_trades = len([t for t in trades if (t.get('pnl', 0) or 0) > 0])
            losing_trades = len([t for t in trades if (t.get('pnl', 0) or 0) < 0])
            win_rate = (winning_trades / total_trades) if total_trades > 0 else 0.0

            total_return = (total_value - initial_capital) / initial_capital if initial_capital > 0 else 0.0

            results['overall'].update({
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_return': total_return,
            })
            
            return results
        except Exception as e:
            self.logger.error(f"Error calculando métricas: {e}")
            return results