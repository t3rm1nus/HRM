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

# Importar componentes reales del sistema HRM
from l2_tactic.signal_generator import L2TacticProcessor
from l2_tactic.config import L2Config
from l1_operational.order_manager import OrderManager
from l1_operational.binance_client import BinanceClient
from l3_strategy.l3_processor import generate_l3_output
from core.state_manager import initialize_state, validate_state_structure

class HRMStrategyTester:
    """Clase principal para ejecutar y evaluar la estrategia HRM usando componentes reales"""

    def __init__(self, config: Dict, data_collector: BinanceDataCollector):
        self.config = config
        self.data_collector = data_collector
        self.logger = logger

        # Inicializar componentes reales del sistema HRM
        self.l2_processor = None
        self.order_manager = None
        self.binance_client = None

        # Pre-cargar modelos para evitar recargas constantes
        self.models_cache = {}
        self._preload_models()

    def _preload_models(self):
        """Pre-carga todos los modelos ML para evitar recargas constantes durante el backtesting"""
        try:
            self.logger.info("🔄 Pre-cargando modelos ML para optimización de backtesting...")

            # Importar funciones de carga de modelos
            from l3_strategy.l3_processor import (
                load_regime_model, load_sentiment_model, load_vol_models, load_portfolio
            )

            # Pre-cargar modelo de regime detection
            try:
                self.models_cache['regime'] = load_regime_model()
                self.logger.info("✅ Modelo de regime detection pre-cargado")
            except Exception as e:
                self.logger.warning(f"⚠️ No se pudo pre-cargar modelo de regime: {e}")
                self.models_cache['regime'] = None

            # Pre-cargar modelos de sentimiento
            try:
                tokenizer, model = load_sentiment_model()
                self.models_cache['sentiment'] = {'tokenizer': tokenizer, 'model': model}
                self.logger.info("✅ Modelos de sentimiento pre-cargados")
            except Exception as e:
                self.logger.warning(f"⚠️ No se pudieron pre-cargar modelos de sentimiento: {e}")
                self.models_cache['sentiment'] = {'tokenizer': None, 'model': None}

            # Pre-cargar modelos de volatilidad
            try:
                garch_btc, garch_eth, lstm_btc, lstm_eth = load_vol_models()
                self.models_cache['volatility'] = {
                    'garch_btc': garch_btc, 'garch_eth': garch_eth,
                    'lstm_btc': lstm_btc, 'lstm_eth': lstm_eth
                }
                self.logger.info("✅ Modelos de volatilidad pre-cargados")
            except Exception as e:
                self.logger.warning(f"⚠️ No se pudieron pre-cargar modelos de volatilidad: {e}")
                self.models_cache['volatility'] = {
                    'garch_btc': None, 'garch_eth': None,
                    'lstm_btc': None, 'lstm_eth': None
                }

            # Pre-cargar modelo de portfolio
            try:
                cov_matrix, optimal_weights = load_portfolio()
                self.models_cache['portfolio'] = {'cov': cov_matrix, 'weights': optimal_weights}
                self.logger.info("✅ Modelo de portfolio pre-cargado")
            except Exception as e:
                self.logger.warning(f"⚠️ No se pudo pre-cargar modelo de portfolio: {e}")
                self.models_cache['portfolio'] = {'cov': None, 'weights': None}

            self.logger.info("🎯 Pre-carga de modelos completada")

        except Exception as e:
            self.logger.error(f"❌ Error durante pre-carga de modelos: {e}")
            self.models_cache = {}

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
        """Ejecuta la estrategia HRM completa usando los componentes reales L1+L2+L3"""
        self.logger.info("🚀 Ejecutando estrategia HRM completa con componentes reales...")

        try:
            # Inicializar componentes reales del sistema HRM
            if self.l2_processor is None:
                try:
                    l2_config = L2Config()
                    self.l2_processor = L2TacticProcessor(l2_config)
                    self.logger.info("✅ L2 Processor inicializado para backtesting")
                except Exception as e:
                    self.logger.error(f"❌ Error inicializando L2: {e}")
                    raise

            if self.order_manager is None:
                try:
                    self.binance_client = BinanceClient()
                    self.order_manager = OrderManager(binance_client=self.binance_client, market_data={})
                    self.logger.info("✅ Order Manager inicializado para backtesting")
                except Exception as e:
                    self.logger.error(f"❌ Error inicializando Order Manager: {e}")
                    raise

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

            # Track open positions for proper trade recording
            open_positions = {}  # symbol -> {'entry_price': float, 'quantity': float, 'entry_timestamp': datetime}

            # Estado del sistema HRM usando la misma estructura que main.py
            symbols = self.config.get('symbols', ['BTCUSDT', 'ETHUSDT'])
            initial_capital = float(self.config.get('initial_capital', 1000.0))  # Reduced to 1000 euros for more activity
            state = initialize_state(symbols, initial_capital)
            state = validate_state_structure(state)

            # Procesar datos históricos como si fueran tiempo real
            # Convertir datos históricos a formato compatible con el sistema
            processed_data = self._prepare_historical_data_for_backtest(data)

            # Ejecutar backtest ciclo por ciclo con límite para evitar loop infinito
            cycle_count = 0
            max_cycles = min(1000, len(processed_data))  # Máximo 1000 ciclos o longitud de datos

            for timestamp, market_snapshot in processed_data.items():
                cycle_count += 1

                # Limitar ciclos para evitar loop infinito
                if cycle_count > max_cycles:
                    self.logger.info(f"🛑 Límite de ciclos alcanzado ({max_cycles}), deteniendo backtest")
                    break

                try:
                    # ✅ FIXED: Proper market data formatting for all components
                    # Create different data formats for different components

                    # 1. For L2 processor - needs historical data with at least 200 points
                    l2_market_data = {}
                    for symbol, data in market_snapshot.items():
                        if isinstance(data, dict) and 'historical_data' in data:
                            df = data['historical_data']
                            if isinstance(df, pd.DataFrame) and len(df) >= 200:
                                l2_market_data[symbol] = df.copy()
                                self.logger.debug(f"L2 data for {symbol}: {len(df)} points")
                            else:
                                self.logger.warning(f"Insufficient L2 data for {symbol}: {len(df) if isinstance(df, pd.DataFrame) else 'N/A'} < 200")

                    # 2. For OrderManager - needs current price data as DataFrames
                    order_manager_market_data = {}
                    for symbol, data in market_snapshot.items():
                        if isinstance(data, dict) and 'historical_data' in data:
                            df = data['historical_data']
                            if isinstance(df, pd.DataFrame) and not df.empty:
                                order_manager_market_data[symbol] = df.copy()
                        elif isinstance(data, dict):
                            # Convert current snapshot to DataFrame
                            df_data = {k: [v] for k, v in data.items() if k != 'historical_data' and pd.api.types.is_numeric_dtype(type(v))}
                            if df_data:
                                order_manager_market_data[symbol] = pd.DataFrame(df_data)

                    # 3. For L3 processor - needs full market data
                    state["market_data"] = market_snapshot.copy()
                    state["market_data_simple"] = l2_market_data  # L2 expects this key
                    state["mercado"] = order_manager_market_data  # Some components expect this key

                    # Generar L3 output (cada 10 minutos en producción, aquí por ciclo para testing)
                    try:
                        l3_output = generate_l3_output(state, preloaded_models=self.models_cache)
                        state["l3_output"] = l3_output  # Store L3 output in state for L2 access
                    except Exception as e:
                        self.logger.warning(f"L3 error en ciclo {cycle_count}: {e}")
                        state["l3_output"] = {}  # Ensure L3 output key exists even on error

                    # Procesar señales L2
                    try:
                        signals = await self.l2_processor.process_signals(state)
                        valid_signals = [s for s in signals if hasattr(s, 'symbol') and hasattr(s, 'side')]
                    except Exception as e:
                        self.logger.error(f"L2 error en ciclo {cycle_count}: {e}")
                        valid_signals = []

                    # Generar y ejecutar órdenes L1
                    try:
                        orders = await self.order_manager.generate_orders(state, valid_signals)
                        processed_orders = await self.order_manager.execute_orders(orders)

                        # Actualizar portfolio
                        await update_portfolio_from_orders(state, processed_orders)

                        # Process trades and track positions for proper entry/exit recording
                        for order in processed_orders:
                            if order.get('status') == 'filled':
                                symbol = order.get('symbol')
                                side = order.get('side')
                                quantity = abs(float(order.get('quantity', 0)))
                                price = float(order.get('filled_price', 0))
                                commission = float(order.get('commission', 0))

                                if side == 'buy':
                                    # Check if we have an existing short position to close
                                    if symbol in open_positions and open_positions[symbol]['quantity'] < 0:
                                        # Close short position
                                        open_qty = abs(open_positions[symbol]['quantity'])
                                        close_qty = min(quantity, open_qty)
                                        entry_price = open_positions[symbol]['entry_price']
                                        exit_price = price

                                        # Calculate P&L for closed portion
                                        pnl = (entry_price - exit_price) * close_qty - commission

                                        trade = {
                                            'symbol': symbol,
                                            'side': 'close_short',
                                            'entry_timestamp': open_positions[symbol]['entry_timestamp'],
                                            'exit_timestamp': timestamp,
                                            'entry_price': entry_price,
                                            'exit_price': exit_price,
                                            'quantity': close_qty,
                                            'pnl': pnl,
                                            'commission': commission
                                        }
                                        results['trades'].append(trade)

                                        # Update remaining position
                                        remaining_qty = open_qty - close_qty
                                        if remaining_qty > 0:
                                            open_positions[symbol]['quantity'] = -remaining_qty
                                        else:
                                            # Position fully closed
                                            if quantity > close_qty:
                                                # Open new long position with remaining quantity
                                                open_positions[symbol] = {
                                                    'entry_price': price,
                                                    'quantity': quantity - close_qty,
                                                    'entry_timestamp': timestamp
                                                }
                                            else:
                                                del open_positions[symbol]
                                    else:
                                        # Open new long position or add to existing
                                        if symbol in open_positions:
                                            # Average the entry price for additional position
                                            existing_qty = open_positions[symbol]['quantity']
                                            existing_value = open_positions[symbol]['entry_price'] * existing_qty
                                            new_value = price * quantity
                                            total_qty = existing_qty + quantity
                                            avg_price = (existing_value + new_value) / total_qty
                                            open_positions[symbol] = {
                                                'entry_price': avg_price,
                                                'quantity': total_qty,
                                                'entry_timestamp': open_positions[symbol]['entry_timestamp']
                                            }
                                        else:
                                            open_positions[symbol] = {
                                                'entry_price': price,
                                                'quantity': quantity,
                                                'entry_timestamp': timestamp
                                            }

                                elif side == 'sell':
                                    # Check if we have an existing long position to close
                                    if symbol in open_positions and open_positions[symbol]['quantity'] > 0:
                                        # Close long position
                                        open_qty = open_positions[symbol]['quantity']
                                        close_qty = min(quantity, open_qty)
                                        entry_price = open_positions[symbol]['entry_price']
                                        exit_price = price

                                        # Calculate P&L for closed portion
                                        pnl = (exit_price - entry_price) * close_qty - commission

                                        trade = {
                                            'symbol': symbol,
                                            'side': 'close_long',
                                            'entry_timestamp': open_positions[symbol]['entry_timestamp'],
                                            'exit_timestamp': timestamp,
                                            'entry_price': entry_price,
                                            'exit_price': exit_price,
                                            'quantity': close_qty,
                                            'pnl': pnl,
                                            'commission': commission
                                        }
                                        results['trades'].append(trade)

                                        # Update remaining position
                                        remaining_qty = open_qty - close_qty
                                        if remaining_qty > 0:
                                            open_positions[symbol]['quantity'] = remaining_qty
                                        else:
                                            # Position fully closed
                                            if quantity > close_qty:
                                                # Open new short position with remaining quantity
                                                open_positions[symbol] = {
                                                    'entry_price': price,
                                                    'quantity': -(quantity - close_qty),
                                                    'entry_timestamp': timestamp
                                                }
                                            else:
                                                del open_positions[symbol]
                                    else:
                                        # Open new short position or add to existing
                                        if symbol in open_positions:
                                            # Average the entry price for additional position
                                            existing_qty = abs(open_positions[symbol]['quantity'])
                                            existing_value = open_positions[symbol]['entry_price'] * existing_qty
                                            new_value = price * quantity
                                            total_qty = existing_qty + quantity
                                            avg_price = (existing_value + new_value) / total_qty
                                            open_positions[symbol] = {
                                                'entry_price': avg_price,
                                                'quantity': -total_qty,
                                                'entry_timestamp': open_positions[symbol]['entry_timestamp']
                                            }
                                        else:
                                            open_positions[symbol] = {
                                                'entry_price': price,
                                                'quantity': -quantity,
                                                'entry_timestamp': timestamp
                                            }

                    except Exception as e:
                        self.logger.error(f"L1 error en ciclo {cycle_count}: {e}")

                    # Logging de progreso cada 100 ciclos
                    if cycle_count % 100 == 0:
                        self.logger.info(f"📊 Ciclo {cycle_count}/{max_cycles} completado - Señales: {len(valid_signals)}")

                except Exception as e:
                    self.logger.error(f"Error en ciclo {cycle_count}: {e}")
                    continue

            # Calcular métricas finales
            results = await self._calculate_final_metrics(results, state)

            self.logger.info(f"🎯 Backtest HRM completado: {len(results['trades'])} trades, {cycle_count} ciclos")

            # Limpiar recursos al final del backtesting completo
            try:
                # Limpiar modelos ML
                from l3_strategy.l3_processor import cleanup_models
                cleanup_models()
                self.logger.info("🧹 Modelos limpiados al finalizar backtesting")
            except Exception as e:
                self.logger.warning(f"⚠️ Error limpiando modelos: {e}")

            try:
                # Cerrar conexión Binance client
                if self.binance_client:
                    await self.binance_client.close()
                    self.logger.info("🔌 Conexión Binance cerrada correctamente")
            except Exception as e:
                self.logger.warning(f"⚠️ Error cerrando conexión Binance: {e}")

            return results

        except Exception as e:
            self.logger.error(f"❌ Error crítico en backtest HRM: {e}")
            raise

    def _prepare_historical_data_for_backtest(self, data: Dict) -> Dict:
        """Convierte datos históricos al formato esperado por el sistema HRM con diversidad mejorada"""
        processed_data = {}

        # Encontrar todos los timestamps únicos
        all_timestamps = set()
        for symbol in data:
            for interval in data[symbol]:
                if isinstance(data[symbol][interval], pd.DataFrame):
                    all_timestamps.update(data[symbol][interval].index)

        # Ordenar timestamps
        sorted_timestamps = sorted(all_timestamps)

        # Solo procesar timestamps donde tengamos suficientes datos históricos previos
        min_history_required = 200  # L2 requiere al menos 200 puntos

        # Add synthetic data generation for diversity
        synthetic_scenarios = self._generate_synthetic_scenarios(sorted_timestamps, data)

        for i, timestamp in enumerate(sorted_timestamps):
            # Solo procesar si tenemos suficiente historial
            if i < min_history_required:
                continue

            market_snapshot = {}

            for symbol in data:
                # Usar el intervalo más granular disponible (5m preferido)
                for interval in ['5m', '15m', '1h', '1d']:
                    if interval in data[symbol] and isinstance(data[symbol][interval], pd.DataFrame):
                        df = data[symbol][interval]

                        # Crear una ventana histórica de los últimos min_history_required puntos
                        historical_window = sorted_timestamps[max(0, i - min_history_required):i+1]

                        # Filtrar datos disponibles en esta ventana
                        available_data = df[df.index.isin(historical_window)]

                        if len(available_data) >= min_history_required:
                            # Apply synthetic scenario modifications for diversity
                            scenario_modifier = synthetic_scenarios.get(timestamp, {})
                            symbol_modifier = scenario_modifier.get(symbol, {})

                            # Use the most recent point for the snapshot
                            current_row = available_data.iloc[-1].copy()

                            # Apply scenario modifications
                            if symbol_modifier:
                                if 'volatility_multiplier' in symbol_modifier:
                                    # Increase/decrease volatility
                                    vol_mult = symbol_modifier['volatility_multiplier']
                                    current_row['high'] = current_row['open'] * (1 + vol_mult * np.random.uniform(0.01, 0.05))
                                    current_row['low'] = current_row['open'] * (1 - vol_mult * np.random.uniform(0.01, 0.05))
                                    current_row['close'] = current_row['open'] * (1 + vol_mult * np.random.normal(0, 0.02))

                                if 'trend_modifier' in symbol_modifier:
                                    # Apply trend bias
                                    trend_mod = symbol_modifier['trend_modifier']
                                    current_row['close'] *= (1 + trend_mod)

                                if 'volume_multiplier' in symbol_modifier:
                                    # Modify volume
                                    vol_mult = symbol_modifier['volume_multiplier']
                                    current_row['volume'] *= vol_mult

                            # Convertir a formato esperado por el sistema
                            symbol_data = {
                                'open': float(current_row['open']),
                                'high': float(current_row['high']),
                                'low': float(current_row['low']),
                                'close': float(current_row['close']),
                                'volume': float(current_row['volume']),
                                'timestamp': timestamp
                            }

                            # Agregar indicadores técnicos si existen
                            for col in ['rsi', 'macd', 'macd_signal', 'bollinger_upper', 'bollinger_lower']:
                                if col in current_row.index and not pd.isna(current_row[col]):
                                    symbol_data[col] = float(current_row[col])

                            # Agregar el DataFrame histórico completo para que L2 pueda calcular indicadores
                            symbol_data['historical_data'] = available_data.copy()

                            market_snapshot[symbol] = symbol_data
                            break

            if market_snapshot:
                processed_data[timestamp] = market_snapshot

        self.logger.info(f"✅ Datos históricos preparados con diversidad: {len(processed_data)} snapshots de mercado")
        self.logger.info(f"   Escenarios sintéticos aplicados: {len(synthetic_scenarios)} timestamps modificados")
        return processed_data

    def _generate_synthetic_scenarios(self, timestamps: List, original_data: Dict) -> Dict:
        """
        Generate synthetic market scenarios to increase backtesting diversity
        """
        synthetic_scenarios = {}
        np.random.seed(42)  # For reproducible results

        # Define different market scenarios
        scenarios = {
            'high_volatility': {'volatility_multiplier': 2.0, 'volume_multiplier': 1.5},
            'low_volatility': {'volatility_multiplier': 0.3, 'volume_multiplier': 0.7},
            'bull_trend': {'trend_modifier': 0.02, 'volume_multiplier': 1.2},
            'bear_trend': {'trend_modifier': -0.02, 'volume_multiplier': 1.3},
            'sideways': {'volatility_multiplier': 0.5, 'trend_modifier': 0.0},
            'flash_crash': {'volatility_multiplier': 5.0, 'trend_modifier': -0.1, 'volume_multiplier': 3.0},
            'breakout': {'volatility_multiplier': 3.0, 'trend_modifier': 0.05, 'volume_multiplier': 2.5}
        }

        # Apply scenarios to random timestamps (about 20% of data)
        scenario_timestamps = np.random.choice(timestamps, size=int(len(timestamps) * 0.2), replace=False)

        for timestamp in scenario_timestamps:
            scenario_name = np.random.choice(list(scenarios.keys()))
            scenario_params = scenarios[scenario_name]

            # Apply scenario to all symbols
            symbol_scenarios = {}
            for symbol in original_data.keys():
                # Add some randomness to scenario parameters
                modified_params = {}
                for param, value in scenario_params.items():
                    if isinstance(value, (int, float)):
                        # Add 20% randomness
                        noise = np.random.normal(1.0, 0.2)
                        modified_params[param] = value * noise
                    else:
                        modified_params[param] = value

                symbol_scenarios[symbol] = modified_params

            synthetic_scenarios[timestamp] = symbol_scenarios

        self.logger.info(f"🎭 Generated {len(synthetic_scenarios)} synthetic scenarios: {list(scenarios.keys())}")
        return synthetic_scenarios


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
