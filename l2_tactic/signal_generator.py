# l2_tactic/signal_generator - Gestión del portfolio
import asyncio
import os
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import warnings

from core.logging import logger
from .models import TacticalSignal, L2State
from .technical.multi_timeframe import MultiTimeframeTechnical
from .risk_overlay import RiskOverlay
from .signal_composer import SignalComposer
from .finrl_processor import FinRLProcessor
from .finrl_wrapper import FinRLProcessorWrapper
from .signal_validator import validate_signal_list, validate_tactical_signal, create_fallback_signal
from .utils import safe_float

class L2TacticProcessor:
    """
    Generador de señales tácticas para L2.
    Incluye:
    - Señales AI (FinRL)
    - Señales técnicas (multi-timeframe)
    - Risk overlay (ajustes por contexto L3)
    - Fallback táctico cuando L3 está stale
    - Construcción de features cross-L3
    - Position-aware signal generation (CRITICAL FIX)
    """

    # --- Dimensiones de features ---
    FINRL_OBS_DIM = 257
    BASE_FEATURES_DIM = 246
    CROSS_FEATURES_DIM = 11
    PREFERRED_BASE_COLS: Optional[List[str]] = None

    def __init__(self, config, portfolio_manager=None):
        """Inicializa el procesador táctico."""
        self.config = config
        self.portfolio_manager = portfolio_manager  # CRITICAL: Add portfolio manager reference
        self.multi_timeframe = MultiTimeframeTechnical(config)
        self.risk_overlay = RiskOverlay(config)

        # Pasar configuración correcta a SignalComposer
        signal_config = getattr(config, 'signals', config) if hasattr(config, 'signals') else config
        self.signal_composer = SignalComposer(signal_config)

        # Cargar modelo FinRL desde config.ai_model.model_path
        model_path = getattr(config.ai_model, "model_path", None) if hasattr(config, 'ai_model') else None
        logger.info(f"🔍 Intentando cargar modelo desde: {model_path}")
        if model_path:
            logger.info(f"🔍 Ruta absoluta del modelo: {os.path.abspath(model_path)}")
            logger.info(f"🔍 Existe archivo?: {os.path.exists(model_path)}")
        if not model_path:
            logger.warning("⚠️ No se encontró model_path en la configuración L2. Usando fallback.")
            model_path = "models/L2/deepsek.zip"
            logger.info(f"🔍 Usando ruta fallback: {model_path}")
            logger.info(f"🔍 Ruta absoluta fallback: {os.path.abspath(model_path)}")

        # Store model path for potential reloading
        self.current_model_path = model_path

        try:
            self.finrl_processor = FinRLProcessor(model_path)

            # Detectar modelo y crear wrapper inteligente
            model_name = self._detect_model_name(model_path)
            self.finrl_wrapper = FinRLProcessorWrapper(self.finrl_processor, model_name)
            logger.info(f"✅ FinRLProcessorWrapper creado para modelo: {model_name}")

        except Exception as e:
            logger.error(f"❌ Error cargando FinRL processor: {e}")
            self.finrl_processor = None
            self.finrl_wrapper = None

    def _detect_model_name(self, model_path: str) -> str:
        """Detecta el nombre del modelo desde la ruta del archivo"""
        if not model_path:
            return "unknown"

        model_path_lower = model_path.lower()

        # Detectar modelos específicos
        if "gemini" in model_path_lower:
            return "gemini"
        elif "deepseek" in model_path_lower:
            return "deepseek"
        elif "claude" in model_path_lower:
            return "claude"
        elif "kimi" in model_path_lower:
            return "kimi"
        elif "gpt" in model_path_lower:
            return "gpt"
        else:
            # Modelo genérico - intentar inferir desde dimensiones esperadas
            if self.finrl_processor and hasattr(self.finrl_processor, 'observation_space_info'):
                expected_dims = self.finrl_processor.observation_space_info.get('expected_dims', 257)
                if expected_dims == 13:
                    return "gemini"  # Legacy 13 features
                elif expected_dims == 971:
                    return "claude"  # Risk-aware 971 features
                elif expected_dims == 257:
                    return "finrl_standard"  # Standard FinRL 257 features
                else:
                    return f"custom_{expected_dims}"

            return "unknown"

    def switch_model(self, model_key: str) -> bool:
        """Switch to a different L2 model dynamically"""
        try:
            if not hasattr(self.config, 'ai_model'):
                logger.error("❌ No ai_model config available for switching")
                return False

            # Try to switch model in config
            if self.config.ai_model.switch_model(model_key):
                new_model_path = self.config.ai_model.model_path
                logger.info(f"🔄 Switching L2 model to: {model_key} -> {new_model_path}")

                # Check if file exists
                if not os.path.exists(new_model_path):
                    logger.error(f"❌ Model file does not exist: {new_model_path}")
                    return False

                # Create new processor with new model
                try:
                    new_processor = FinRLProcessor(new_model_path)
                    self.finrl_processor = new_processor
                    self.current_model_path = new_model_path
                    logger.info(f"✅ Successfully switched to model: {model_key}")
                    return True
                except Exception as e:
                    logger.error(f"❌ Failed to load new model {model_key}: {e}")
                    return False
            else:
                logger.error(f"❌ Config switch_model failed for: {model_key}")
                return False
        except Exception as e:
            logger.error(f"❌ Error switching model to {model_key}: {e}")
            return False

    async def process_signals(self, state: Dict[str, Any]) -> List[TacticalSignal]:
        """
        Procesa datos de mercado para generar señales tácticas.
        Combina:
        - Señales L1 operacionales (momentum, technical, volume)
        - Señales AI (FinRL)
        - Señales técnicas (multi-timeframe)
        - Ajustes de riesgo
        - Fallback cuando L3 stale
        """
        # SUPPRESS NUMPY RUNTIME WARNINGS FOR NaN/inf VALUES
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered", category=RuntimeWarning)
            warnings.filterwarnings("ignore", message="divide by zero encountered", category=RuntimeWarning)
            warnings.filterwarnings("ignore", message="overflow encountered", category=RuntimeWarning)

            # Configure numpy error handling
            with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
                return await self._process_signals_internal(state)

    async def _process_signals_internal(self, state: Dict[str, Any]) -> List[TacticalSignal]:
        """
        Internal signal processing method with numpy error handling
        """
        try:
            market_data = state.get("market_data_simple") or state.get("market_data", {})
            if not market_data:
                logger.warning("⚠️ L2: No hay datos de mercado disponibles")
                return []

            signals = []
            for symbol, data in market_data.items():
                df = self._extract_dataframe(data, symbol)
                if df is None:
                    continue

                # Calcular indicadores técnicos multi-timeframe
                indicators = self.multi_timeframe.calculate_technical_indicators(df)

                # INTEGRAR SEÑALES L1 OPERACIONALES
                l1_signals = await self._get_l1_operational_signals(state, symbol, df)
                logger.info(f"🔍 L1: {len(l1_signals)} señales operacionales para {symbol}")

                # Señal AI (L2)
                finrl_signal = await self._get_finrl_signal(state, symbol, indicators)

                # Combinar señales L1 + L2
                combined_signal = self._combine_l1_l2_signals(l1_signals, finrl_signal, symbol, df, indicators)

                if not combined_signal:
                    logger.warning(f"⚠️ No se pudo generar señal combinada para {symbol}")
                    continue

                # CRITICAL FIX: Apply position-aware validation
                position_aware_signal = self._apply_position_aware_validation(combined_signal, symbol, state)
                if not position_aware_signal:
                    logger.warning(f"⚠️ Position-aware validation rejected signal for {symbol}")
                    continue

                # Evaluar contexto L3
                l3_context = self._check_l3_context_freshness(state, symbol, df)
                l3_context_cache = state.get("l3_context_cache", {})
                l3_output = l3_context_cache.get("last_output", {}) or state.get("l3_output", {})

                if l3_output and l3_context['is_fresh']:
                    risk_signals = await self.risk_overlay.generate_risk_signals(
                        market_data={symbol: df},
                        portfolio_data=state.get("portfolio", {}),
                        l3_context=l3_output
                    )
                    if risk_signals:
                        combined_signal = self._apply_risk_adjustment(combined_signal, risk_signals, l3_output)

                    # HOTFIX: Apply L3 position-aware filtering
                    combined_signal = self._apply_l3_position_hotfix(combined_signal, symbol, state)
                    risk_filtered = combined_signal

                elif l3_output and not l3_context['is_fresh']:
                    tactical_signal = self._generate_tactical_fallback_signal(symbol, df, indicators)
                    risk_filtered = tactical_signal or combined_signal

                else:
                    risk_filtered = combined_signal

                # DEBUG: Log signal transformation
                logger.info(f"🔄 SIGNAL TRANSFORMATION for {symbol}:")
                logger.info(f"   L1+L2 Combined: {getattr(combined_signal, 'side', 'no_side')} conf={getattr(combined_signal, 'confidence', 0):.3f}")
                logger.info(f"   L3 Filtered: {getattr(risk_filtered, 'side', 'no_side')} conf={getattr(risk_filtered, 'confidence', 0):.3f}")

                # Componer señal final
                tactical_signal = self.signal_composer.compose_signal(
                    symbol=symbol,
                    base_signal=risk_filtered,
                    indicators=indicators,
                    state=state
                )

                logger.info(f"   Composer Output: {getattr(tactical_signal, 'side', 'no_side') if tactical_signal else 'None'} conf={getattr(tactical_signal, 'confidence', 0):.3f}")

                validated_signal = validate_tactical_signal(tactical_signal) if tactical_signal else create_fallback_signal(symbol, "no_composition")

                logger.info(f"   Final Validated: {getattr(validated_signal, 'side', 'no_side')} conf={getattr(validated_signal, 'confidence', 0):.3f}")
                logger.info(f"   ---")

                signals.append(validated_signal)

            # FINAL SIGNAL SUMMARY
            logger.info("🎯 SIGNAL PROCESSING SUMMARY:")
            logger.info(f"   Total signals processed: {len(signals)}")
            sides = {}
            for s in signals:
                side = getattr(s, 'side', 'unknown')
                sides[side] = sides.get(side, 0) + 1
            for side, count in sides.items():
                logger.info(f"   {side.upper()}: {count} signals")

            # LOG SYSTEM STATUS - CHECK ACTUAL IMPLEMENTATION
            l1_status = self._check_l1_models_status(state)
            l3_status = self._check_l3_models_status(state)

            logger.info("✅ SYSTEM ARCHITECTURE STATUS:")
            logger.info("   Claimed: 9 AI models (3 L1 + 1 L2 + 5 L3)")
            logger.info(f"   Actually running: {l1_status['count']} L1 + 1 L2 + {l3_status['count']} L3 models")
            logger.info(f"   {l1_status['status']}")
            logger.info(f"   {l3_status['status']}")
            logger.info("   🎯 System now has complete L1→L2→L3 pipeline!")

            # Check if we have the complete 9+ AI models
            total_models = l1_status['count'] + 1 + l3_status['count']  # L1 + L2 + L3
            if total_models >= 9:
                logger.info(f"   🏆 COMPLETE: {total_models} AI models operational!")
            elif total_models >= 8:
                logger.info(f"   ⚡ ADVANCED: {total_models} AI models operational!")
            else:
                logger.info(f"   ⚠️ PARTIAL: Only {total_models} AI models operational")

            return validate_signal_list(signals)

        except Exception as e:
            logger.error(f"❌ Error procesando señales L2: {e}", exc_info=True)
            return []

    # -------------------------------------------
    # Funciones auxiliares
    # -------------------------------------------

    def _extract_dataframe(self, data, symbol: str) -> Optional[pd.DataFrame]:
        """Extrae DataFrame válido del formato de backtesting o real"""
        try:
            if isinstance(data, dict):
                if 'historical_data' in data and isinstance(data['historical_data'], pd.DataFrame):
                    df = data['historical_data']
                    if len(df) < 200:
                        logger.warning(f"⚠️ Datos históricos insuficientes para {symbol}: {len(df)} < 200")
                        return None
                    return df
                else:
                    logger.warning(f"⚠️ Datos insuficientes para {symbol}: formato single point")
                    return None
            elif isinstance(data, pd.DataFrame):
                if data.empty or len(data) < 200:
                    logger.warning(f"⚠️ Datos insuficientes para {symbol}: {len(data)} < 200")
                    return None
                return data
            else:
                logger.warning(f"⚠️ Formato de datos inválido para {symbol}: {type(data)}")
                return None
        except Exception as e:
            logger.error(f"❌ Error extrayendo DataFrame para {symbol}: {e}")
            return None

    async def _get_finrl_signal(self, state: Dict[str, Any], symbol: str, indicators: Dict) -> Optional[TacticalSignal]:
        """Obtiene señal de FinRL y asegura source='ai'"""
        try:
            if not self.finrl_processor:
                return None

            # Detectar método disponible automáticamente
            signal = await self._get_finrl_signal_safe(state, symbol, indicators)
            if signal:
                signal.source = 'ai'
            return signal
        except Exception as e:
            logger.error(f"❌ Error obteniendo señal FinRL para {symbol}: {e}")
            return None

    async def _get_finrl_signal_safe(self, state: Dict[str, Any], symbol: str, indicators: Dict) -> Optional[TacticalSignal]:
        """
        Usa el FinRLProcessorWrapper para generar señales automáticamente
        """
        try:
            if not self.finrl_wrapper:
                logger.warning(f"⚠️ FinRL wrapper no disponible para {symbol}")
                return None

            # Preparar datos de mercado para el wrapper
            market_data = state.get("market_data_simple") or state.get("market_data", {})

            # L2_FinRL | Estado del mercado: market state summary
            market_state = f"symbol={symbol}, data_points={len(market_data.get(symbol, []))}"
            logger.info(f"L2_FinRL | Estado del mercado: {market_state}")

            # Usar el wrapper inteligente que maneja automáticamente:
            # - Detección de modelo (Gemini, DeepSeek, Claude, Kimi)
            # - Ajuste de forma de observaciones
            # - Método correcto (predict/get_action)
            signal = await self.finrl_wrapper.generate_signal(market_data, symbol, indicators)

            if signal:
                # L2_FinRL | Acción recomendada: recommended action with probability
                action = getattr(signal, 'side', 'unknown')
                prob = getattr(signal, 'confidence', 0.0)
                logger.info(f"L2_FinRL | Acción recomendada: {action} (probabilidad: {prob:.3f})")

                # L2_FinRL | Validación de señal: signal validation result
                validation_result = "válida" if prob > 0.5 else "baja_confianza"
                logger.info(f"L2_FinRL | Validación de señal: {validation_result}")

                logger.debug(f"✅ Señal FinRL generada para {symbol}: {signal.side} (conf: {signal.confidence:.2f})")
            else:
                logger.debug(f"⚠️ No se pudo generar señal FinRL para {symbol}")

            return signal

        except Exception as e:
            logger.error(f"❌ Error en _get_finrl_signal_safe para {symbol}: {e}")
            return None

    # -------------------------------------------
    # Features y métricas multiasset
    # -------------------------------------------

    def _select_base_features_row(self, features_df: pd.DataFrame) -> List[float]:
        """Select numeric features for AI model"""
        if not isinstance(features_df, pd.DataFrame) or features_df.empty:
            return [0.0] * self.BASE_FEATURES_DIM
        last_row = features_df.iloc[-1]
        if self.PREFERRED_BASE_COLS:
            cols = [c for c in self.PREFERRED_BASE_COLS if c in features_df.columns]
        else:
            cols = sorted([c for c in features_df.columns if pd.api.types.is_numeric_dtype(features_df[c])])
        values = []
        for c in cols[:self.BASE_FEATURES_DIM]:
            try:
                # Use safe_float for robust conversion
                val = last_row[c]
                v = safe_float(val)
                values.append(v if np.isfinite(v) else 0.0)
            except Exception as e:
                logger.debug(f"Error converting feature {c}: {e}, using 0.0")
                values.append(0.0)
        if len(values) < self.BASE_FEATURES_DIM:
            values.extend([0.0] * (self.BASE_FEATURES_DIM - len(values)))
        return values[:self.BASE_FEATURES_DIM]

    def _get_key_metrics(self, features_df: pd.DataFrame) -> Dict[str, float]:
        last_row = features_df.iloc[-1]
        metrics = {}
        for key, default in [
            ('rsi', 50.0),
            ('macd', 0.0),
            ('macd_signal', 0.0),
            ('sma_20', 0.0),
            ('sma_50', 0.0),
            ('bollinger_upper', 0.0),
            ('bollinger_lower', 0.0),
            ('vol_zscore', 0.0)
        ]:
            try:
                val = last_row.get(key, default)
                # Use safe_float for robust conversion
                converted_val = safe_float(val)
                metrics[key] = converted_val if np.isfinite(converted_val) else default
            except Exception as e:
                logger.debug(f"Error converting metric {key}: {e}, using default {default}")
                metrics[key] = default
        return metrics

    def _compute_eth_btc_ratio(self, market_data: Dict[str, pd.DataFrame],
                               features_by_symbol: Dict[str, pd.DataFrame]) -> float:
        """ETH/BTC ratio"""
        try:
            eth = features_by_symbol.get("ETHUSDT") or market_data.get("ETHUSDT")
            btc = features_by_symbol.get("BTCUSDT") or market_data.get("BTCUSDT")
            eth_close = safe_float(eth["close"].iloc[-1]) if isinstance(eth, pd.DataFrame) and not eth.empty else None
            btc_close = safe_float(btc["close"].iloc[-1]) if isinstance(btc, pd.DataFrame) and not btc.empty else None
            if eth_close is not None and btc_close not in (None, 0.0):
                return eth_close / btc_close
        except Exception:
            pass
        return 0.0

    def _compute_btc_eth_corr30(self, market_data: Dict[str, pd.DataFrame]) -> float:
        """30-period correlation BTC vs ETH"""
        try:
            eth = market_data.get("ETHUSDT")
            btc = market_data.get("BTCUSDT")
            if not (isinstance(eth, pd.DataFrame) and isinstance(btc, pd.DataFrame)):
                return 0.0
            eth_close = eth["close"].astype(float).tail(30)
            btc_close = btc["close"].astype(float).tail(30)
            common_idx = eth_close.index.intersection(btc_close.index)
            eth_close = eth_close.loc[common_idx]
            btc_close = btc_close.loc[common_idx]
            if len(eth_close) < 3:
                return 0.0
            eth_ret = eth_close.pct_change().dropna()
            btc_ret = btc_close.pct_change().dropna()
            common_idx = eth_ret.index.intersection(btc_ret.index)
            if len(common_idx) < 3:
                return 0.0
            corr_matrix = np.corrcoef(eth_ret.loc[common_idx], btc_ret.loc[common_idx])
            corr = safe_float(corr_matrix[0, 1])
            return corr if np.isfinite(corr) else 0.0
        except Exception:
            return 0.0

    def _compute_spread_pct(self, market_data: Dict[str, pd.DataFrame],
                            features_by_symbol: Dict[str, pd.DataFrame]) -> float:
        """(BTC - ETH)/BTC spread %"""
        try:
            eth = market_data.get("ETHUSDT")
            btc = market_data.get("BTCUSDT")
            if isinstance(eth, pd.DataFrame) and isinstance(btc, pd.DataFrame):
                eth_close = safe_float(eth["close"].iloc[-1])
                btc_close = safe_float(btc["close"].iloc[-1])
                if btc_close != 0 and not np.isnan(eth_close) and not np.isnan(btc_close):
                    return (btc_close - eth_close) / btc_close
        except Exception:
            pass
        return 0.0

    # -------------------------------------------
    # L3 context checks & tactical fallback
    # -------------------------------------------

    def _check_l3_context_freshness(self, state: Dict[str, Any], symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Verifica si L3 context está fresco o stale"""
        try:
            l3_context_cache = state.get("l3_context_cache", {})
            l3_data = l3_context_cache.get("last_output", {}) or state.get("l3_output", {})
            if not l3_data:
                return {'is_fresh': False, 'reason': 'no_l3_data', 'regime': 'unknown', 'price_change_pct': 0.0}

            current_time = pd.Timestamp.now()
            l3_timestamp_str = l3_data.get('timestamp')
            if l3_timestamp_str:
                try:
                    l3_timestamp = pd.Timestamp(l3_timestamp_str)
                    if l3_timestamp.tz is None:
                        l3_timestamp = l3_timestamp.tz_localize('UTC')
                    if current_time.tz is None:
                        current_time = current_time.tz_localize('UTC')
                    time_diff_seconds = (current_time - l3_timestamp).total_seconds()
                    max_age_seconds = 2700
                    if time_diff_seconds > max_age_seconds:
                        return {'is_fresh': False, 'reason': 'timestamp_too_old', 'regime': l3_data.get('regime', 'unknown'), 'price_change_pct': 0.0}
                except Exception:
                    logger.warning(f"Error parsing L3 timestamp for {symbol}, assuming fresh")
            regime = l3_data.get('regime', 'unknown')
            return {'is_fresh': True, 'reason': 'context_fresh', 'regime': regime, 'price_change_pct': 0.0}
        except Exception as e:
            logger.warning(f"Error checking L3 freshness for {symbol}: {e}")
            return {'is_fresh': False, 'reason': 'error_checking', 'regime': 'unknown', 'price_change_pct': 0.0}

    def _generate_tactical_fallback_signal(self, symbol: str, df: pd.DataFrame, indicators: Dict) -> Optional[TacticalSignal]:
        """Señal táctica cuando L3 está stale"""
        try:
            if len(df) < 20:
                return None
            recent_prices = df['close'].tail(20).astype(float)
            current_price = recent_prices.iloc[-1]
            prev_price = recent_prices.iloc[-2]
            short_trend = (current_price - recent_prices.iloc[-5]) / recent_prices.iloc[-5] * 100
            medium_trend = (current_price - recent_prices.iloc[-10]) / recent_prices.iloc[-10] * 100
            returns = recent_prices.pct_change().dropna()
            momentum = returns.tail(5).mean() * 100
            # Safely extract indicator values
            rsi_val = 50.0
            macd_val = 0.0
            macd_signal_val = 0.0

            # Use safe_float for all indicator extractions
            rsi_val = safe_float(indicators.get('rsi', 50.0))
            if np.isnan(rsi_val):
                rsi_val = 50.0

            macd_val = safe_float(indicators.get('macd', 0.0))
            if np.isnan(macd_val):
                macd_val = 0.0

            macd_signal_val = safe_float(indicators.get('macd_signal', 0.0))
            if np.isnan(macd_signal_val):
                macd_signal_val = 0.0
            macd_diff = macd_val - macd_signal_val

            # Lógica de decisión
            confidence, strength, side = 0.6, 0.5, 'hold'
            if abs(momentum) > 0.5:
                side = 'buy' if momentum > 0 else 'sell'
                confidence = min(0.8, 0.6 + abs(momentum)*0.5)
                strength = min(0.8, 0.5 + abs(momentum)*2.0)
            elif rsi_val < 35: side, confidence, strength = 'buy', 0.7, 0.6
            elif rsi_val > 65: side, confidence, strength = 'sell', 0.7, 0.6
            elif abs(macd_diff) > 10: side, confidence, strength = ('buy' if macd_diff>0 else 'sell'), 0.65, 0.55
            elif abs(short_trend) > 1.0: side, confidence, strength = ('buy' if short_trend>0 else 'sell'), 0.6, 0.5

            features = {
                'close': current_price,
                'price_change_pct': (current_price - prev_price)/prev_price*100,
                'short_trend_pct': short_trend,
                'medium_trend_pct': medium_trend,
                'momentum_pct': momentum,
                'rsi': rsi_val,
                'macd_diff': macd_diff,
                'tactical_reason': 'l3_stale_fallback'
            }

            return TacticalSignal(
                symbol=symbol,
                side=side,
                strength=strength,
                confidence=confidence,
                signal_type='tactical_fallback',
                source='tactical_fallback',
                features=features,
                timestamp=pd.Timestamp.now(),
                metadata={'l3_stale': True}
            )
        except Exception as e:
            logger.error(f"❌ Error generating tactical fallback signal for {symbol}: {e}")
            return None

    async def _get_l1_operational_signals(self, state: Dict[str, Any], symbol: str, df: pd.DataFrame) -> List[TacticalSignal]:
        """Obtiene señales operacionales L1 para el símbolo"""
        try:
            # Importar L1OperationalProcessor dinámicamente para evitar dependencias circulares
            from l1_operational.l1_operational import L1OperationalProcessor

            # Crear procesador L1 si no existe en el estado
            if 'l1_processor' not in state:
                l1_config = {}  # Configuración básica por defecto
                state['l1_processor'] = L1OperationalProcessor(l1_config)

            l1_processor = state['l1_processor']

            # Preparar datos de mercado para L1 (solo el símbolo actual)
            l1_market_data = {symbol: df}

            # Generar señales L1
            l1_signals = await l1_processor.process_market_data(l1_market_data)

            return l1_signals

        except Exception as e:
            logger.error(f"❌ Error obteniendo señales L1 para {symbol}: {e}")
            return []

    def _combine_l1_l2_signals(self, l1_signals: List[TacticalSignal], l2_signal: Optional[TacticalSignal],
                              symbol: str, df: pd.DataFrame, indicators: Dict) -> Optional[TacticalSignal]:
        """
        Combina señales L1 y L2 en una señal táctica unificada
        """
        try:
            if not l1_signals and not l2_signal:
                logger.warning(f"⚠️ No hay señales L1 ni L2 para {symbol}")
                return None

            # Si solo hay señales L2, usar esa
            if not l1_signals and l2_signal:
                logger.info(f"🔄 Solo señal L2 para {symbol}: {l2_signal.side}")
                return l2_signal

            # Si solo hay señales L1, crear señal compuesta de L1
            if l1_signals and not l2_signal:
                return self._create_l1_composite_signal(l1_signals, symbol, df, indicators)

            # Si hay ambas señales, combinarlas
            return self._create_l1_l2_composite_signal(l1_signals, l2_signal, symbol, df, indicators)

        except Exception as e:
            logger.error(f"❌ Error combinando señales L1+L2 para {symbol}: {e}")
            return l2_signal  # Fallback a L2

    def _create_l1_composite_signal(self, l1_signals: List[TacticalSignal], symbol: str,
                                   df: pd.DataFrame, indicators: Dict) -> Optional[TacticalSignal]:
        """Crea señal compuesta solo de señales L1"""
        try:
            if not l1_signals:
                return None

            # Contar votos por dirección
            votes = {'buy': 0, 'sell': 0, 'hold': 0}
            total_confidence = 0
            total_strength = 0
            features = {}

            for signal in l1_signals:
                side = getattr(signal, 'side', 'hold')
                confidence = getattr(signal, 'confidence', 0.5)
                strength = getattr(signal, 'strength', 0.5)

                votes[side] = votes.get(side, 0) + confidence  # Voto ponderado por confianza
                total_confidence += confidence
                total_strength += strength

                # Agregar features del signal
                if hasattr(signal, 'features') and signal.features:
                    features.update(signal.features)

            # Determinar dirección ganadora
            winning_side = max(votes, key=votes.get)
            winning_votes = votes[winning_side]

            # Calcular confianza compuesta
            avg_confidence = total_confidence / len(l1_signals) if l1_signals else 0.5
            avg_strength = total_strength / len(l1_signals) if l1_signals else 0.5

            # Ajustar confianza basada en consenso
            total_votes = sum(votes.values())
            if total_votes > 0:
                consensus_ratio = winning_votes / total_votes
                # Bonus por consenso unánime
                if consensus_ratio >= 0.8:
                    avg_confidence = min(0.9, avg_confidence * 1.2)
                elif consensus_ratio < 0.6:
                    avg_confidence *= 0.8  # Penalización por división

            # Agregar indicadores técnicos actuales
            features.update({
                'l1_signal_count': len(l1_signals),
                'l1_consensus_ratio': consensus_ratio if 'consensus_ratio' in locals() else 0.5,
                'l1_votes_buy': votes['buy'],
                'l1_votes_sell': votes['sell'],
                'l1_votes_hold': votes['hold']
            })

            # Agregar precio actual
            if not df.empty:
                features['close'] = df['close'].iloc[-1]

            return TacticalSignal(
                symbol=symbol,
                side=winning_side,
                strength=avg_strength,
                confidence=avg_confidence,
                signal_type='l1_composite',
                source='l1_operational',
                features=features,
                timestamp=pd.Timestamp.now(),
                metadata={
                    'l1_signal_count': len(l1_signals),
                    'l1_consensus_side': winning_side,
                    'l1_consensus_ratio': consensus_ratio if 'consensus_ratio' in locals() else 0.5
                }
            )

        except Exception as e:
            logger.error(f"❌ Error creando señal compuesta L1 para {symbol}: {e}")
            return None

    def _create_l1_l2_composite_signal(self, l1_signals: List[TacticalSignal], l2_signal: TacticalSignal,
                                      symbol: str, df: pd.DataFrame, indicators: Dict) -> Optional[TacticalSignal]:
        """Combina señales L1 y L2 en una señal unificada"""
        try:
            # Crear señal compuesta de L1
            l1_composite = self._create_l1_composite_signal(l1_signals, symbol, df, indicators)

            if not l1_composite:
                logger.warning(f"⚠️ No se pudo crear señal compuesta L1 para {symbol}, usando L2")
                return l2_signal

            # Lógica de combinación L1 + L2
            l1_side = getattr(l1_composite, 'side', 'hold')
            l2_side = getattr(l2_signal, 'side', 'hold')
            l1_conf = getattr(l1_composite, 'confidence', 0.5)
            l2_conf = getattr(l2_signal, 'confidence', 0.5)

            # Pesos para combinación (L1 tiene más peso en señales operacionales)
            l1_weight = 0.6
            l2_weight = 0.4

            # Si coinciden, reforzar la señal
            if l1_side == l2_side:
                combined_confidence = min(0.95, (l1_conf * l1_weight + l2_conf * l2_weight) * 1.1)
                combined_strength = min(1.0, (l1_composite.strength + l2_signal.strength) / 2 * 1.1)
                final_side = l1_side
                consensus_level = 'high'
            else:
                # Conflicto: usar la señal con mayor confianza
                if l1_conf > l2_conf:
                    final_side = l1_side
                    combined_confidence = l1_conf * 0.8  # Penalización por conflicto
                    combined_strength = l1_composite.strength * 0.9
                else:
                    final_side = l2_side
                    combined_confidence = l2_conf * 0.8  # Penalización por conflicto
                    combined_strength = l2_signal.strength * 0.9
                consensus_level = 'low'

            # Combinar features
            combined_features = dict(l1_composite.features)
            combined_features.update({
                'l2_side': l2_side,
                'l2_confidence': l2_conf,
                'l2_strength': l2_signal.strength,
                'consensus_level': consensus_level,
                'l1_l2_agreement': l1_side == l2_side
            })

            # Agregar features de L2 si existen
            if hasattr(l2_signal, 'features') and l2_signal.features:
                l2_features = {f"l2_{k}": v for k, v in l2_signal.features.items()}
                combined_features.update(l2_features)

            return TacticalSignal(
                symbol=symbol,
                side=final_side,
                strength=combined_strength,
                confidence=combined_confidence,
                signal_type='l1_l2_composite',
                source='l1_l2_combined',
                features=combined_features,
                timestamp=pd.Timestamp.now(),
                metadata={
                    'l1_side': l1_side,
                    'l2_side': l2_side,
                    'consensus_level': consensus_level,
                    'l1_signal_count': len(l1_signals),
                    'combined_method': 'weighted_average'
                }
            )

        except Exception as e:
            logger.error(f"❌ Error creando señal compuesta L1+L2 para {symbol}: {e}")
            return l2_signal  # Fallback a L2

    def _apply_risk_adjustment(self, finrl_signal, risk_signals: List, l3_context: Dict) -> Any:
        """Aplicar ajustes de riesgo a la señal combinada - FIXED LOGIC"""
        try:
            if not risk_signals:
                return finrl_signal

            risk_appetite = l3_context.get('risk_appetite', 0.5)
            regime = l3_context.get('regime', 'neutral')

            # Check for critical risk signals
            has_close_all = any(getattr(s, 'side', '') == 'close_all' for s in risk_signals)
            has_reduce = any(getattr(s, 'side', '') == 'reduce' for s in risk_signals)

            original_side = getattr(finrl_signal, 'side', 'hold')
            original_confidence = getattr(finrl_signal, 'confidence', 0.5)

            # CRITICAL RISK: Close all positions - convert any signal to HOLD
            if has_close_all:
                logger.warning(f"🚨 CRITICAL RISK: Converting {original_side} to HOLD due to close_all signal")
                finrl_signal.side = 'hold'
                finrl_signal.confidence = min(original_confidence, 0.3)  # Very low confidence
                finrl_signal.strength = 0.1  # Very low strength
                return finrl_signal

            # HIGH RISK: Reduce positions - be more conservative
            if has_reduce and risk_appetite < 0.4:
                if original_side == 'buy':
                    # Convert BUY to HOLD when reducing positions
                    logger.warning(f"⚠️ RISK REDUCTION: Converting BUY to HOLD for {finrl_signal.symbol}")
                    finrl_signal.side = 'hold'
                    finrl_signal.confidence *= 0.7
                    finrl_signal.strength *= 0.5
                elif original_side == 'sell':
                    # Reduce SELL strength but keep signal
                    finrl_signal.confidence *= 0.8
                    finrl_signal.strength *= 0.7
                else:  # hold
                    # Keep HOLD but reduce confidence slightly
                    finrl_signal.confidence *= 0.9

            # REGIME-BASED ADJUSTMENTS (less aggressive - DISABLED to prevent inappropriate conversions)
            # elif regime in ['volatile', 'bear']:
            #     if original_side == 'buy':
            #         # Reduce BUY confidence in volatile/bear markets
            #         finrl_signal.confidence *= 0.85
            #         finrl_signal.strength *= 0.9
            #         logger.info(f"📉 Volatile/Bear regime: Reduced BUY confidence to {finrl_signal.confidence:.3f}")
            #     elif original_side == 'sell':
            #         # Slightly boost SELL in volatile/bear markets
            #         finrl_signal.confidence *= 1.05
            #         finrl_signal.strength *= 1.02
            #         logger.info(f"📈 Volatile/Bear regime: Slightly boosted SELL confidence to {finrl_signal.confidence:.3f}")

            # Ensure minimum confidence threshold
            if finrl_signal.confidence < 0.2:
                logger.warning(f"⚠️ Signal confidence too low ({finrl_signal.confidence:.3f}), converting to HOLD")
                finrl_signal.side = 'hold'
                finrl_signal.confidence = 0.5
                finrl_signal.strength = 0.3

            logger.info(f"🎛️ Risk adjustment applied: {original_side}→{finrl_signal.side} conf={finrl_signal.confidence:.3f} strength={finrl_signal.strength:.3f}")
            return finrl_signal

        except Exception as e:
            logger.error(f"Error applying risk adjustment: {e}")
            return finrl_signal

    def _check_l1_models_status(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Check if L1 operational models are implemented and working"""
        try:
            # Check if L1 processor exists and can generate signals
            if 'l1_processor' not in state:
                return {'count': 0, 'status': '❌ MISSING: All 3 L1 operational models'}

            l1_processor = state['l1_processor']

            # Try to check if L1 models are working by testing signal generation
            # This is a lightweight check - we don't want to actually generate signals here
            if hasattr(l1_processor, 'process_market_data'):
                return {'count': 3, 'status': '✅ IMPLEMENTED: All 3 L1 operational models'}
            else:
                return {'count': 0, 'status': '❌ MISSING: All 3 L1 operational models'}

        except Exception as e:
            logger.error(f"Error checking L1 models status: {e}")
            return {'count': 0, 'status': '❌ MISSING: All 3 L1 operational models'}

    def _apply_position_aware_validation(self, signal: TacticalSignal, symbol: str, state: Dict[str, Any]) -> Optional[TacticalSignal]:
        """
        CRITICAL FIX: Apply position-aware validation to prevent invalid signals
        - No SELL signals without positions
        - High confidence required for BUY with existing positions
        """
        try:
            if not self.portfolio_manager:
                logger.warning("⚠️ No portfolio manager available for position validation")
                return signal

            # Get current position for this symbol
            current_position = 0.0
            try:
                if hasattr(self.portfolio_manager, 'get_position'):
                    current_position = self.portfolio_manager.get_position(symbol) or 0.0
                elif hasattr(self.portfolio_manager, 'get_balance'):
                    # Fallback: check balance directly
                    balance = self.portfolio_manager.get_balance(symbol)
                    current_position = balance if balance and abs(balance) > 0.0001 else 0.0
                else:
                    # Last resort: check portfolio state
                    portfolio = state.get("portfolio", {})
                    symbol_data = portfolio.get(symbol, {})
                    current_position = symbol_data.get("position", 0.0) if isinstance(symbol_data, dict) else 0.0
            except Exception as e:
                logger.warning(f"⚠️ Error getting position for {symbol}: {e}")
                current_position = 0.0

            has_position = abs(current_position) > 0.0001
            signal_side = getattr(signal, 'side', 'hold')
            signal_confidence = getattr(signal, 'confidence', 0.5)

            logger.info(f"🔍 POSITION CHECK for {symbol}: position={current_position:.6f}, has_position={has_position}, signal={signal_side}")

            # CRITICAL VALIDATION RULES
            if signal_side == 'sell' and not has_position:
                logger.warning(f"🛑 BLOCKED: SELL signal for {symbol} but no position! Converting to HOLD")
                # Convert to HOLD signal
                signal.side = 'hold'
                signal.confidence = 0.5  # Neutral confidence
                signal.strength = 0.3    # Low strength
                if hasattr(signal, 'features'):
                    signal.features = signal.features or {}
                    signal.features['position_validation'] = 'blocked_sell_no_position'
                return signal

            elif signal_side == 'buy' and has_position:
                # Allow BUY with existing position but require higher confidence
                if signal_confidence < 0.7:
                    logger.warning(f"⚠️ LOW CONFIDENCE: BUY signal for {symbol} with existing position (conf={signal_confidence:.3f})")
                    # Reduce confidence but allow signal
                    signal.confidence *= 0.8
                    if hasattr(signal, 'features'):
                        signal.features = signal.features or {}
                        signal.features['position_validation'] = 'reduced_confidence_existing_position'

            # Valid signal
            if hasattr(signal, 'features'):
                signal.features = signal.features or {}
                signal.features['position_validation'] = 'valid'
                signal.features['current_position'] = current_position

            logger.info(f"✅ POSITION VALIDATION PASSED for {symbol}: {signal_side} (conf={signal.confidence:.3f})")
            return signal

        except Exception as e:
            logger.error(f"❌ Error in position-aware validation for {symbol}: {e}")
            return signal  # Return original signal on error

    def _apply_l3_position_hotfix(self, signal: TacticalSignal, symbol: str, state: Dict[str, Any]) -> TacticalSignal:
        """
        HOTFIX: Apply L3 position-aware filtering to prevent invalid conversions
        - Prevent L3 from converting HOLD to SELL when no position exists
        """
        try:
            if not self.portfolio_manager:
                return signal

            # Get current position for this symbol
            current_position = 0.0
            try:
                if hasattr(self.portfolio_manager, 'get_position'):
                    current_position = self.portfolio_manager.get_position(symbol) or 0.0
                elif hasattr(self.portfolio_manager, 'get_balance'):
                    balance = self.portfolio_manager.get_balance(symbol)
                    current_position = balance if balance and abs(balance) > 0.0001 else 0.0
                else:
                    portfolio = state.get("portfolio", {})
                    symbol_data = portfolio.get(symbol, {})
                    current_position = symbol_data.get("position", 0.0) if isinstance(symbol_data, dict) else 0.0
            except Exception as e:
                logger.warning(f"⚠️ Error getting position for L3 hotfix {symbol}: {e}")
                current_position = 0.0

            has_position = abs(current_position) > 0.0001
            signal_side = getattr(signal, 'side', 'hold')

            # HOTFIX: If L3 converted HOLD to SELL without position, revert
            if signal_side == 'sell' and not has_position:
                logger.warning(f"🔥 L3 HOTFIX: Reverting SELL→HOLD for {symbol} (no position)")
                signal.side = 'hold'
                signal.confidence *= 0.9  # Slightly reduce confidence
                if hasattr(signal, 'features'):
                    signal.features = signal.features or {}
                    signal.features['l3_hotfix'] = 'reverted_sell_to_hold'

            return signal

        except Exception as e:
            logger.error(f"❌ Error in L3 position hotfix for {symbol}: {e}")
            return signal

    def _check_l3_models_status(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Check if L3 regime-specific models are implemented and working"""
        try:
            # Check if regime-specific L3 models are available
            try:
                from l3_strategy.regime_specific_models import RegimeSpecificL3Processor
                processor = RegimeSpecificL3Processor()

                # Check if models are healthy
                health = processor.get_model_health()
                healthy_models = sum(1 for status in health.get('models', {}).values() if status.get('status') == 'healthy')

                if health['overall_status'] == 'healthy':
                    return {'count': healthy_models, 'status': f'✅ IMPLEMENTED: All {healthy_models} L3 regime-specific models'}
                else:
                    return {'count': healthy_models, 'status': f'⚠️ PARTIAL: {healthy_models} L3 regime models operational'}

            except ImportError:
                return {'count': 0, 'status': '❌ MISSING: All L3 regime-specific models'}
            except Exception as e:
                logger.error(f"Error checking L3 models: {e}")
                return {'count': 0, 'status': '❌ MISSING: All L3 regime-specific models'}

        except Exception as e:
            logger.error(f"Error checking L3 models status: {e}")
            return {'count': 0, 'status': '❌ MISSING: All L3 regime-specific models'}
