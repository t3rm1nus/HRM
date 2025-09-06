# l2_tactic/signal_generator.py
import asyncio
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from core.logging import logger
from .models import TacticalSignal, L2State
from .technical.multi_timeframe import MultiTimeframeTechnical
from .risk_overlay import RiskOverlay
from .signal_composer import SignalComposer
from .finrl_integration import FinRLProcessor

class L2TacticProcessor:
    """
    Generador de señales tácticas para L2.
    ARREGLADO: Usa FinRLProcessor correctamente integrado.
    ARREGLADO: Manejo robusto de DataFrames de features.
    ARREGLADO: Construcción de observación (1, 63) para FinRL con cross-features.
    ARREGLADO: Usa L2State para state['l2'] en lugar de dict.
    """

    # --- Configuración del vector esperado por FinRL ---
    FINRL_OBS_DIM = 63
    BASE_FEATURES_DIM = 52       # features base por símbolo (p.ej. generate_features del repo HRM)
    CROSS_FEATURES_DIM = 11      # cross/L3 para completar hasta 63

    # Si tienes un orden de columnas “canónico” del entrenamiento, defínelo aquí.
    # Si no, tomamos las columnas numéricas disponibles en orden estable.
    PREFERRED_BASE_COLS: Optional[List[str]] = None  # e.g.: ["price_rsi", "price_macd", ...]  (52 exactas)

    def __init__(self, config: dict):
        self.config = config
        self.technical = MultiTimeframeTechnical(config)
        self.risk = RiskOverlay(config)
        self.signal_composer = SignalComposer(config)
        # Case-insensitive key check
        finrl_config = None
        for key in config:
            if key.lower() == "finrl_config":
                finrl_config = config[key]
                break
        if not finrl_config:
            logger.error(f"Available config keys: {list(config.keys())}")
            raise ValueError("No FINRL_CONFIG found in config")
        model_path = finrl_config.get("model_path")
        if not model_path:
            raise ValueError("No model_path specified in FINRL_CONFIG")
        self.finrl_model = FinRLProcessor(model_path)
        logger.info("🎯 L2TacticProcessor inicializado correctamente")

    # ------------------- UTILIDADES FINRL -------------------

    def _select_base_features_row(self, features_df: pd.DataFrame) -> List[float]:
        """
        Selecciona hasta 52 features numéricos de la última fila del DataFrame de features.
        - Si PREFERRED_BASE_COLS está definida y existen en el DF, usa ese orden.
        - Si no, usa columnas numéricas en orden estable (por nombre).
        - Rellena con ceros si hay menos de 52.
        """
        if not isinstance(features_df, pd.DataFrame) or features_df.empty:
            return [0.0] * self.BASE_FEATURES_DIM

        last_row = features_df.iloc[-1]

        # Determinar columnas candidatas
        if self.PREFERRED_BASE_COLS:
            cols = [c for c in self.PREFERRED_BASE_COLS if c in features_df.columns]
        else:
            # columnas numéricas disponibles (ordenadas por nombre para estabilidad)
            numeric_cols = [
                c for c in features_df.columns
                if pd.api.types.is_numeric_dtype(features_df[c])
            ]
            cols = sorted(numeric_cols)

        # Extraer valores
        values = []
        for c in cols[:self.BASE_FEATURES_DIM]:
            try:
                v = last_row[c]
                values.append(float(v) if np.isfinite(v) else 0.0)
            except Exception:
                values.append(0.0)

        # Rellenar hasta 52
        if len(values) < self.BASE_FEATURES_DIM:
            values.extend([0.0] * (self.BASE_FEATURES_DIM - len(values)))

        # Cortar por si acaso
        return values[:self.BASE_FEATURES_DIM]

    def _get_key_metrics(self, features_df: pd.DataFrame) -> Dict[str, float]:
        """Get key technical indicators and metrics from features"""
        last_row = features_df.iloc[-1]
        return {
            'rsi': float(last_row.get('rsi', 50.0)),
            'macd': float(last_row.get('macd', 0.0)),
            'macd_signal': float(last_row.get('macd_signal', 0.0)),
            'sma_20': float(last_row.get('sma_20', 0.0)),
            'sma_50': float(last_row.get('sma_50', 0.0)),
            'bollinger_upper': float(last_row.get('bollinger_upper', 0.0)),
            'bollinger_lower': float(last_row.get('bollinger_lower', 0.0)),
            'vol_zscore': float(last_row.get('vol_zscore', 0.0))
        }

    def _compute_eth_btc_ratio(self, market_data: Dict[str, pd.DataFrame],
                               features_by_symbol: Dict[str, pd.DataFrame]) -> float:
        """
        ETH/BTC close ratio con fallback.
        """
        try:
            # Prioriza features (si contienen 'close'), si no recurre a market_data
            for src in ("features", "market"):
                if src == "features":
                    eth = features_by_symbol.get("ETHUSDT")
                    btc = features_by_symbol.get("BTCUSDT")
                    eth_close = float(eth["close"].iloc[-1]) if isinstance(eth, pd.DataFrame) and "close" in eth.columns and not eth.empty else None
                    btc_close = float(btc["close"].iloc[-1]) if isinstance(btc, pd.DataFrame) and "close" in btc.columns and not btc.empty else None
                else:
                    eth = market_data.get("ETHUSDT")
                    btc = market_data.get("BTCUSDT")
                    eth_close = float(eth["close"].iloc[-1]) if isinstance(eth, pd.DataFrame) and not eth.empty else None
                    btc_close = float(btc["close"].iloc[-1]) if isinstance(btc, pd.DataFrame) and not btc.empty else None

                if eth_close is not None and btc_close not in (None, 0.0):
                    return eth_close / btc_close
        except Exception:
            pass
        return 0.0

    def _compute_btc_eth_corr30(self, market_data: Dict[str, pd.DataFrame]) -> float:
        """
        Correlación de 30 muestras entre returns de close BTC y ETH.
        Si no hay suficientes datos, devuelve 0.0.
        """
        try:
            eth = market_data.get("ETHUSDT")
            btc = market_data.get("BTCUSDT")
            if not (isinstance(eth, pd.DataFrame) and isinstance(btc, pd.DataFrame)):
                return 0.0
            if eth.empty or btc.empty or "close" not in eth.columns or "close" not in btc.columns:
                return 0.0
            # Alinear las últimas 30 muestras por índice si es posible
            n = 30
            eth_close = eth["close"].astype(float).tail(n)
            btc_close = btc["close"].astype(float).tail(n)
            # si no coinciden longitudes tras tail, reindexa por la intersección del índice
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
            corr = float(np.corrcoef(eth_ret.loc[common_idx], btc_ret.loc[common_idx])[0, 1])
            if np.isfinite(corr):
                return corr
        except Exception:
            pass
        return 0.0

    def _compute_spread_pct(self, market_data: Dict[str, pd.DataFrame],
                            features_by_symbol: Dict[str, pd.DataFrame]) -> float:
        """
        (BTC - ETH*ratio_ref)/BTC como proxy simple de spread en %.
        Aquí usamos (BTC - ETH)/BTC para tener algo informativo aunque no homogéneo de unidades.
        """
        try:
            # preferir market_data para homogeneidad
            eth = market_data.get("ETHUSDT")
            btc = market_data.get("BTCUSDT")
            if isinstance(eth, pd.DataFrame) and not eth.empty and isinstance(btc, pd.DataFrame) and not btc.empty:
                eth_close = float(eth["close"].iloc[-1])
                btc_close = float(btc["close"].iloc[-1])
                if btc_close != 0:
                    return (btc_close - eth_close) / btc_close
        except Exception:
            pass
        return 0.0

    def _build_cross_l3_features(self,
                                 market_data: Dict[str, pd.DataFrame],
                                 features_by_symbol: Dict[str, pd.DataFrame]) -> List[float]:
        """
        Construye 11 features cross/L3. Implementa algunos reales y el resto placeholders.
        Orden propuesto (estable):
          0: eth_btc_ratio
          1: btc_eth_corr30
          2: spread_pct
          3: l3_regime               (placeholder 0.0 si no disponible)
          4: l3_risk_appetite        (placeholder 0.5 por defecto)
          5: l3_alloc_BTC            (placeholder 0.0)
          6: l3_alloc_ETH            (placeholder 0.0)
          7: l3_alloc_CASH           (placeholder 0.0)
          8: cross_vol_ratio         (vol ETH / vol BTC últimas 20) si se puede
          9: cross_vol_corr20        (corr 20 returns de volumen) si se puede
          10: cross_momentum_spread  (MACD_BTC - MACD_ETH) si columnas existen
        """
        feats = []

        # 0: ETH/BTC ratio
        feats.append(self._compute_eth_btc_ratio(market_data, features_by_symbol))

        # 1: Correlación 30 sobre returns de close
        feats.append(self._compute_btc_eth_corr30(market_data))

        # 2: Spread %
        feats.append(self._compute_spread_pct(market_data, features_by_symbol))

        # 3–7: L3 placeholders (si tienes estos en features, puedes mapearlos aquí)
        def pick_feature(df_map: Dict[str, pd.DataFrame], key: str, default: float) -> float:
            try:
                btc_df = df_map.get("BTCUSDT")
                if isinstance(btc_df, pd.DataFrame) and key in btc_df.columns and not btc_df.empty:
                    v = float(btc_df[key].iloc[-1])
                    return v if np.isfinite(v) else default
            except Exception:
                pass
            return default

        feats.append(pick_feature(features_by_symbol, "l3_regime", 0.0))         # 3
        feats.append(pick_feature(features_by_symbol, "l3_risk_appetite", 0.5))  # 4
        feats.append(pick_feature(features_by_symbol, "l3_alloc_BTC", 0.0))      # 5
        feats.append(pick_feature(features_by_symbol, "l3_alloc_ETH", 0.0))      # 6
        feats.append(pick_feature(features_by_symbol, "l3_alloc_CASH", 0.0))     # 7

        # 8–9: Cross volumen
        try:
            eth = market_data.get("ETHUSDT")
            btc = market_data.get("BTCUSDT")
            if isinstance(eth, pd.DataFrame) and isinstance(btc, pd.DataFrame):
                n = 20
                v_eth = eth["volume"].astype(float).tail(n)
                v_btc = btc["volume"].astype(float).tail(n)
                common_idx = v_eth.index.intersection(v_btc.index)
                v_eth = v_eth.loc[common_idx]
                v_btc = v_btc.loc[common_idx]
                # 8: ratio medias
                ratio = float(v_eth.mean() / v_btc.mean()) if v_btc.mean() != 0 else 0.0
                feats.append(ratio if np.isfinite(ratio) else 0.0)
                # 9: correlación returns de volumen
                v_eth_ret = v_eth.pct_change().dropna()
                v_btc_ret = v_btc.pct_change().dropna()
                common_idx = v_eth_ret.index.intersection(v_btc_ret.index)
                if len(common_idx) >= 3:
                    corr = float(np.corrcoef(v_eth_ret.loc[common_idx], v_btc_ret.loc[common_idx])[0, 1])
                    feats.append(corr if np.isfinite(corr) else 0.0)
                else:
                    feats.append(0.0)
            else:
                feats.extend([0.0, 0.0])
        except Exception:
            feats.extend([0.0, 0.0])

        # 10: Momentum spread (MACD_BTC - MACD_ETH) si existen columnas
        try:
            btc_f = features_by_symbol.get("BTCUSDT")
            eth_f = features_by_symbol.get("ETHUSDT")
            if (isinstance(btc_f, pd.DataFrame) and "macd" in btc_f.columns and not btc_f.empty and
                isinstance(eth_f, pd.DataFrame) and "macd" in eth_f.columns and not eth_f.empty):
                val = float(btc_f["macd"].iloc[-1]) - float(eth_f["macd"].iloc[-1])
                feats.append(val if np.isfinite(val) else 0.0)
            else:
                feats.append(0.0)
        except Exception:
            feats.append(0.0)

        # Ajuste a 11
        if len(feats) < self.CROSS_FEATURES_DIM:
            feats.extend([0.0] * (self.CROSS_FEATURES_DIM - len(feats)))
        elif len(feats) > self.CROSS_FEATURES_DIM:
            feats = feats[:self.CROSS_FEATURES_DIM]

        return [float(x) for x in feats]

    def _build_finrl_observation(self,
                                 market_data: Dict[str, pd.DataFrame],
                                 features_by_symbol: Dict[str, pd.DataFrame]) -> np.ndarray:
        """
        Construye la observación (1, 63) para el modelo FinRL:
          - 52 features base de BTCUSDT (o lo que haya, respetando orden estable)
          - 11 features cross/L3
        Tolerante a sets de 19 columnas: rellena hasta 52.
        """
        # 1) Base features: BTCUSDT como principal
        btc_features_df = features_by_symbol.get("BTCUSDT")
        base_vec = self._select_base_features_row(btc_features_df)

        # 2) Cross/L3
        cross_vec = self._build_cross_l3_features(market_data, features_by_symbol)

        # 3) Componer y garantizar forma
        obs_vec = base_vec + cross_vec

        if len(obs_vec) != self.FINRL_OBS_DIM:
            logger.warning(f"⚠️ Observación de tamaño {len(obs_vec)}; ajustando a {self.FINRL_OBS_DIM}")
            if len(obs_vec) < self.FINRL_OBS_DIM:
                obs_vec.extend([0.0] * (self.FINRL_OBS_DIM - len(obs_vec)))
            else:
                obs_vec = obs_vec[:self.FINRL_OBS_DIM]

        # Limpieza final
        obs = np.array([obs_vec], dtype=np.float32)
        obs[~np.isfinite(obs)] = 0.0
        return obs

    # ------------------- PIPELINE DE SEÑALES -------------------

    async def ai_signals(self, market_data: Dict[str, pd.DataFrame], features_by_symbol: Dict[str, pd.DataFrame]) -> List[TacticalSignal]:
        """
        Genera señales basadas en el modelo PPO (FinRLProcessor).
        Ahora construimos y pasamos 'observation' con forma (1, 63).
        """
        signals = []
        try:
            universe = self.config.get('signals', {}).get('universe', ['BTCUSDT', 'ETHUSDT'])
            logger.debug(f"🤖 Generando señales IA para universo: {universe}")
            # Preconstruimos una única observación global basada en BTC + cross con ETH
            observation = self._build_finrl_observation(market_data, features_by_symbol)

            for symbol in universe:
                if symbol.upper() == "USDT":
                    continue

                # LOG DETALLADO DE ENTRADA
                logger.info(f"[L2-DUMP] market_data[{symbol}]: shape={market_data.get(symbol, pd.DataFrame()).shape}")
                features = features_by_symbol.get(symbol)
                if isinstance(features, pd.DataFrame):
                    logger.info(f"[L2-DUMP] features_by_symbol[{symbol}]: shape={features.shape}, columns={list(features.columns)}")
                    logger.info(f"[L2-DUMP] Última fila features[{symbol}]: {features.iloc[-1].to_dict() if not features.empty else '{}'}")
                else:
                    logger.info(f"[L2-DUMP] features_by_symbol[{symbol}]: {features}")

                # Comprobación segura de DataFrame
                if not isinstance(features, pd.DataFrame) or features.empty:
                    logger.warning(f"⚠️ Sin features válidos para {symbol} (vacío o no es DataFrame)")

                try:
                    # Paquete para FinRL: incluimos 'observation' (numpy array (1, 63)) además de ohlcv e indicadores
                    last_md = market_data.get(symbol, pd.DataFrame())
                    last_md_dict = last_md.iloc[-1].to_dict() if isinstance(last_md, pd.DataFrame) and not last_md.empty else {}

                    feature_dict = features.iloc[-1].to_dict() if isinstance(features, pd.DataFrame) and not features.empty else {}

                    market_data_symbol = {
                        'ohlcv': last_md_dict,
                        'indicators': feature_dict,
                        'observation': observation,  # <- clave nueva: algunos wrappers la detectan directamente
                        # Extras opcionales para logging/modelos que los busquen:
                        'change_24h': feature_dict.get('change_24h', feature_dict.get('price_change_24h', 0.0)),
                        'l3_regime': feature_dict.get('l3_regime', 0.0),
                        'l3_risk_appetite': feature_dict.get('l3_risk_appetite', 0.5),
                        'l3_alloc_BTC': feature_dict.get('l3_alloc_BTC', 0.0),
                        'l3_alloc_ETH': feature_dict.get('l3_alloc_ETH', 0.0),
                        'l3_alloc_CASH': feature_dict.get('l3_alloc_CASH', 0.0)
                    }

                    # Generar señal con FinRLProcessor
                    signal = self.finrl_model.generate_signal(symbol=symbol, market_data=market_data_symbol)
                    if signal:
                        signals.append(signal)
                        try:
                            logger.info(f"🎯 Señal IA: {symbol} {signal.side} strength={getattr(signal, 'strength', 0.0):.3f}")
                        except Exception:
                            logger.info(f"🎯 Señal IA: {getattr(signal, 'symbol', symbol)} {getattr(signal, 'side', '?')}")
                    else:
                        logger.debug(f"🤖 Sin señal para {symbol}")
                except Exception as e:
                    logger.error(f"❌ Error procesando señal para {symbol}: {e}", exc_info=True)
                    continue
            logger.info(f"🤖 Señales IA generadas: {len(signals)}")
            return signals
        except Exception as e:
            logger.error(f"❌ Error generando señales IA: {e}", exc_info=True)
            return []

    async def technical_signals(self, market_data: Dict[str, pd.DataFrame], technical_indicators: Dict[str, pd.DataFrame]) -> List[TacticalSignal]:
        """
        Genera señales técnicas usando MultiTimeframeTechnical.
        """
        try:
            logger.debug(f"📊 Datos de mercado para señales técnicas: {list(market_data.keys())}")
            signals = await self.technical.generate_signals(market_data, technical_indicators)
            logger.info(f"📊 Señales técnicas generadas: {len(signals)}")
            if not signals:
                logger.warning("⚠️ No se generaron señales técnicas, verificar datos de entrada o umbrales")
            return signals
        except Exception as e:
            logger.error(f"❌ Error generando señales técnicas: {e}", exc_info=True)
            return []

    async def risk_signals(self, market_data: Dict[str, pd.DataFrame], portfolio_data: Dict[str, Any]) -> List[TacticalSignal]:
        """
        Genera señales de riesgo usando RiskOverlay.
        """
        try:
            logger.debug(f"🛡️ Datos para señales de riesgo - Mercado: {list(market_data.keys())}, Portfolio: {portfolio_data}")
            signals = await self.risk.generate_risk_signals(market_data, portfolio_data)
            logger.info(f"🛡️ Señales de riesgo generadas: {len(signals)}")
            if not signals:
                logger.warning("⚠️ No se generaron señales de riesgo, verificar datos de entrada o umbrales")
            return signals
        except Exception as e:
            logger.error(f"❌ Error generando señales de riesgo: {e}", exc_info=True)
            return []

    async def process(self, market_data: Dict[str, pd.DataFrame], features_by_symbol: Dict[str, pd.DataFrame], state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecuta la capa Tactic:
        - Genera señales (AI + técnico + riesgo)
        - Combina y guarda en state['l2']
        - Convierte señales a órdenes para L1
        - Devuelve dict: {"signals", "orders_for_l1", "metadata"}
        """
        if features_by_symbol is None:
            features_by_symbol = {}
        ai_signals = []
        technical_signals = []
        risk_signals = []
        all_signals = []

        try:
            logger.info("🎯 Iniciando procesamiento L2TacticProcessor")
            logger.info(f"[L2-GLOBAL] market_data keys: {list(market_data.keys())}")
            logger.info(f"[L2-GLOBAL] features_by_symbol keys: {list(features_by_symbol.keys())}")
            portfolio_data = state.get("portfolio", {})

            # Generación de señales
            ai_signals = await self.ai_signals(market_data, features_by_symbol)
            technical_signals = await self.technical_signals(market_data, features_by_symbol)
            risk_signals = await self.risk_signals(market_data, portfolio_data)

            # Combinar señales
            all_signals = ai_signals + technical_signals + risk_signals
            if all_signals:
                try:
                    all_signals = self.signal_composer.compose(all_signals)
                    logger.info(f"✅ Señales compuestas: {len(all_signals)}")
                    for signal in all_signals:
                        symbol = getattr(signal, 'symbol', None) or (signal.get('symbol') if isinstance(signal, dict) else None)
                        side = getattr(signal, 'side', None) or (signal.get('side') if isinstance(signal, dict) else None)
                        if not symbol or not side:
                            logger.error(f"❌ Señal inválida: {getattr(signal, '__dict__', signal)}")
                            all_signals = []
                            break
                except Exception as e:
                    logger.error(f"❌ Error al componer señales: {e}", exc_info=True)
                    all_signals = []
            else:
                logger.warning("⚠️ No hay señales para componer")

            # Guardar en state['l2']
            now = pd.Timestamp.utcnow()
            if not isinstance(state.get('l2'), L2State):
                state['l2'] = L2State()
            state['l2'].signals = all_signals
            state['l2'].last_update = now
            logger.debug(f"Actualizado state.l2.signals con {len(all_signals)} señales")
            
            # Convertir señales a órdenes para L1
            orders_for_l1 = []
            for signal in all_signals:
                try:
                    def get_attr_or_key(obj, key, default=None):
                        if hasattr(obj, key):
                            return getattr(obj, key)
                        if isinstance(obj, dict):
                            return obj.get(key, default)
                        return default

                    order = {
                        "symbol": get_attr_or_key(signal, "symbol"),
                        "side": get_attr_or_key(signal, "side"),
                        "type": "market",
                        "strength": get_attr_or_key(signal, "strength"),
                        "confidence": get_attr_or_key(signal, "confidence"),
                        "signal_type": get_attr_or_key(signal, "signal_type", "tactical"),
                        "timestamp": get_attr_or_key(signal, "timestamp", now),
                        "metadata": get_attr_or_key(signal, "metadata", {})
                    }
                    if order["symbol"] and order["side"]:
                        orders_for_l1.append(order)
                    else:
                        logger.warning(f"⚠️ Orden inválida para señal: {getattr(signal, '__dict__', signal)}")
                except Exception as e:
                    logger.error(f"❌ Error convirtiendo señal a orden: {e}", exc_info=True)

            logger.info(f"✅ L2TacticProcessor completado: {len(all_signals)} señales, {len(orders_for_l1)} órdenes")

            return {
                "signals": all_signals,
                "orders_for_l1": orders_for_l1,
                "metadata": {
                    "ai_signals": len(ai_signals),
                    "technical_signals": len(technical_signals),
                    "risk_signals": len(risk_signals),
                    "total_signals": len(all_signals)
                }
            }

        except Exception as e:
            logger.error(f"❌ Error crítico en L2TacticProcessor.process(): {e}", exc_info=True)
            now = pd.Timestamp.utcnow()
            if not isinstance(state.get('l2'), L2State):
                state['l2'] = L2State()
            state['l2'].signals = []
            state['l2'].last_update = now
            return {
                "signals": [],
                "orders_for_l1": [],
                "metadata": {"error": str(e)}
            }