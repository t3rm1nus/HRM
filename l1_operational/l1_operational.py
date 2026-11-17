# l1_operational/l1_operational.py
import asyncio
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime

from core.logging import logger
from .models import L1Model, L1Signal
from l2_tactic.models import TacticalSignal, SignalDirection, SignalSource

class L1OperationalProcessor:
    """
    Procesador L1 que genera señales operacionales básicas
    y las convierte al formato esperado por L2
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.l1_model = L1Model(self.config.get('l1_model', {}))
        self.signal_cache = {}
        self.last_update = None

    async def process_market_data(self, market_data: Dict[str, pd.DataFrame]) -> List[TacticalSignal]:
        """
        Procesa datos de mercado y genera señales tácticas para L2

        Args:
            market_data: Dict con DataFrames OHLCV por símbolo

        Returns:
            Lista de TacticalSignal para L2
        """
        try:
            logger.info("🔍 L1: Procesando datos de mercado para generar señales operacionales")

            # Generar señales L1
            l1_result = self.l1_model.predict(market_data)

            signals_l1 = l1_result.get('signals', [])
            metrics = l1_result.get('metrics', {})

            logger.info(f"📊 L1: Generadas {len(signals_l1)} señales operacionales")
            logger.info(f"📈 L1: Métricas - Buy: {metrics.get('buy_signals', 0)}, "
                       f"Sell: {metrics.get('sell_signals', 0)}, "
                       f"Hold: {metrics.get('hold_signals', 0)}")

            # Convertir señales L1 a formato L2 (TacticalSignal)
            tactical_signals = self._convert_to_tactical_signals(signals_l1)

            # Cache de señales para debugging
            self.signal_cache = {
                'l1_signals': signals_l1,
                'tactical_signals': tactical_signals,
                'metrics': metrics,
                'timestamp': datetime.now()
            }
            self.last_update = datetime.now()

            logger.info(f"✅ L1: Convertidas {len(tactical_signals)} señales tácticas para L2")

            return tactical_signals

        except Exception as e:
            logger.opt(exception=True).error(f"❌ Error procesando datos de mercado en L1: {e}")
            return []

    def _convert_to_tactical_signals(self, l1_signals: List[L1Signal]) -> List[TacticalSignal]:
        """
        Convierte señales L1 al formato TacticalSignal esperado por L2
        """
        tactical_signals = []

        for l1_signal in l1_signals:
            try:
                # Mapear dirección L1 a formato L2
                side = self._map_direction_to_side(l1_signal.direction)

                # Crear features para L2
                features = self._create_tactical_features(l1_signal)

                # Crear TacticalSignal
                tactical_signal = TacticalSignal(
                    symbol=l1_signal.symbol,
                    strength=l1_signal.strength,
                    confidence=l1_signal.confidence,
                    side=side,
                    signal_type='operational',  # Tipo específico de L1
                    source='l1_operational',    # Fuente L1
                    features=features,
                    timestamp=pd.Timestamp(l1_signal.timestamp),
                    metadata={
                        'l1_signal_type': l1_signal.signal_type.value,
                        'l1_model': l1_signal.metadata.get('model', 'unknown'),
                        'l1_reason': l1_signal.metadata.get('reason', 'unknown')
                    }
                )

                tactical_signals.append(tactical_signal)

            except Exception as e:
                logger.error(f"Error convirtiendo señal L1 para {l1_signal.symbol}: {e}")

        return tactical_signals

    def _map_direction_to_side(self, direction: str) -> str:
        """Mapea dirección L1 a formato side de L2"""
        direction_map = {
            'buy': 'buy',
            'sell': 'sell',
            'hold': 'hold'
        }
        return direction_map.get(direction.lower(), 'hold')

    def _create_tactical_features(self, l1_signal: L1Signal) -> Dict[str, Any]:
        """Crea features para TacticalSignal basado en señal L1"""
        features = dict(l1_signal.features)  # Copiar features originales

        # Agregar metadatos específicos de L1
        features.update({
            'l1_signal_type': l1_signal.signal_type.value,
            'l1_model': l1_signal.metadata.get('model', 'unknown'),
            'l1_confidence': l1_signal.confidence,
            'l1_strength': l1_signal.strength,
            'l1_timestamp': l1_signal.timestamp.isoformat() if hasattr(l1_signal.timestamp, 'isoformat') else str(l1_signal.timestamp)
        })

        # Agregar indicadores técnicos si están disponibles
        if 'close' in features:
            features['price'] = features['close']

        # Asegurar que todos los valores sean serializables
        for key, value in features.items():
            if isinstance(value, (pd.Timestamp, datetime)):
                features[key] = value.isoformat()
            elif not isinstance(value, (int, float, str, bool)):
                features[key] = str(value)

        return features

    def get_signal_summary(self) -> Dict[str, Any]:
        """Retorna resumen de señales generadas"""
        if not self.signal_cache:
            return {'status': 'no_signals_generated'}

        cache = self.signal_cache
        return {
            'total_l1_signals': len(cache.get('l1_signals', [])),
            'total_tactical_signals': len(cache.get('tactical_signals', [])),
            'metrics': cache.get('metrics', {}),
            'last_update': cache.get('timestamp'),
            'signal_types': self._count_signal_types(cache.get('l1_signals', []))
        }

    def _count_signal_types(self, signals: List[L1Signal]) -> Dict[str, int]:
        """Cuenta tipos de señales L1"""
        counts = {}
        for signal in signals:
            st = signal.signal_type.value
            counts[st] = counts.get(st, 0) + 1
        return counts

    async def health_check(self) -> Dict[str, Any]:
        """Verificación de salud del procesador L1"""
        try:
            # Verificar que el modelo L1 esté operativo
            if not hasattr(self.l1_model, 'models'):
                return {'status': 'error', 'message': 'L1 model not properly initialized'}

            model_count = len(self.l1_model.models)
            active_models = sum(1 for m in self.l1_model.models.values() if m is not None)

            return {
                'status': 'healthy' if active_models == model_count else 'degraded',
                'total_models': model_count,
                'active_models': active_models,
                'last_update': self.last_update.isoformat() if self.last_update else None,
                'signal_cache_size': len(self.signal_cache)
            }

        except Exception as e:
            logger.error(f"Error in L1 health check: {e}")
            return {'status': 'error', 'message': str(e)}

# Funciones de compatibilidad
class L1ModelOld:
    def __init__(self):
        pass

    def predict(self, data):
        return {
            'accuracy': 0.85,
            'precision': 0.80,
            'f1_score': 0.82,
            'profit_contribution': 1000,
            'latency_ms': 50
        }

class BusAdapterAsync:
    def __init__(self):
        from core.logging import logger
        logger.info("[BusAdapterAsync] Inicializado (pendiente de start())")
