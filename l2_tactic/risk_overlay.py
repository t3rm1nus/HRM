"""
Risk Overlay Module - CORREGIDO
==================
Módulo para generar señales de ajuste de riesgo compatible con tu configuración
"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

from core.logging import logger
from .models import TacticalSignal

class RiskOverlay:
    """
    Generador de señales de ajuste de riesgo - CORREGIDO
    """
    
    def __init__(self, config=None):
        self.config = config
        
        # Importar configuración del sistema
        try:
            from comms.config import RISK_CONFIG
            self.max_drawdown_limit = RISK_CONFIG.max_drawdown_limit
            self.max_expected_vol = RISK_CONFIG.max_expected_vol
            self.correlation_limit = RISK_CONFIG.correlation_limit
            logger.info(f"🛡️ RiskOverlay usando configuración del sistema")
        except ImportError:
            # Fallback si no se puede importar
            self.max_drawdown_limit = getattr(config, 'max_drawdown_limit', 0.15) if config else 0.15
            self.max_expected_vol = getattr(config, 'max_expected_vol', 0.05) if config else 0.05
            self.correlation_limit = getattr(config, 'correlation_limit', 0.8) if config else 0.8
            
            # Si config es un dict en lugar de objeto
            if isinstance(config, dict):
                self.max_drawdown_limit = config.get('max_drawdown_limit', 0.15)
                self.max_expected_vol = config.get('max_expected_vol', 0.05)
                self.correlation_limit = config.get('correlation_limit', 0.8)
        
        logger.info(f"🛡️ RiskOverlay inicializado - MaxDD: {self.max_drawdown_limit:.1%}, MaxVol: {self.max_expected_vol:.1%}")
        
    async def generate_risk_signals(self, market_data: Dict[str, Any], portfolio_data: Dict[str, Any]) -> List[TacticalSignal]:
        """
        Genera señales de ajuste de riesgo basadas en condiciones de mercado y portfolio
        """
        signals = []
        
        try:
            # Verificar volatilidad excesiva
            vol_signals = await self._check_volatility_risk(market_data)
            signals.extend(vol_signals)
            
            # Verificar correlación
            corr_signals = await self._check_correlation_risk(market_data, portfolio_data)
            signals.extend(corr_signals)
            
            # Verificar drawdown - CORREGIDO
            dd_signals = await self._check_drawdown_risk(portfolio_data)
            signals.extend(dd_signals)
            
            logger.info(f"🛡️ Señales de riesgo generadas: {len(signals)}")
            return signals
            
        except Exception as e:
            logger.error(f"❌ Error generando señales de riesgo: {e}")
            return []
    
    async def _check_volatility_risk(self, market_data: Dict[str, Any]) -> List[TacticalSignal]:
        """
        Verifica riesgo de volatilidad excesiva
        """
        signals = []
        
        try:
            for symbol, data in market_data.items():
                if symbol == 'USDT' or not isinstance(data, dict):
                    continue
                    
                # Obtener volatilidad de diferentes posibles estructuras
                volatility = 0
                if 'volatility' in data:
                    if isinstance(data['volatility'], dict):
                        volatility = data['volatility'].get('1d', 0)
                    else:
                        volatility = data['volatility']
                elif 'vol' in data:
                    volatility = data['vol']
                elif 'change_24h' in data:
                    volatility = abs(data['change_24h'])  # Usar cambio 24h como proxy
                
                if volatility > self.max_expected_vol:
                    signal = TacticalSignal(
                        symbol=symbol,
                        signal_type='risk_high_volatility',
                        strength=-(volatility - self.max_expected_vol),  # Negative = reduce
                        confidence=0.8,
                        side='reduce',
                        features={'volatility': volatility, 'max_vol': self.max_expected_vol},
                        timestamp=datetime.now().timestamp(),
                        metadata={'risk_type': 'volatility', 'action': 'reduce_position'}
                    )
                    signals.append(signal)
                    logger.warning(f"⚠️ Alta volatilidad {symbol}: {volatility:.3f} > {self.max_expected_vol:.3f}")
            
            return signals
            
        except Exception as e:
            logger.error(f"❌ Error verificando riesgo de volatilidad: {e}")
            return []
    
    async def _check_correlation_risk(self, market_data: Dict[str, Any], portfolio_data: Dict[str, Any]) -> List[TacticalSignal]:
        """
        Verifica riesgo de correlación alta entre activos
        """
        signals = []
        
        try:
            # Buscar datos BTC y ETH en diferentes formatos posibles
            btc_data = None
            eth_data = None
            
            # Intentar diferentes keys
            for key in market_data.keys():
                if 'BTC' in key.upper():
                    btc_data = market_data[key]
                elif 'ETH' in key.upper():
                    eth_data = market_data[key]
            
            if btc_data and eth_data and isinstance(btc_data, dict) and isinstance(eth_data, dict):
                # Obtener cambios 24h
                btc_change = btc_data.get('change_24h', btc_data.get('change', 0))
                eth_change = eth_data.get('change_24h', eth_data.get('change', 0))
                
                # Si ambos se mueven fuertemente en la misma dirección
                if abs(btc_change) > 0.03 and abs(eth_change) > 0.03:
                    correlation_risk = (btc_change > 0) == (eth_change > 0)  # Misma dirección
                    
                    if correlation_risk:
                        signal = TacticalSignal(
                            symbol='PORTFOLIO',
                            signal_type='risk_high_correlation',
                            strength=-0.3,  # Reducir exposición
                            confidence=0.6,
                            side='reduce',
                            features={'btc_change': btc_change, 'eth_change': eth_change},
                            timestamp=datetime.now().timestamp(),
                            metadata={'risk_type': 'correlation', 'pairs': 'BTC-ETH'}
                        )
                        signals.append(signal)
                        logger.warning(f"⚠️ Alta correlación BTC-ETH: {btc_change:.3f}, {eth_change:.3f}")
            
            return signals
            
        except Exception as e:
            logger.error(f"❌ Error verificando riesgo de correlación: {e}")
            return []
    
    async def _check_drawdown_risk(self, portfolio_data: Dict[str, Any]) -> List[TacticalSignal]:
        """
        Verifica riesgo de drawdown excesivo - CORREGIDO
        """
        signals = []
        
        try:
            # Obtener drawdown actual de diferentes posibles estructuras
            current_dd = 0
            
            if isinstance(portfolio_data, dict):
                current_dd = portfolio_data.get('drawdown', 
                            portfolio_data.get('current_drawdown',
                            portfolio_data.get('dd', 0)))
            
            # Si drawdown excede el límite
            if current_dd > self.max_drawdown_limit:
                signal = TacticalSignal(
                    symbol='PORTFOLIO',
                    signal_type='risk_max_drawdown',
                    strength=-(current_dd - self.max_drawdown_limit) * 2,  # Señal negativa fuerte
                    confidence=0.9,
                    side='close_all',
                    features={'current_dd': current_dd, 'max_dd': self.max_drawdown_limit},
                    timestamp=datetime.now().timestamp(),
                    metadata={'risk_type': 'drawdown', 'action': 'close_positions'}
                )
                signals.append(signal)
                logger.error(f"🚨 DRAWDOWN CRÍTICO: {current_dd:.1%} > {self.max_drawdown_limit:.1%}")
            
            return signals
            
        except Exception as e:
            logger.error(f"❌ Error verificando riesgo de drawdown: {e}")
            return []