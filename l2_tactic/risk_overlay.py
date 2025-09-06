"""
Risk Overlay Module - CORREGIDO
==================
Módulo para generar señales de ajuste de riesgo compatible con tu configuración
"""

import asyncio
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from core.logging import logger
from .models import TacticalSignal

class RiskOverlay:
    """
    Generador de señales de ajuste de riesgo - CORREGIDO
    """
    
    def __init__(self, config=None):
        self.config = config
        # Forzar umbral de drawdown
        self.max_drawdown_limit = 0.01  # Forzado a 1%
        self.max_expected_vol = 0.05
        self.correlation_limit = 0.8
        logger.info(f"🛡️ RiskOverlay inicializado - MaxDD: {self.max_drawdown_limit:.1%}, MaxVol: {self.max_expected_vol:.1%}")
        
    async def generate_risk_signals(self, market_data: Dict[str, Any], portfolio_data: Dict[str, Any]) -> List[TacticalSignal]:
        """
        Genera señales de ajuste de riesgo basadas en condiciones de mercado y portfolio
        """
        signals = []
        logger.debug(f"Portfolio data: {portfolio_data}")
        
        try:
            # Verificar volatilidad excesiva
            vol_signals = await self._check_volatility_risk(market_data)
            logger.info(f"[DEBUG] Señales de volatilidad generadas: {len(vol_signals)}")
            signals.extend(vol_signals)

            # Verificar correlación
            corr_signals = await self._check_correlation_risk(market_data, portfolio_data)
            logger.info(f"[DEBUG] Señales de correlación generadas: {len(corr_signals)}")
            signals.extend(corr_signals)

            # Verificar drawdown
            dd_signals = await self._check_drawdown_risk(portfolio_data)
            logger.info(f"[DEBUG] Señales de drawdown generadas: {len(dd_signals)}")
            signals.extend(dd_signals)

            logger.info(f"🛡️ Señales de riesgo generadas: {len(signals)}")
            return signals

        except Exception as e:
            logger.error(f"❌ Error generando señales de riesgo: {e}", exc_info=True)
            return []
    
    async def _check_volatility_risk(self, market_data: Dict[str, pd.DataFrame]) -> List[TacticalSignal]:
        """
        Verifica riesgo de volatilidad excesiva
        """
        signals = []
        
        
        try:
            for symbol, data in market_data.items():
                if symbol == 'USDT' or not isinstance(data, pd.DataFrame) or data.empty:
                    logger.warning(f"⚠️ Datos de mercado vacíos o inválidos para {symbol}")
                    continue
                    
                # Calcular volatilidad desde la columna 'close'
                if 'close' not in data.columns:
                    logger.warning(f"⚠️ Columna 'close' no encontrada para {symbol}")
                    continue
                
                volatility = data['close'].pct_change().std() * np.sqrt(252)  # Volatilidad anualizada
                logger.debug(f"Volatilidad calculada para {symbol}: {volatility:.3f}")
                
                if volatility > self.max_expected_vol:
                    signal = TacticalSignal(
                        symbol=symbol,
                        signal_type='risk_high_volatility',
                        strength=-(volatility - self.max_expected_vol),
                        confidence=0.8,
                        side='reduce',
                        features={'volatility': volatility, 'max_vol': self.max_expected_vol},
                        timestamp=pd.Timestamp.now(),
                        metadata={'risk_type': 'volatility', 'action': 'reduce_position'}
                    )
                    signals.append(signal)
                    logger.warning(f"⚠️ Alta volatilidad {symbol}: {volatility:.3f} > {self.max_expected_vol:.3f}")
            
            return signals
            
        except Exception as e:
            logger.error(f"❌ Error verificando riesgo de volatilidad: {e}", exc_info=True)
            return []
    
    async def _check_correlation_risk(self, market_data: Dict[str, pd.DataFrame], portfolio_data: Dict[str, Any]) -> List[TacticalSignal]:
        """
        Verifica riesgo de correlación entre activos
        """
        signals = []
        
        try:
            btc_data = market_data.get('BTCUSDT')
            eth_data = market_data.get('ETHUSDT')
            
            if not isinstance(btc_data, pd.DataFrame) or btc_data.empty or not isinstance(eth_data, pd.DataFrame) or eth_data.empty:
                logger.warning("⚠️ Datos insuficientes para calcular correlación BTC-ETH")
                return signals
            
            if 'close' not in btc_data.columns or 'close' not in eth_data.columns:
                logger.warning("⚠️ Columna 'close' no encontrada para BTCUSDT o ETHUSDT")
                return signals
            
            # Calcular correlación de retornos
            btc_returns = btc_data['close'].pct_change().dropna()
            eth_returns = eth_data['close'].pct_change().dropna()
            if len(btc_returns) > 10 and len(eth_returns) > 10:
                correlation = btc_returns.corr(eth_returns)
                logger.debug(f"Correlación BTC-ETH: {correlation:.3f}")
                
                if correlation > self.correlation_limit:
                    signal = TacticalSignal(
                        symbol='PORTFOLIO',
                        signal_type='risk_high_correlation',
                        strength=-(correlation - self.correlation_limit),
                        confidence=0.6,
                        side='reduce',
                        features={'correlation': correlation, 'btc_returns': btc_returns.iloc[-1], 'eth_returns': eth_returns.iloc[-1]},
                        timestamp=pd.Timestamp.now(),
                        metadata={'risk_type': 'correlation', 'pairs': 'BTC-ETH'}
                    )
                    signals.append(signal)
                    logger.warning(f"⚠️ Alta correlación BTC-ETH: {correlation:.3f} > {self.correlation_limit:.3f}")
            
            return signals

        except Exception as e:
            logger.error(f"❌ Error verificando riesgo de correlación: {e}", exc_info=True)
            return []
    
    async def _check_drawdown_risk(self, portfolio_data: Dict[str, Any]) -> List[TacticalSignal]:
        """
        Verifica riesgo de drawdown excesivo
        """
        signals = []
        
        try:
            current_dd = portfolio_data.get('drawdown', 
                          portfolio_data.get('current_drawdown',
                          portfolio_data.get('dd', 0)))
            logger.debug(f"Current drawdown: {current_dd:.1%}")
            
            if current_dd > self.max_drawdown_limit:
                signal = TacticalSignal(
                    symbol='PORTFOLIO',
                    signal_type='risk_max_drawdown',
                    strength=-(current_dd - self.max_drawdown_limit) * 2,
                    confidence=0.9,
                    side='close_all',
                    features={'current_dd': current_dd, 'max_dd': self.max_drawdown_limit},
                    timestamp=pd.Timestamp.now(),
                    metadata={'risk_type': 'drawdown', 'action': 'close_positions'}
                )
                signals.append(signal)
                logger.error(f"🚨 DRAWDOWN CRÍTICO: {current_dd:.1%} > {self.max_drawdown_limit:.1%}")
            
            return signals
            
        except Exception as e:
            logger.error(f"❌ Error verificando riesgo de drawdown: {e}", exc_info=True)
            return []