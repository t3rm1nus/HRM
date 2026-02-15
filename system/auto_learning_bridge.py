#!/usr/bin/env python3
"""
Puente entre el ciclo de trading y el auto-learning.
Registra trades ejecutados y calcula métricas básicas.
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from core.logging import logger

class AutoLearningBridge:
    """Puente para registrar trades en el auto-learning"""
    
    def __init__(self, auto_learning_integration):
        self.al_integration = auto_learning_integration
        self.pending_trades = {}  # Trades abiertos esperando cierre
        
    async def record_order_execution(self, order: Dict[str, Any], 
                                     l3_context: Dict[str, Any],
                                     market_data: Dict[str, Any]):
        """
        Registrar una orden ejecutada para auto-learning.
        
        Args:
            order: Orden ejecutada
            l3_context: Contexto L3 (regimen, señal, confianza)
            market_data: Datos de mercado actuales
        """
        try:
            symbol = order.get("symbol", "UNKNOWN")
            action = order.get("action", "hold")
            
            if action == "buy":
                # Registrar entrada
                trade_data = {
                    "symbol": symbol,
                    "side": "buy",
                    "entry_price": order.get("price", 0.0),
                    "exit_price": order.get("price", 0.0),  # Placeholder
                    "quantity": order.get("quantity", 0.0),
                    "pnl": 0.0,  # Placeholder - se actualiza al cerrar
                    "pnl_pct": 0.0,
                    "model_used": self._extract_model_source(order),
                    "confidence": order.get("confidence", 0.5),
                    "regime": l3_context.get("regime", "neutral"),
                    "features": self._extract_features(market_data, symbol),
                    "timestamp": datetime.now().isoformat()
                }
                
                # Guardar referencia para emparejar con sell posterior
                self.pending_trades[symbol] = trade_data
                
                # Registrar en auto-learning
                if self.al_integration:
                    self.al_integration.record_trade_for_learning(trade_data)
                    
                logger.info(f"🤖 AUTO-LEARNING | Trade registrado: {symbol} BUY @ {trade_data['entry_price']:.2f}")
                
            elif action == "sell":
                # Buscar trade de entrada correspondiente
                entry_trade = self.pending_trades.pop(symbol, None)
                
                if entry_trade:
                    # Calcular PnL real
                    exit_price = order.get("price", 0.0)
                    entry_price = entry_trade["entry_price"]
                    quantity = order.get("quantity", 0.0)
                    
                    pnl = (exit_price - entry_price) * quantity
                    pnl_pct = (exit_price - entry_price) / entry_price if entry_price > 0 else 0
                    
                    # Actualizar trade con datos de cierre
                    closed_trade = {
                        **entry_trade,
                        "side": "sell",
                        "exit_price": exit_price,
                        "pnl": pnl,
                        "pnl_pct": pnl_pct,
                        "closed_at": datetime.now().isoformat()
                    }
                    
                    # Registrar trade cerrado
                    if self.al_integration:
                        self.al_integration.record_trade_for_learning(closed_trade)
                    
                    pnl_emoji = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"
                    logger.info(f"🤖 AUTO-LEARNING | Trade cerrado: {symbol} SELL @ {exit_price:.2f} | PnL: {pnl_emoji} ${pnl:.2f} ({pnl_pct:.2%})")
                else:
                    logger.warning(f"🤖 AUTO-LEARNING | Sell sin entrada previa: {symbol}")
                    
        except Exception as e:
            logger.error(f"❌ Error registrando trade para auto-learning: {e}")
    
    def _extract_model_source(self, order: Dict) -> str:
        """Extraer qué modelo generó la orden"""
        source = order.get("source", "unknown")
        metadata = order.get("metadata", {})
        
        if "finrl" in source.lower():
            return "l2_finrl"
        elif "technical" in source.lower():
            return "l2_technical"
        elif "ensemble" in source.lower():
            return "l2_ensemble"
        elif "l1" in source.lower():
            return "l1_operational"
        else:
            return source
    
    def _extract_features(self, market_data: Dict, symbol: str) -> Dict[str, float]:
        """Extraer features técnicas del market data"""
        features = {}
        
        try:
            data = market_data.get(symbol, {})
            if isinstance(data, dict):
                features["close"] = data.get("close", 0)
                features["volume"] = data.get("volume", 0)
                features["rsi"] = data.get("rsi", 50)
                features["macd"] = data.get("macd", 0)
                features["sma_20"] = data.get("sma_20", 0)
                features["sma_50"] = data.get("sma_50", 0)
            elif hasattr(data, 'iloc'):
                # Es un DataFrame
                features["close"] = float(data["close"].iloc[-1])
                features["volume"] = float(data["volume"].iloc[-1]) if "volume" in data.columns else 0
                features["rsi"] = float(data["rsi"].iloc[-1]) if "rsi" in data.columns else 50
                features["macd"] = float(data["macd"].iloc[-1]) if "macd" in data.columns else 0
        except Exception:
            pass
        
        return features
    
    def get_pending_trades_count(self) -> int:
        """Obtener número de trades abiertos pendientes"""
        return len(self.pending_trades)
    
    def get_status(self) -> Dict[str, Any]:
        """Obtener estado del puente"""
        return {
            "pending_trades": len(self.pending_trades),
            "pending_symbols": list(self.pending_trades.keys()),
            "al_integration_connected": self.al_integration is not None
        }
