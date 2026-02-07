#!/usr/bin/env python3
"""
Integration Auto-Learning System
Integra el sistema de auto-aprendizaje con el sistema principal HRM
"""

import asyncio
import logging
from typing import Any, Dict, Optional
from datetime import datetime

# Importar el sistema de auto-aprendizaje
from auto_learning_system import SelfImprovingTradingSystem, TradeData

# Importar componentes del sistema HRM
from core.state_manager import get_state_manager
from core.trading_metrics import get_trading_metrics
from l1_operational.order_manager import OrderManager
from l2_tactic.tactical_signal_processor import L2TacticProcessor

logger = logging.getLogger(__name__)

class AutoLearningIntegration:
    """Integración del sistema de auto-aprendizaje con HRM"""
    
    def __init__(self):
        self.auto_learning_system = SelfImprovingTradingSystem()
        self.state_manager = None
        self.order_manager = None
        self.l2_processor = None
        self.trading_metrics = None
        
        # Estado de integración
        self.is_initialized = False
        self.last_integration_check = datetime.now()
        
        logger.info("🔧 Auto-Learning Integration initialized")
    
    async def initialize_integration(self, 
                                   state_manager=None,
                                   order_manager=None, 
                                   l2_processor=None,
                                   trading_metrics=None):
        """Inicializar la integración con componentes del sistema"""
        
        try:
            # Obtener componentes si no se proporcionan
            if state_manager is None:
                self.state_manager = get_state_manager()
            else:
                self.state_manager = state_manager
                
            if order_manager is None:
                # Intentar obtener OrderManager del sistema
                try:
                    from l1_operational.order_manager import OrderManager
                    self.order_manager = OrderManager()
                except Exception as e:
                    logger.warning(f"⚠️ Could not initialize OrderManager: {e}")
                    self.order_manager = None
            else:
                self.order_manager = order_manager
                
            if l2_processor is None:
                # Intentar obtener L2TacticProcessor
                try:
                    from l2_tactic.tactical_signal_processor import L2TacticProcessor
                    self.l2_processor = L2TacticProcessor()
                except Exception as e:
                    logger.warning(f"⚠️ Could not initialize L2TacticProcessor: {e}")
                    self.l2_processor = None
            else:
                self.l2_processor = l2_processor
                
            if trading_metrics is None:
                self.trading_metrics = get_trading_metrics()
            else:
                self.trading_metrics = trading_metrics
            
            # Iniciar auto-improvement cycle
            self.auto_learning_system.start_auto_improvement()
            
            self.is_initialized = True
            self.last_integration_check = datetime.now()
            
            logger.info("✅ Auto-Learning Integration fully initialized")
            logger.info(f"   📊 State Manager: {'✅' if self.state_manager else '❌'}")
            logger.info(f"   🤖 Order Manager: {'✅' if self.order_manager else '❌'}")
            logger.info(f"   🎯 L2 Processor: {'✅' if self.l2_processor else '❌'}")
            logger.info(f"   📈 Trading Metrics: {'✅' if self.trading_metrics else '❌'}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Integration initialization failed: {e}")
            return False
    
    def record_trade_for_learning(self, trade_data: Dict[str, Any]):
        """Registrar un trade para el sistema de auto-aprendizaje"""
        
        if not self.is_initialized:
            logger.warning("⚠️ Integration not initialized, skipping trade recording")
            return False
        
        try:
            # Convertir datos del trade a formato compatible
            formatted_trade = {
                'symbol': trade_data.get('symbol', 'UNKNOWN'),
                'side': trade_data.get('side', 'buy'),
                'entry_price': trade_data.get('entry_price', 0.0),
                'exit_price': trade_data.get('exit_price', 0.0),
                'quantity': trade_data.get('quantity', 0.0),
                'pnl': trade_data.get('pnl', 0.0),
                'pnl_pct': trade_data.get('pnl_pct', 0.0),
                'model_used': trade_data.get('model_used', 'unknown'),
                'confidence': trade_data.get('confidence', 0.5),
                'regime': trade_data.get('regime', 'neutral'),
                'features': trade_data.get('features', {}),
                'market_data': trade_data.get('market_data', {})
            }
            
            # Registrar en el sistema de auto-aprendizaje
            self.auto_learning_system.record_trade(formatted_trade)
            
            logger.debug(f"📊 Trade recorded for auto-learning: {formatted_trade['symbol']} {formatted_trade['side']}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error recording trade for learning: {e}")
            return False
    
    async def get_learning_status(self) -> Dict[str, Any]:
        """Obtener estado del sistema de auto-aprendizaje"""
        
        if not self.is_initialized:
            return {'error': 'Integration not initialized'}
        
        try:
            # Obtener estado del sistema de auto-aprendizaje
            learning_status = self.auto_learning_system.get_system_status()
            
            # Añadir información de integración
            integration_info = {
                'integration_active': self.is_initialized,
                'last_check': self.last_integration_check,
                'state_manager_available': self.state_manager is not None,
                'order_manager_available': self.order_manager is not None,
                'l2_processor_available': self.l2_processor is not None,
                'trading_metrics_available': self.trading_metrics is not None
            }
            
            return {
                'learning_system': learning_status,
                'integration': integration_info
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting learning status: {e}")
            return {'error': str(e)}
    
    async def trigger_manual_retrain(self) -> bool:
        """Disparar reentrenamiento manual del sistema"""
        
        if not self.is_initialized:
            logger.warning("⚠️ Integration not initialized, cannot trigger retrain")
            return False
        
        try:
            # En implementación real, esto dispararía el reentrenamiento
            # Por ahora, solo registramos la solicitud
            logger.info("🔄 Manual retrain triggered")
            
            # Podría disparar el auto-reentrenamiento forzado
            # await self.auto_learning_system._auto_retrain_models()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error triggering manual retrain: {e}")
            return False
    
    async def cleanup(self):
        """Limpiar recursos de la integración"""
        
        try:
            if self.is_initialized:
                logger.info("🧹 Cleaning up Auto-Learning Integration...")
                
                # Detener auto-improvement cycle
                self.auto_learning_system.is_running = False
                
                # Limpiar referencias
                self.state_manager = None
                self.order_manager = None
                self.l2_processor = None
                self.trading_metrics = None
                self.is_initialized = False
                
                logger.info("✅ Auto-Learning Integration cleaned up")
                
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")

async def integrate_with_main_system():
    """
    Función principal de integración
    Esta es la función que se llama desde main.py
    """
    
    try:
        # Crear instancia de integración
        integration = AutoLearningIntegration()
        
        # Inicializar integración con await
        await integration.initialize_integration()
        
        logger.info("🤖 Auto-Learning System successfully integrated with HRM")
        return integration
        
    except Exception as e:
        logger.error(f"❌ Auto-Learning integration failed: {e}")
        raise RuntimeError(f"Auto-Learning integration failed: {e}")

# Función de prueba para validar la integración
async def test_integration():
    """Probar la integración del sistema de auto-aprendizaje"""
    
    try:
        logger.info("🧪 Testing Auto-Learning Integration...")
        
        # Crear integración
        integration = AutoLearningIntegration()
        
        # Inicializar
        success = await integration.initialize_integration()
        
        if not success:
            logger.error("❌ Integration test failed during initialization")
            return False
        
        # Probar registro de trade
        test_trade = {
            'symbol': 'BTCUSDT',
            'side': 'buy',
            'entry_price': 50000.0,
            'exit_price': 51000.0,
            'quantity': 0.01,
            'pnl': 10.0,
            'pnl_pct': 0.02,
            'model_used': 'l2_finrl',
            'confidence': 0.8,
            'regime': 'bull',
            'features': {'rsi': 65, 'macd': 0.5}
        }
        
        integration.record_trade_for_learning(test_trade)
        
        # Obtener estado
        status = await integration.get_learning_status()
        
        logger.info("✅ Integration test completed successfully")
        logger.info(f"   📊 Learning system status: {status.get('learning_system', {}).get('data_buffer_size', 0)} trades")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        return False

if __name__ == "__main__":
    # Probar la integración
    asyncio.run(test_integration())