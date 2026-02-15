#!/usr/bin/env python3
"""Verificar estado del sistema de auto-learning"""

import asyncio
import sys
import os
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from integration_auto_learning import AutoLearningIntegration
from auto_learning_system import SelfImprovingTradingSystem

# Try to import storage module for trade log checking
try:
    from storage.paper_trade_logger import get_paper_logger, PAPER_LOGGER_AVAILABLE
except ImportError:
    PAPER_LOGGER_AVAILABLE = False

async def check_status():
    print("=" * 70)
    print("🔍 VERIFICACIÓN DEL SISTEMA DE AUTO-LEARNING HRM")
    print("=" * 70)
    
    # Check if there's a running system by looking for trade log
    trades_from_log = 0
    if PAPER_LOGGER_AVAILABLE:
        try:
            paper_logger = get_paper_logger()
            session_summary = paper_logger.get_session_summary()
            trades_from_log = session_summary.get('total_trades', 0)
            print(f"\n📊 Trades registrados en PaperTradeLogger: {trades_from_log}")
        except Exception as e:
            print(f"\n⚠️  No se pudo acceder a PaperTradeLogger: {e}")
    
    # Verificar sistema principal
    try:
        al_system = SelfImprovingTradingSystem.get_instance()
        
        # Try to get async status first (more accurate when system is running)
        try:
            status = await al_system.get_system_status_async()
            print("   ✅ Usando get_system_status_async() - sistema posiblemente en ejecución")
        except Exception:
            # Fallback to sync status
            status = al_system.get_system_status()
            print("   ℹ️  Usando get_system_status() - sistema no está en ejecución")
        
        print("\n📊 Estado del Sistema:")
        print(f"   🏃 Running: {'✅' if status['is_running'] else '❌'} {status['is_running']}")
        print(f"   📦 Buffer size: {status['data_buffer_size']} trades")
        print(f"   🧠 Modelos activos: {status['models_count']}")
        print(f"   🎯 Ensemble size: {status['ensemble_size']}")
        print(f"   🛡️  Anti-overfitting: {'✅ ACTIVO' if status['anti_overfitting_active'] else '❌ INACTIVO'}")
        
        print("\n📈 Métricas de Performance:")
        metrics = status['performance_metrics']
        print(f"   Total trades: {metrics.get('total_trades', 0)}")
        print(f"   Winning trades: {metrics.get('winning_trades', 0)}")
        if metrics.get('total_trades', 0) > 0:
            win_rate = metrics.get('winning_trades', 0) / metrics.get('total_trades', 1)
            print(f"   Win rate: {win_rate:.2%}")
        else:
            print(f"   Win rate: N/A")
        print(f"   Total PnL: ${metrics.get('total_pnl', 0):.2f}")
        print(f"   Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        
        print("\n🔗 Integración de Componentes:")
        integration = status['integration']
        print(f"   State Manager:     {'✅' if integration['state_manager'] else '❌'}")
        print(f"   Order Manager:     {'✅' if integration['order_manager'] else '❌'}")
        print(f"   Portfolio Manager: {'✅' if integration['portfolio_manager'] else '❌'}")
        print(f"   L2 Processor:      {'✅' if integration['l2_processor'] else '❌'}")
        print(f"   Trading Metrics:   {'✅' if integration['trading_metrics'] else '❌'}")
        
        # Verificar si puede entrenar
        can_train, reason = al_system.can_train()
        print(f"\n🎓 Entrenamiento:")
        print(f"   Puede entrenar: {'✅ SÍ' if can_train else '❌ NO'}")
        print(f"   Razón: {reason}")
        
        print("\n" + "=" * 70)
        
        # Diagnóstico
        if status['data_buffer_size'] == 0 and trades_from_log == 0:
            print("⚠️  ADVERTENCIA CRÍTICA: No hay trades registrados")
            print("   El sistema de auto-learning no está recibiendo datos de trades.")
            print("   \n   Causas probables:")
            print("   1. El sistema HRM no está ejecutándose")
            print("   2. No se han generado señales de trading")
            print("   3. Las órdenes no están siendo ejecutadas")
            print("   4. El AutoLearningBridge no está conectado")
            print("\n   Solución:")
            print("   1. Ejecutar: python main.py")
            print("   2. Esperar a que se generen señales y trades")
            print("   3. Verificar que el bridge está conectado en main.py")
            
        elif status['data_buffer_size'] == 0 and trades_from_log > 0:
            print("📝 INFO: Hay trades en PaperTradeLogger pero no en el buffer de auto-learning")
            print(f"   Trades en PaperTradeLogger: {trades_from_log}")
            print("   Esto indica que el bridge puede no estar registrando correctamente.")
            print("\n   Posibles causas:")
            print("   - El AutoLearningBridge no está conectado al TradingPipeline")
            print("   - Hay un error silencioso en el registro de trades")
            
        elif status['data_buffer_size'] < 50:
            print("⏳ ACUMULANDO DATOS:")
            print(f"   Trades actuales: {status['data_buffer_size']}")
            print(f"   Faltan {50 - status['data_buffer_size']} trades para trigger de performance")
            print(f"   Faltan {100 - status['data_buffer_size']} trades para trigger de data volume")
            print("\n   El sistema está acumulando datos. Los triggers se activarán automáticamente.")
            
        elif status['data_buffer_size'] < 500:
            print("📊 SUFICIENTES DATOS:")
            print(f"   Trades acumulados: {status['data_buffer_size']}")
            print("   Los triggers de performance pueden activarse si:")
            print("   - Win rate cae por debajo de 52%")
            print("   - Drawdown excede 12%")
            print("\n   Trigger de data volume: Necesita 500+ trades")
            
        else:
            print("✅ SISTEMA OPERATIVO:")
            print(f"   Trades acumulados: {status['data_buffer_size']}")
            print("   Todos los triggers están activos y listos.")
            print("   El sistema auto-reentrenará cuando se cumplan las condiciones.")
        
        print("=" * 70)
        
        # Recomendaciones
        print("\n💡 RECOMENDACIONES:")
        if not integration['order_manager'] or not integration['portfolio_manager']:
            print("   🔴 CRÍTICO: Falta integración con OrderManager o PortfolioManager")
            print("      → Verificar inicialización en main.py")
        
        if status['data_buffer_size'] == 0:
            print("   🔴 CRÍTICO: Implementar AutoLearningBridge (Fase 1 del plan)")
            print("      → Crear system/auto_learning_bridge.py")
            print("      → Modificar trading_pipeline_manager.py")
            print("      → Conectar en main.py")
        
        if metrics.get('total_pnl', 0) < 0:
            print("   🟡 El sistema está en pérdida - el auto-learning ayudará a mejorar")
        
        if status['anti_overfitting_active']:
            print("   🟢 Las 9 capas de protección anti-overfitting están activas")
        
        print()
        
    except Exception as e:
        print(f"\n❌ ERROR al verificar estado: {e}")
        print("   El sistema de auto-learning puede no estar inicializado.")
        print(f"   Excepción: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(check_status())
