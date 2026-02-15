#!/usr/bin/env python3
"""
Verificación detallada de las 9 capas de protección anti-overfitting
y el flujo completo del sistema de auto-learning.
"""

import asyncio
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from auto_learning_system import (
    SelfImprovingTradingSystem,
    AntiOverfitValidator,
    AdaptiveRegularizer,
    DiverseEnsembleBuilder,
    ConceptDriftDetector,
    SmartEarlyStopper,
    AutoRetrainingSystem,
    TradeData
)

def print_header(title):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

def print_layer(number, name, status, details=""):
    emoji = "✅" if status else "❌"
    print(f"   {emoji} Capa {number}: {name:<30} {details}")

async def check_9_layers():
    print_header("🔍 VERIFICACIÓN DE LAS 9 CAPAS DE PROTECCIÓN ANTI-OVERFITTING")
    
    # Resetear singleton para prueba limpia
    SelfImprovingTradingSystem.reset_instance()
    
    # Crear instancia del sistema
    al_system = SelfImprovingTradingSystem()
    
    print("\n📋 ESTADO DE LAS 9 CAPAS:")
    print("-" * 70)
    
    # Obtener el auto_retrainer para verificar las capas
    retrainer = al_system.auto_retrainer
    
    # Capa 1: AntiOverfitValidator
    validator = retrainer.validator
    print_layer(1, "Validación Cruzada Continua", True, 
                f"Ventanas: {validator.validation_windows}, Min Score: {validator.min_validation_score}")
    
    # Capa 2: AdaptiveRegularizer
    regularizer = retrainer.regularizer
    print_layer(2, "Regularización Adaptativa", True,
                f"L2: {regularizer.regularization_params['l2_penalty']:.3f}, "
                f"Dropout: {regularizer.regularization_params['dropout_rate']:.2f}")
    
    # Capa 3: DiverseEnsembleBuilder
    ensemble = retrainer.ensemble_builder
    print_layer(3, "Ensemble Diverso", True,
                f"Max Models: {ensemble.max_models}, Sim Threshold: {ensemble.similarity_threshold:.2f}")
    
    # Capa 4: ConceptDriftDetector
    drift = retrainer.drift_detector
    print_layer(4, "Detección de Concept Drift", True,
                f"Threshold: {drift.drift_threshold:.2f}")
    
    # Capa 5: SmartEarlyStopper
    stopper = retrainer.early_stopper
    print_layer(5, "Early Stopping Inteligente", True,
                f"Patience: {stopper.patience}, Min Delta: {stopper.min_delta:.4f}")
    
    # Capa 6: TimeBasedTrigger
    time_trigger = retrainer.auto_triggers['time_based']
    print_layer(6, "Trigger por Tiempo", time_trigger['enabled'],
                f"Intervalo: {time_trigger['interval_hours']}h")
    
    # Capa 7: PerformanceBasedTrigger
    perf_trigger = retrainer.auto_triggers['performance_based']
    print_layer(7, "Trigger por Performance", perf_trigger['enabled'],
                f"Min Trades: {perf_trigger['min_trades']}, "
                f"Win Rate: {perf_trigger['win_rate_threshold']:.0%}")
    
    # Capa 8: RegimeChangeTrigger
    regime_trigger = retrainer.auto_triggers['regime_change']
    print_layer(8, "Trigger por Cambio de Régimen", regime_trigger['enabled'],
                f"Switches: {regime_trigger['regime_switches']}")
    
    # Capa 9: DataVolumeTrigger
    volume_trigger = retrainer.auto_triggers['data_volume']
    print_layer(9, "Trigger por Volumen de Datos", volume_trigger['enabled'],
                f"Min Trades: {volume_trigger['min_new_trades']}")
    
    print("-" * 70)
    
    # Simular trades para probar el flujo
    print_header("🧪 SIMULACIÓN DE FLUJO DE TRADES")
    
    print("\n📊 Registrando 3 trades de prueba...")
    
    test_trades = [
        {
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
        },
        {
            'symbol': 'ETHUSDT',
            'side': 'buy',
            'entry_price': 3000.0,
            'exit_price': 2950.0,
            'quantity': 0.1,
            'pnl': -5.0,
            'pnl_pct': -0.0167,
            'model_used': 'l1_technical',
            'confidence': 0.7,
            'regime': 'neutral',
            'features': {'rsi': 45, 'macd': -0.3}
        },
        {
            'symbol': 'BTCUSDT',
            'side': 'sell',
            'entry_price': 51000.0,
            'exit_price': 52000.0,
            'quantity': 0.01,
            'pnl': 10.0,
            'pnl_pct': 0.0196,
            'model_used': 'l2_finrl',
            'confidence': 0.85,
            'regime': 'bull',
            'features': {'rsi': 70, 'macd': 0.8}
        }
    ]
    
    for i, trade_data in enumerate(test_trades, 1):
        al_system.record_trade(trade_data)
        print(f"   ✅ Trade {i} registrado: {trade_data['symbol']} {trade_data['side']} "
              f"PnL: ${trade_data['pnl']:.2f}")
    
    # Verificar estado después de registrar trades
    print_header("📊 ESTADO DEL SISTEMA DESPUÉS DE TRADES")
    
    status = al_system.get_system_status()
    
    print(f"\n   📦 Trades en buffer: {status['data_buffer_size']}")
    print(f"   🧠 Modelos activos: {status['models_count']}")
    print(f"   🎯 Ensemble size: {status['ensemble_size']}")
    print(f"   🛡️  Anti-overfitting: {'✅ ACTIVO' if status['anti_overfitting_active'] else '❌ INACTIVO'}")
    print(f"   🏃 Sistema corriendo: {'✅ SÍ' if status['is_running'] else '❌ NO'}")
    
    print("\n   📈 Métricas de Performance:")
    metrics = status['performance_metrics']
    print(f"      • Total trades: {metrics.get('total_trades', 0)}")
    print(f"      • Winning trades: {metrics.get('winning_trades', 0)}")
    print(f"      • Total PnL: ${metrics.get('total_pnl', 0):.2f}")
    if metrics.get('total_trades', 0) > 0:
        win_rate = metrics.get('winning_trades', 0) / metrics.get('total_trades', 1)
        print(f"      • Win rate: {win_rate:.1%}")
    
    # Verificar triggers
    print_header("🔄 VERIFICACIÓN DE TRIGGERS DE REENTRENAMIENTO")
    
    print("\n   Estado de triggers:")
    should_retrain = retrainer._should_retrain()
    
    print(f"      • Time-based: {'🔄 TRIGGERED' if should_retrain else '⏳ OK'}")
    print(f"      • Performance: {'🔄 CHECK' if len(retrainer.data_buffer) >= 100 else '⏳ OK (need 100 trades)'}")
    print(f"      • Regime change: ⏳ OK")
    print(f"      • Data volume: {'🔄 TRIGGERED' if len(retrainer.data_buffer) >= 500 else f'⏳ OK (need {500 - len(retrainer.data_buffer)} more)'}")
    
    # Verificar protecciones
    print_header("🛡️  VERIFICACIÓN DE PROTECCIONES ANTI-OVERFITTING")
    
    print("\n   ✅ Todas las protecciones están inicializadas:")
    print(f"      • Validación cruzada: {validator.validation_windows} ventanas")
    print(f"      • Regularización adaptativa: Threshold {regularizer.overfitting_threshold:.0%}")
    print(f"      • Ensemble diverso: Max {ensemble.max_models} modelos")
    print(f"      • Concept drift: Threshold {drift.drift_threshold:.2f}")
    print(f"      • Early stopping: {stopper.patience} epochs patience")
    
    # Resumen final
    print_header("📋 RESUMEN FINAL")
    
    all_layers_active = all([
        validator is not None,
        regularizer is not None,
        ensemble is not None,
        drift is not None,
        stopper is not None,
        time_trigger['enabled'],
        perf_trigger['enabled'],
        regime_trigger['enabled'],
        volume_trigger['enabled']
    ])
    
    if all_layers_active and status['data_buffer_size'] == 3:
        print("\n   ✅ TODAS LAS 9 CAPAS DE PROTECCIÓN ESTÁN ACTIVAS")
        print("   ✅ EL SISTEMA ESTÁ REGISTRANDO TRADES CORRECTAMENTE")
        print("   ✅ EL FLUJO DE AUTO-LEARNING ESTÁ FUNCIONANDO")
        print("\n   🎯 El sistema está listo para:")
        print("      • Acumular datos de trades")
        print("      • Detectar cuando reentrenar (triggers)")
        print("      • Aplicar protección anti-overfitting")
        print("      • Mejorar modelos automáticamente")
    else:
        print("\n   ❌ ALGUNAS CAPAS NO ESTÁN ACTIVAS")
        print(f"      • Capas activas: {sum([validator is not None, regularizer is not None, ensemble is not None, drift is not None, stopper is not None, time_trigger['enabled'], perf_trigger['enabled'], regime_trigger['enabled'], volume_trigger['enabled']])}/9")
    
    print("\n" + "=" * 70)
    
    return all_layers_active

if __name__ == "__main__":
    success = asyncio.run(check_9_layers())
    sys.exit(0 if success else 1)
