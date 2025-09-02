#!/usr/bin/env python3
"""
HRM Feature Flow Tracer
Rastrea el flujo de features desde L2 hasta L1 para diagnosticar el problema 52 vs 12
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import logging
from datetime import datetime
import sys
import os

# Añadir el directorio raíz al path para imports
sys.path.insert(0, os.path.abspath('.'))

class FeatureFlowTracer:
    def __init__(self, project_root="./"):
        self.project_root = Path(project_root)
        self.logger = self._setup_logger()
        self.symbols = ['BTCUSDT', 'ETHUSDT']
        
    def _setup_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s'
        )
        return logging.getLogger(__name__)
    
    def trace_feature_flow(self):
        """Rastrea el flujo completo de features L2 -> L1"""
        
        print("=" * 80)
        print("🔍 HRM FEATURE FLOW TRACER")
        print("=" * 80)
        
        # 1. Simular preparación de features L2
        print("\n📊 PASO 1: Simulando preparación de features L2...")
        l2_features = self._simulate_l2_features()
        
        # 2. Verificar qué esperan los modelos L1
        print("\n🤖 PASO 2: Verificando requirements de modelos L1...")
        l1_requirements = self._check_l1_model_requirements()
        
        # 3. Simular el paso de L2 a L1
        print("\n🔄 PASO 3: Simulando transferencia L2 -> L1...")
        self._simulate_l2_to_l1_transfer(l2_features, l1_requirements)
        
        # 4. Verificar pipeline de data/loaders.py
        print("\n⚙️ PASO 4: Verificando data/loaders.py...")
        self._check_data_loaders()
        
        # 5. Sugerir soluciones
        print("\n💡 PASO 5: Soluciones recomendadas...")
        self._suggest_solutions()
    
    def _simulate_l2_features(self):
        """Simula la preparación de features como en main.py"""
        
        print("   Simulando features L2 (como en main.py)...")
        
        # Simular datos básicos como en main.py
        features_by_symbol = {}
        
        for symbol in self.symbols:
            # Estas son las features que main.py prepara para L2
            features = {
                # Price features
                'price_rsi': 45.0,
                'price_macd': -10.5,
                'price_macd_signal': -8.2,
                'price_macd_hist': -2.3,
                'price_change_24h': 0.02,
                
                # Volume features
                'volume_rsi': 55.0,
                'volume_change_24h': 0.15,
                'volume_ratio': 1.2,
                
                # Bollinger Bands
                'bb_upper': 110000.0,
                'bb_middle': 109000.0, 
                'bb_lower': 108000.0,
                'bb_position': 0.3,
                
                # EMAs
                'ema_12': 109100.0,
                'ema_26': 109200.0,
                'ema_50': 109300.0,
                
                # SMAs
                'sma_20': 109150.0,
                'sma_50': 109250.0,
                
                # Multi-timeframe features (simuladas)
                **{f'tf_{tf}_{indicator}': np.random.random() 
                   for tf in ['1m', '5m', '15m', '1h'] 
                   for indicator in ['rsi', 'macd', 'bb_pos', 'volume_ratio']},
                
                # Features adicionales para llegar a 52
                **{f'feature_{i}': np.random.random() 
                   for i in range(20, 35)}
            }
            
            features_by_symbol[symbol] = features
            feature_count = len(features)
            print(f"   ✅ {symbol}: {feature_count} features preparadas")
            
            # Mostrar primeras 5 features
            first_5 = list(features.keys())[:5]
            print(f"      Primeras 5: {first_5}")
        
        return features_by_symbol
    
    def _check_l1_model_requirements(self):
        """Verifica qué esperan los modelos L1"""
        
        model_requirements = {}
        model_paths = [
            "models/L1/modelo1_lr.pkl",
            "models/L1/modelo2_rf.pkl", 
            "models/L1/modelo3_lgbm.pkl"
        ]
        
        for model_path in model_paths:
            full_path = self.project_root / model_path
            
            if not full_path.exists():
                print(f"   ❌ Modelo no encontrado: {model_path}")
                continue
                
            try:
                with open(full_path, 'rb') as f:
                    model = pickle.load(f)
                    
                model_name = model_path.split('/')[-1].replace('.pkl', '')
                
                # Extraer número de features esperadas
                expected_features = None
                if hasattr(model, 'n_features_in_'):
                    expected_features = model.n_features_in_
                elif hasattr(model, 'n_features_'):
                    expected_features = model.n_features_
                elif hasattr(model, 'feature_importances_'):
                    expected_features = len(model.feature_importances_)
                
                if expected_features:
                    model_requirements[model_name] = expected_features
                    print(f"   ✅ {model_name}: Espera {expected_features} features")
                else:
                    print(f"   ⚠️ {model_name}: No se puede determinar # features")
                    
            except Exception as e:
                print(f"   ❌ Error cargando {model_path}: {str(e)}")
        
        return model_requirements
    
    def _simulate_l2_to_l1_transfer(self, l2_features, l1_requirements):
        """Simula el paso de features de L2 a L1"""
        
        print("   Analizando transferencia de features...")
        
        # El problema está probablemente aquí
        print("   🔍 PROBLEMA IDENTIFICADO:")
        print("   - L2 prepara ~52 features por símbolo")
        print("   - Pero L1 recibe solo 12 features")
        print("   - Esto indica que hay una transformación/filtrado intermedio")
        
        # Verificar si trend_ai.py usa solo algunas features
        print("\n   🧐 Posibles causas:")
        print("   1. trend_ai.py usa solo un subset de features")
        print("   2. data/loaders.py filtra features")
        print("   3. Los modelos L1 fueron entrenados con menos features")
        print("   4. Hay una transformación en l1_operational/order_manager.py")
    
    def _check_data_loaders(self):
        """Verifica el módulo data/loaders.py"""
        
        loaders_path = self.project_root / "data" / "loaders.py"
        
        if not loaders_path.exists():
            print("   ❌ data/loaders.py no encontrado")
            return
        
        print("   📄 Analizando data/loaders.py...")
        
        try:
            # Importar y verificar función generate_features
            sys.path.insert(0, str(self.project_root))
            from data.loaders import generate_features
            
            # Crear datos de prueba
            test_data = pd.DataFrame({
                'open': [100, 101, 102, 103, 104],
                'high': [101, 102, 103, 104, 105],
                'low': [99, 100, 101, 102, 103],
                'close': [100.5, 101.5, 102.5, 103.5, 104.5],
                'volume': [1000, 1100, 1200, 1300, 1400]
            })
            
            features = generate_features(test_data)
            
            if isinstance(features, pd.DataFrame):
                feature_count = len(features.columns)
                print(f"   ✅ generate_features() produce {feature_count} features")
                print(f"      Primeras 5: {list(features.columns)[:5]}")
                
                # Verificar si produce exactamente 12 features
                if feature_count == 12:
                    print("   🎯 CAUSA ENCONTRADA: generate_features() produce exactamente 12 features")
                    print("      Esta es probablemente la causa del problema!")
                    
            else:
                print(f"   ⚠️ generate_features() retorna tipo: {type(features)}")
                
        except ImportError as e:
            print(f"   ❌ Error importando data.loaders: {e}")
        except Exception as e:
            print(f"   ❌ Error verificando data/loaders.py: {e}")
    
    def _suggest_solutions(self):
        """Sugiere soluciones al problema de features"""
        
        print("   💡 SOLUCIONES RECOMENDADAS:")
        print()
        
        print("   🔧 SOLUCIÓN INMEDIATA:")
        print("   1. Usar predict_disable_shape_check=true en LightGBM")
        print("      - Modificar l1_operational/trend_ai.py")
        print("      - Añadir parámetro predict_disable_shape_check=True")
        print()
        
        print("   🎯 SOLUCIÓN CORRECTA:")
        print("   1. Identificar exactamente qué features usan los modelos L1")
        print("   2. Modificar la preparación de features para coincidir")
        print("   3. O re-entrenar modelos L1 con las 52 features de L2")
        print()
        
        print("   📋 PASOS ESPECÍFICOS:")
        print("   1. Verificar data/loaders.py generate_features()")
        print("   2. Revisar l1_operational/trend_ai.py") 
        print("   3. Alinear features entre L2 y L1")
        print("   4. Considerar re-entrenamiento de modelos L1")

def main():
    tracer = FeatureFlowTracer()
    tracer.trace_feature_flow()

if __name__ == "__main__":
    main()