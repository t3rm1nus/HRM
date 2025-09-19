#!/usr/bin/env python3
"""
Test script for Grok models in L2
==================================

Permite probar y cambiar entre diferentes modelos de Grok disponibles.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from l2_tactic.config import DEFAULT_L2_CONFIG
from l2_tactic.ai_model_integration import AIModelWrapper

def test_model_switching():
    """Prueba el cambio entre modelos disponibles"""
    print("🧪 TESTING GROK MODEL SWITCHING")
    print("=" * 50)

    config = DEFAULT_L2_CONFIG

    # Mostrar modelos disponibles
    print("📋 Modelos disponibles:")
    for key, path in config.ai_model.available_models.items():
        print(f"  • {key}: {path}")

    print(f"\n🎯 Modelo actual: {config.ai_model.model_path}")

    # Probar cambio a modelo ultra-optimizado
    print("\n🔄 Cambiando a modelo ultra-optimizado...")
    success = config.ai_model.switch_model("grok_ultra_optimized")

    if success:
        print(f"✅ Cambio exitoso. Nuevo modelo: {config.ai_model.model_path}")

        # Intentar cargar el modelo
        try:
            print("🤖 Intentando cargar el modelo...")
            model_wrapper = AIModelWrapper(config.ai_model)
            if model_wrapper.model_loaded:
                print("✅ Modelo cargado correctamente")
                model_info = model_wrapper.get_model_info()
                print(f"📊 Info del modelo: {model_info}")
            else:
                print("⚠️  Modelo no pudo cargarse (archivo no existe o error)")
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
    else:
        print("❌ Error cambiando modelo")

    # Volver al modelo original
    print("\n🔄 Volviendo al modelo original...")
    config.ai_model.switch_model("grok_original")
    print(f"✅ Modelo actual: {config.ai_model.model_path}")

def test_model_info():
    """Muestra información detallada de los modelos"""
    print("\n📊 INFORMACIÓN DETALLADA DE MODELOS")
    print("=" * 50)

    config = DEFAULT_L2_CONFIG
    info = config.ai_model.get_model_info()

    print(f"Modelo actual: {info['current_model']}")
    print(f"Tipo: {info['model_type']}")
    print(f"Threshold: {info['prediction_threshold']}")

    print("\nModelos disponibles:")
    for key, path in info['available_models'].items():
        exists = Path(path.replace('*', '20231201_1200')).exists() if '*' in path else Path(path).exists()
        status = "✅" if exists else "❌"
        print(f"  {status} {key}: {path}")

def main():
    """Función principal"""
    print("🚀 L2 GROK MODEL TESTER")
    print("=" * 50)

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "switch":
            if len(sys.argv) > 2:
                model_key = sys.argv[2]
                config = DEFAULT_L2_CONFIG
                success = config.ai_model.switch_model(model_key)
                if success:
                    print(f"✅ Modelo cambiado a: {model_key}")
                    print(f"📁 Path: {config.ai_model.model_path}")
                else:
                    print(f"❌ Modelo '{model_key}' no encontrado")
            else:
                print("Uso: python test_grok_models.py switch <model_key>")
                print("Modelos disponibles: grok_original, grok_ultra_optimized, grok_ultra_optimized_timestamped")

        elif command == "info":
            test_model_info()

        elif command == "test":
            test_model_switching()

        else:
            print("Comandos disponibles:")
            print("  info    - Muestra información de modelos")
            print("  test    - Ejecuta pruebas de cambio de modelo")
            print("  switch <key> - Cambia a modelo específico")
    else:
        # Ejecutar pruebas por defecto
        test_model_info()
        test_model_switching()

if __name__ == "__main__":
    main()
