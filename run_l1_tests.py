#!/usr/bin/env python3
"""
Script para ejecutar las pruebas de L1 desde el directorio raíz del proyecto.
"""

import sys
import os

# Agregar el directorio actual al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # Ejecutar las pruebas de L1
    from l1_operational.test_clean_l1 import main
    
    print("🔧 Ejecutando pruebas de L1 desde el directorio raíz...")
    main()
    
except ImportError as e:
    print(f"❌ Error de importación: {e}")
    print("\n💡 Soluciones:")
    print("   1. Asegúrate de estar en el directorio raíz del proyecto")
    print("   2. Verifica que todos los archivos de L1 estén presentes")
    print("   3. Instala las dependencias: pip install -r l1_operational/requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error ejecutando las pruebas: {e}")
    sys.exit(1)
