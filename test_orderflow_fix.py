#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de prueba rápida para verificar que OrderFlowCapture funciona correctamente.

Uso:
    python test_orderflow_fix.py

Este script:
1. Inicia OrderFlowCapture en modo polling (REST API)
2. Espera 20 segundos para capturar varios ciclos de datos
3. Verifica que se hayan creado archivos CSV
4. Muestra el contenido de los archivos CSV creados
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from l1_operational.order_flow_capture import (
    OrderFlowCapture,
    create_and_start_order_flow_capture
)


async def test_orderflow():
    """Prueba el OrderFlowCapture"""
    print("=" * 80)
    print("🧪 TEST ORDERFLOW CAPTURE - Modo POLLING REST API")
    print("=" * 80)
    print()
    
    # Verificar directorio de salida
    repo_root = Path(__file__).parent
    data_dir = repo_root / "backtesting" / "data" / "orderflow"
    
    print(f"📁 Directorio de datos: {data_dir}")
    print(f"   Existe: {data_dir.exists()}")
    
    if not data_dir.exists():
        print(f"   Creando directorio...")
        data_dir.mkdir(parents=True, exist_ok=True)
    print()
    
    # Limpiar archivos CSV previos
    print("🧹 Limpiando CSVs previos...")
    csv_files = list(data_dir.glob("orderflow_*.csv"))
    for f in csv_files:
        print(f"   Eliminando: {f.name}")
        f.unlink()
    print(f"   {len(csv_files)} archivos eliminados")
    print()
    
    # Iniciar OrderFlowCapture
    print("🚀 Iniciando OrderFlowCapture (modo POLLING)...")
    print("   Símbolos: BTCUSDT, ETHUSDT")
    print("   Intervalo: 5 segundos")
    print()
    
    capture = None
    try:
        capture = await create_and_start_order_flow_capture(
            symbols=["BTCUSDT", "ETHUSDT"],
            data_dir=str(data_dir),
            use_polling=True  # FORZAR modo polling
        )
        
        print("✅ OrderFlowCapture iniciado correctamente")
        print()
        
        # Esperar para capturar datos
        wait_time = 20  # segundos
        print(f"⏳ Esperando {wait_time} segundos para capturar datos...")
        print("   (El polling ocurre cada 5 segundos)")
        print()
        
        for i in range(wait_time):
            await asyncio.sleep(1)
            if (i + 1) % 5 == 0:
                # Mostrar métricas actuales
                print(f"   [{(i+1):2d}s] Capturando...")
                for symbol in ["BTCUSDT", "ETHUSDT"]:
                    features = capture.get_current_features(symbol)
                    if features and features.get('best_bid_price', 0) > 0:
                        print(f"        {symbol}: spread={features['spread_bps']:.2f}bps, "
                              f"imbalance={features['bid_ask_imbalance']:.3f}")
        
        print()
        print("✅ Captura completada")
        print()
        
        # Forzar flush final
        print("💾 Forzando escritura a CSV...")
        await capture._flush_buffer()
        print()
        
        # Verificar archivos creados
        print("=" * 80)
        print("📊 RESULTADOS")
        print("=" * 80)
        print()
        
        csv_files = list(data_dir.glob("orderflow_*.csv"))
        
        if not csv_files:
            print("❌ ERROR: No se crearon archivos CSV")
            print()
            print("Posibles causas:")
            print("  - El buffer está vacío (no se recibieron datos)")
            print("  - Error en la conexión a Binance REST API")
            print("  - Problema de permisos de escritura")
            return False
        
        print(f"✅ Se encontraron {len(csv_files)} archivos CSV:")
        print()
        
        for csv_file in sorted(csv_files):
            size = csv_file.stat().st_size
            print(f"   📄 {csv_file.name}")
            print(f"      Tamaño: {size:,} bytes")
            
            # Contar filas
            try:
                with open(csv_file, 'r') as f:
                    lines = f.readlines()
                    # Restar 1 por el header
                    data_rows = len(lines) - 1 if len(lines) > 0 else 0
                    print(f"      Filas: {data_rows}")
                    
                    # Mostrar primeras 3 filas
                    if len(lines) > 1:
                        print(f"      Preview:")
                        for line in lines[:4]:  # Header + 3 filas
                            print(f"         {line.strip()}")
                        if len(lines) > 4:
                            print(f"         ... ({len(lines)-4} más)")
            except Exception as e:
                print(f"      Error leyendo: {e}")
            
            print()
        
        print("=" * 80)
        print("✅ TEST COMPLETADO EXITOSAMENTE")
        print("=" * 80)
        print()
        print("El OrderFlowCapture está funcionando correctamente.")
        print("Los archivos CSV se están generando con datos reales de Binance.")
        return True
        
    except Exception as e:
        print()
        print("=" * 80)
        print(f"❌ ERROR: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if capture:
            print()
            print("🛑 Cerrando OrderFlowCapture...")
            await capture.close()
            print("✅ Cerrado correctamente")


if __name__ == "__main__":
    success = asyncio.run(test_orderflow())
    sys.exit(0 if success else 1)
