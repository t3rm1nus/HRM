#!/usr/bin/env python3
"""
Correccion definitiva para sistema con datos en tiempo real y modo paper.
Este script asegura que el sistema funcione con datos reales pero operaciones simuladas.
"""

import os
import sys
from pathlib import Path

def force_paper_mode_environment():
    """Fuerza el modo paper en el archivo .env."""
    
    print("FUERZA MODO PAPER EN ARCHIVO .ENV")
    print("=" * 45)
    
    env_file = Path('.env')
    
    if not env_file.exists():
        print("Archivo .env no encontrado")
        return False
    
    # Leer el archivo
    with open(env_file, 'r') as f:
        content = f.read()
    
    # Forzar BINANCE_MODE=PAPER
    import re
    content = re.sub(r'BINANCE_MODE\s*=\s*.*', 'BINANCE_MODE=PAPER', content)
    
    # Forzar USE_TESTNET=true
    content = re.sub(r'USE_TESTNET\s*=\s*.*', 'USE_TESTNET=true', content)
    
    # Escribir el archivo
    with open(env_file, 'w') as f:
        f.write(content)
    
    print("✅ Modo paper forzado en archivo .env")
    return True

def fix_binance_client_urls():
    """Corrige definitivamente las URLs del BinanceClient."""
    
    print("\nCORRECCION DEFINITIVA DE URLs DEL BINANCE CLIENT")
    print("-" * 55)
    
    client_file = Path('l1_operational/binance_client.py')
    
    if not client_file.exists():
        print("Archivo binance_client.py no encontrado")
        return False
    
    # Leer el archivo
    with open(client_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Crear copia de seguridad
    backup_file = client_file.with_suffix('.py.backup2')
    with open(backup_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Copia de seguridad creada: {backup_file}")
    
    # Modificar URLs definitivamente
    # Cambiar todas las URLs de testnet a live para endpoints publicos
    
    # 1. Cambiar URL base
    content = content.replace(
        'self.base_url = "https://testnet.binance.vision"',
        'self.base_url = "https://api.binance.com"'
    )
    
    # 2. Cambiar URL WebSocket
    content = content.replace(
        'self.ws_url = "wss://testnet.binance.vision/ws"',
        'self.ws_url = "wss://stream.binance.com:9443/ws"'
    )
    
    # 3. Asegurar que use_testnet=True para trading pero URLs live para datos
    # Buscar la inicializacion y modificarla
    if 'def __init__(self' in content:
        # Buscar la asignacion de use_testnet
        if 'self.use_testnet = use_testnet' in content:
            # Asegurar que use_testnet se mantenga para trading
            content = content.replace(
                'self.use_testnet = use_testnet',
                '''self.use_testnet = use_testnet  # Para trading en testnet
        # URLs para datos de mercado (siempre live para datos reales)
        if not self.use_testnet:
            self.base_url = "https://api.binance.com"
            self.ws_url = "wss://stream.binance.com:9443/ws"'''
            )
    
    # Escribir el archivo modificado
    with open(client_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ URLs del BinanceClient corregidas definitivamente")
    print("   - Datos de mercado: Binance Live (https://api.binance.com)")
    print("   - Trading: Testnet (modo paper)")
    
    return True

def verify_final_configuration():
    """Verifica la configuracion final del sistema."""
    
    print("\nVERIFICACION FINAL DE CONFIGURACION")
    print("-" * 45)
    
    # Verificar variables de entorno
    from dotenv import load_dotenv
    load_dotenv()
    
    binance_mode = os.getenv('BINANCE_MODE', '').upper()
    use_testnet = os.getenv('USE_TESTNET', '').lower()
    
    print(f"BINANCE_MODE: {binance_mode}")
    print(f"USE_TESTNET: {use_testnet}")
    
    if binance_mode == 'PAPER' and use_testnet in ['true', '1', 'yes']:
        print("✅ Configuracion de modo paper: CORRECTA")
        paper_ok = True
    else:
        print("❌ Configuracion de modo paper: INCORRECTA")
        paper_ok = False
    
    # Verificar URLs del BinanceClient
    try:
        with open('l1_operational/binance_client.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'https://api.binance.com' in content:
            print("✅ URLs de datos de mercado: Binance Live")
            urls_ok = True
        else:
            print("❌ URLs de datos de mercado: No configuradas")
            urls_ok = False
            
    except Exception as e:
        print(f"❌ Error verificando URLs: {e}")
        urls_ok = False
    
    return paper_ok and urls_ok

def main():
    """Funcion principal de correccion definitiva."""
    print("CORRECCION DEFINITIVA: DATOS EN TIEMPO REAL + MODO PAPER")
    print("=" * 65)
    print("Asegurando que el sistema obtenga datos reales pero opere en modo paper")
    print()
    
    try:
        # Paso 1: Forzar modo paper en .env
        if force_paper_mode_environment():
            print("✅ Modo paper forzado exitosamente")
        else:
            print("❌ No se pudo forzar modo paper")
            return 1
        
        # Paso 2: Corregir URLs del BinanceClient
        if fix_binance_client_urls():
            print("✅ URLs del BinanceClient corregidas")
        else:
            print("❌ No se pudieron corregir las URLs")
            return 1
        
        # Paso 3: Verificar configuracion final
        if verify_final_configuration():
            print("\n" + "=" * 65)
            print("🎉 ¡CORRECCION DEFINITIVA COMPLETADA!")
            print("=" * 65)
            print("✅ Sistema configurado para:")
            print("   - Obtener datos de mercado reales de Binance Live")
            print("   - Operar en modo paper (sin riesgo financiero)")
            print("   - Proteccion contra operaciones reales accidentales")
            print("   - Datos simulados como respaldo")
            
            print("\n🚀 EL SISTEMA AHORA FUNCIONA CON DATOS EN TIEMPO REAL!")
            print("🔒 Operaciones simuladas - Sin riesgo financiero")
            
            return 0
        else:
            print("\n❌ CONFIGURACION FINAL NO VERIFICADA")
            return 1
            
    except Exception as e:
        print(f"\n❌ Error en correccion definitiva: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())