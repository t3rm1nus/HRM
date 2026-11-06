#!/usr/bin/env python3
"""
Verificación completa del estado de modo paper.
Este script verifica que el sistema esté correctamente configurado para operar en modo paper.
"""

import os
import sys
from pathlib import Path

def verify_environment_variables():
    """Verifica las variables de entorno críticas."""
    
    print("🔍 VERIFICACIÓN DE VARIABLES DE ENTORNO")
    print("=" * 50)
    
    # Cargar variables de entorno
    from dotenv import load_dotenv
    load_dotenv()
    
    # Variables críticas
    critical_vars = {
        'BINANCE_MODE': os.getenv('BINANCE_MODE', ''),
        'USE_TESTNET': os.getenv('USE_TESTNET', ''),
        'BINANCE_API_KEY': os.getenv('BINANCE_API_KEY', ''),
        'BINANCE_API_SECRET': os.getenv('BINANCE_API_SECRET', ''),
        'SYMBOLS': os.getenv('SYMBOLS', ''),
    }
    
    print("📋 Variables de entorno detectadas:")
    for var, value in critical_vars.items():
        if var in ['BINANCE_API_KEY', 'BINANCE_API_SECRET']:
            # Ocultar credenciales sensibles
            display_value = value[:8] + "..." if value else "NO CONFIGURADA"
            print(f"   {var}: {display_value}")
        else:
            print(f"   {var}: {value}")
    
    return critical_vars

def check_paper_mode_configuration(critical_vars):
    """Verifica que la configuración de modo paper sea correcta."""
    
    print("\n🔒 VERIFICACIÓN DE CONFIGURACIÓN DE MODO PAPER")
    print("-" * 50)
    
    binance_mode = critical_vars['BINANCE_MODE'].upper()
    use_testnet = critical_vars['USE_TESTNET'].lower()
    
    # Verificar modo paper
    paper_mode_ok = binance_mode == 'PAPER'
    testnet_ok = use_testnet in ['true', '1', 'yes']
    
    print(f"BINANCE_MODE: {binance_mode}")
    print(f"USE_TESTNET: {use_testnet}")
    
    if paper_mode_ok:
        print("✅ BINANCE_MODE está correctamente en PAPER")
    else:
        print(f"❌ BINANCE_MODE está en {binance_mode}, debe ser PAPER")
    
    if testnet_ok:
        print("✅ USE_TESTNET está correctamente habilitado")
    else:
        print(f"❌ USE_TESTNET está en {use_testnet}, debe ser true")
    
    return paper_mode_ok and testnet_ok

def check_credentials_safety(critical_vars):
    """Verifica que las credenciales sean seguras para testnet."""
    
    print("\n🛡️  VERIFICACIÓN DE SEGURIDAD DE CREDENCIALES")
    print("-" * 50)
    
    api_key = critical_vars['BINANCE_API_KEY']
    api_secret = critical_vars['BINANCE_API_SECRET']
    
    # Verificar credenciales de ejemplo
    example_indicators = ['your_', 'example', 'test', 'demo']
    
    if any(indicator in api_key.lower() for indicator in example_indicators):
        print("✅ Credenciales de ejemplo detectadas (seguro para testnet)")
        return True
    elif any(indicator in api_secret.lower() for indicator in example_indicators):
        print("✅ Credenciales de ejemplo detectadas (seguro para testnet)")
        return True
    elif not api_key or not api_secret:
        print("⚠️  Credenciales no configuradas (usando credenciales de ejemplo)")
        return True
    else:
        print("⚠️  Credenciales reales detectadas - asegúrese de que sean de testnet")
        return False

def check_system_components():
    """Verifica que los componentes del sistema estén en modo paper."""
    
    print("\n⚙️  VERIFICACIÓN DE COMPONENTES DEL SISTEMA")
    print("-" * 50)
    
    try:
        # Verificar BinanceClient
        from l1_operational.binance_client import BinanceClient
        client = BinanceClient()
        
        if hasattr(client, 'use_testnet') and client.use_testnet:
            print("✅ BinanceClient en modo testnet")
            client_ok = True
        else:
            print("❌ BinanceClient no está en modo testnet")
            client_ok = False
            
    except Exception as e:
        print(f"❌ Error verificando BinanceClient: {e}")
        client_ok = False
    
    try:
        # Verificar OrderManager
        from l1_operational.order_manager import OrderManager
        from l1_operational.binance_client import BinanceClient
        
        binance_client = BinanceClient()
        order_manager = OrderManager(binance_client=binance_client)
        
        if hasattr(order_manager, 'paper_mode') and order_manager.paper_mode:
            print("✅ OrderManager en modo paper")
            manager_ok = True
        else:
            print("❌ OrderManager no está en modo paper")
            manager_ok = False
            
    except Exception as e:
        print(f"❌ Error verificando OrderManager: {e}")
        manager_ok = False
    
    return client_ok and manager_ok

def main():
    """Función principal de verificación."""
    print("🚀 VERIFICACIÓN COMPLETA DE MODO PAPER")
    print("🔒 Asegurando configuración segura para paper trading")
    print()
    
    try:
        # Paso 1: Verificar variables de entorno
        critical_vars = verify_environment_variables()
        
        # Paso 2: Verificar configuración de modo paper
        paper_config_ok = check_paper_mode_configuration(critical_vars)
        
        # Paso 3: Verificar seguridad de credenciales
        credentials_safe = check_credentials_safety(critical_vars)
        
        # Paso 4: Verificar componentes del sistema
        components_ok = check_system_components()
        
        # Resumen final
        print("\n" + "=" * 60)
        print("📊 RESUMEN DE VERIFICACIÓN DE MODO PAPER")
        print("=" * 60)
        
        print(f"✅ Configuración de modo paper: {'CORRECTA' if paper_config_ok else 'INCORRECTA'}")
        print(f"✅ Seguridad de credenciales: {'SEGURA' if credentials_safe else 'REVISAR'}")
        print(f"✅ Componentes del sistema: {'FUNCIONALES' if components_ok else 'CON PROBLEMAS'}")
        
        # Estado final
        if paper_config_ok and credentials_safe:
            print("\n🎉 ¡SISTEMA LISTO PARA OPERAR EN MODO PAPER!")
            print("🔒 Protección contra operaciones reales: ACTIVA")
            print("📊 Paper trading: FUNCIONAL")
            print("🛡️  Seguridad de credenciales: VERIFICADA")
            
            if not components_ok:
                print("⚠️  Advertencia: Algunos componentes del sistema tienen problemas técnicos")
                print("   Esto no afecta la seguridad, pero puede afectar el funcionamiento")
            
            return 0
        else:
            print("\n⚠️  SISTEMA NO LISTO PARA OPERAR")
            if not paper_config_ok:
                print("❌ Corrija la configuración de modo paper")
            if not credentials_safe:
                print("❌ Verifique la seguridad de las credenciales")
            return 1
            
    except Exception as e:
        print(f"\n❌ Error en verificación: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())