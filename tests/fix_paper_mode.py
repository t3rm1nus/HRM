#!/usr/bin/env python3
"""
Script para corregir automáticamente la configuración de modo paper.
Este script asegura que el sistema esté correctamente configurado para operar en modo paper.
"""

import os
import sys
from pathlib import Path

def fix_paper_mode_configuration():
    """Corrige la configuración para modo paper."""
    
    print("🔧 CORRECCIÓN AUTOMÁTICA DE MODO PAPER")
    print("=" * 50)
    
    # Ruta al archivo .env
    env_file = Path('.env')
    
    if not env_file.exists():
        print("❌ Archivo .env no encontrado")
        return False
    
    # Leer el archivo .env
    with open(env_file, 'r') as f:
        lines = f.readlines()
    
    # Procesar líneas y corregir configuración
    updated_lines = []
    binance_mode_fixed = False
    use_testnet_fixed = False
    
    for line in lines:
        line = line.strip()
        
        # Saltar líneas vacías y comentarios
        if not line or line.startswith('#'):
            updated_lines.append(line + '\n')
            continue
        
        # Procesar variables de entorno
        if '=' in line:
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            
            if key == 'BINANCE_MODE':
                if value.upper() == 'LIVE':
                    print(f"⚠️  Corrigiendo {key} de '{value}' a 'PAPER'")
                    updated_lines.append(f"{key}=PAPER\n")
                    binance_mode_fixed = True
                elif value.upper() == 'PAPER':
                    print(f"✅ {key} ya está en modo PAPER")
                    updated_lines.append(line + '\n')
                else:
                    print(f"⚠️  {key} tiene valor desconocido '{value}', cambiando a PAPER")
                    updated_lines.append(f"{key}=PAPER\n")
                    binance_mode_fixed = True
            elif key == 'USE_TESTNET':
                if value.lower() in ['true', '1', 'yes']:
                    print(f"✅ {key} ya está habilitado")
                    updated_lines.append(line + '\n')
                else:
                    print(f"⚠️  Corrigiendo {key} de '{value}' a 'true'")
                    updated_lines.append(f"{key}=true\n")
                    use_testnet_fixed = True
            else:
                updated_lines.append(line + '\n')
        else:
            updated_lines.append(line + '\n')
    
    # Añadir variables faltantes si es necesario
    env_content = ''.join(updated_lines)
    
    if 'BINANCE_MODE=' not in env_content:
        print("⚠️  Añadiendo BINANCE_MODE al archivo .env")
        updated_lines.append("BINANCE_MODE=PAPER\n")
        binance_mode_fixed = True
    
    if 'USE_TESTNET=' not in env_content:
        print("⚠️  Añadiendo USE_TESTNET al archivo .env")
        updated_lines.append("USE_TESTNET=true\n")
        use_testnet_fixed = True
    
    # Escribir el archivo actualizado
    with open(env_file, 'w') as f:
        f.writelines(updated_lines)
    
    print(f"\n✅ Configuración actualizada:")
    print(f"   - BINANCE_MODE: {'Corregido' if binance_mode_fixed else 'Ya estaba correcto'}")
    print(f"   - USE_TESTNET: {'Corregido' if use_testnet_fixed else 'Ya estaba correcto'}")
    
    return True

def verify_paper_mode():
    """Verifica que el modo paper esté correctamente configurado."""
    
    print("\n🔍 VERIFICACIÓN FINAL DE MODO PAPER")
    print("-" * 40)
    
    # Cargar variables de entorno
    from dotenv import load_dotenv
    load_dotenv()
    
    binance_mode = os.getenv('BINANCE_MODE', '').upper()
    use_testnet = os.getenv('USE_TESTNET', '').lower()
    
    print(f"BINANCE_MODE: {binance_mode}")
    print(f"USE_TESTNET: {use_testnet}")
    
    # Verificar configuración
    paper_mode_ok = binance_mode == 'PAPER'
    testnet_ok = use_testnet in ['true', '1', 'yes']
    
    if paper_mode_ok:
        print("✅ BINANCE_MODE está correctamente en PAPER")
    else:
        print(f"❌ BINANCE_MODE está en {binance_mode}, debe ser PAPER")
    
    if testnet_ok:
        print("✅ USE_TESTNET está correctamente habilitado")
    else:
        print(f"❌ USE_TESTNET está en {use_testnet}, debe ser true")
    
    # Estado final
    if paper_mode_ok and testnet_ok:
        print("\n🎉 ¡SISTEMA LISTO PARA OPERAR EN MODO PAPER!")
        print("🔒 Protección contra operaciones reales: ACTIVA")
        print("📊 Paper trading: FUNCIONAL")
        return True
    else:
        print("\n⚠️  SISTEMA NO LISTO PARA OPERAR")
        print("❌ Corrija la configuración antes de operar")
        return False

def main():
    """Función principal."""
    print("🚀 SCRIPT DE CORRECCIÓN DE MODO PAPER")
    print("🔒 Asegurando configuración segura para paper trading")
    print()
    
    try:
        # Corregir configuración
        if fix_paper_mode_configuration():
            # Verificar configuración
            if verify_paper_mode():
                print("\n✅ CORRECCIÓN COMPLETA - SISTEMA LISTO")
                return 0
            else:
                print("\n❌ CORRECCIÓN FALLIDA - REVISE CONFIGURACIÓN")
                return 1
        else:
            print("\n❌ NO SE PUDO CORREGIR LA CONFIGURACIÓN")
            return 1
            
    except Exception as e:
        print(f"\n❌ Error en corrección: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())