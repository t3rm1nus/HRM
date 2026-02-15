#!/usr/bin/env python3
"""
Script de depuración para mostrar variables de entorno actuales.
"""

import os

def show_env_variables():
    """Muestra las variables de entorno relevantes."""
    print("🔍 VARIABLES DE ENTORNO ACTUALES")
    print("=" * 40)
    
    env_vars = [
        'BINANCE_MODE',
        'USE_TESTNET', 
        'BINANCE_API_KEY',
        'BINANCE_API_SECRET',
        'BINANCE_TESTNET_VALIDATION',
        'BINANCE_API_PERMISSIONS',
        'BINANCE_TESTNET_URL',
        'BINANCE_STRICT_TESTNET_MODE',
        'SYMBOLS'
    ]
    
    for var in env_vars:
        value = os.getenv(var, 'NOT_SET')
        print(f"{var}: {value}")
    
    print("\n" + "=" * 40)

if __name__ == "__main__":
    show_env_variables()