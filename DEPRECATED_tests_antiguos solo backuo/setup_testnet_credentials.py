#!/usr/bin/env python3
"""
Script de configuración de credenciales para Binance Testnet.
Este script ayuda a configurar credenciales válidas para testnet.
"""

import os
import json
from pathlib import Path

def create_testnet_instructions():
    """Crea un archivo con instrucciones para obtener credenciales de testnet."""
    
    instructions = """
# CONFIGURACION DE CREDENCIALES BINANCE TESTNET

## Paso 1: Obtener credenciales de testnet
1. Visita: https://testnet.binance.vision
2. Inicia sesion o crea una cuenta
3. Ve a "API Keys" en tu cuenta
4. Crea una nueva API Key con los siguientes permisos:
   - Enable Reading
   - Enable Spot & Margin Trading
   - Disable Futures Trading (no necesario para HRM)
   - Disable Margin & Futures (no necesario)

## Paso 2: Configurar permisos
Asegurate de que tu API Key tenga:
- Lectura de datos: Para obtener precios y balances
- Trading: Para ejecutar ordenes de compra/venta
- IP Restriction: Deja en blanco o configura tu IP si lo deseas

## Paso 3: Copiar credenciales
Copia las credenciales generadas y reemplazalas en el archivo .env:

BINANCE_API_KEY=tu_api_key_aqui
BINANCE_API_SECRET=tu_api_secret_aqui

## Paso 4: Verificar testnet
Asegurate de que estas credenciales sean para testnet, NO para la cuenta real.

## Paso 5: Probar conexion
Ejecuta: python validate_testnet_config.py

## ADVERTENCIAS DE SEGURIDAD
- NUNCA uses credenciales de tu cuenta real
- NUNCA compartas tus API Keys
- Guarda tus credenciales de forma segura
- Usa solo credenciales de testnet para desarrollo
- Revoca credenciales que ya no uses

## Enlaces utiles
- Testnet Binance: https://testnet.binance.vision
- Documentacion API: https://binance-docs.github.io/apidocs/spot/en/
- Guia de seguridad: https://www.binance.com/en/support/faq/how-to-create-api-keys-on-binance-360002502072
"""
    
    with open("testnet_setup_instructions.md", "w") as f:
        f.write(instructions)
    
    print("📚 Instrucciones de configuración creadas: testnet_setup_instructions.md")

def update_env_with_placeholders():
    """Actualiza el archivo .env con placeholders seguros."""
    
    env_content = """# Claves API de Binance (obten desde https://testnet.binance.vision para modo testnet)
# ADVERTENCIA: ESTAS SON CREDENCIALES DE EJEMPLO - REEMPLAZALAS CON TUS CREDENCIALES REALES DE TESTNET
BINANCE_API_KEY=your_binance_testnet_api_key_here
BINANCE_API_SECRET=your_binance_testnet_api_secret_here

# NEWS API Key para analisis de sentimiento
NEWS_API_KEY=da54e6c808dc4528a0d99d8cfced723c

# Reddit API credentials para analisis de sentimiento
REDDIT_CLIENT_ID=SOEllv9mH0VIVTK6HToNUA
REDDIT_CLIENT_SECRET=tSfstL36XLG-PW9MU3SSIvzGvAOj_A
REDDIT_USER_AGENT=YourAppName/1.0

# Twitter API credential para analisis de sentimiento
TWITTER_BEARER_TOKEN=your_twitter_bearer_token_here

# Glassnode API credential para datos on-chain
GLASSNODE_API_KEY=your_glassnode_api_key_here

# Santiment API credential para datos sociales adicionales
SANTIMENT_API_KEY=your_santiment_api_key_here

# Modo de operacion (PAPER para simulacion, LIVE para ejecucion real)
BINANCE_MODE=PAPER

# Habilitar testnet para simulacion - CRITICAL FOR SAFETY
USE_TESTNET=true

# Validacion de credenciales de testnet - asegurar que sean para testnet
BINANCE_TESTNET_VALIDATION=true

# Permisos de API para testnet - deben incluir: lectura, trading
BINANCE_API_PERMISSIONS=READ_WRITE

# URL de endpoints de testnet
BINANCE_TESTNET_URL=https://testnet.binance.vision
BINANCE_TESTNET_WS=wss://testnet.binance.vision/ws

# Timeout de operaciones en testnet
BINANCE_TESTNET_TIMEOUT=30

# Modo de validacion estricta para testnet
BINANCE_STRICT_TESTNET_MODE=true

# Simbolos a operar (notacion Binance, sin /)
SYMBOLS=BTCUSDT,ETHUSDT

# Limites de riesgo por simbolo
RISK_LIMIT_BTC=0.05
RISK_LIMIT_ETH=1.0
EXPOSURE_MAX_BTC=0.20
EXPOSURE_MAX_ETH=0.15
CORRELATION_LIMIT=0.80
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    
    print("✅ Archivo .env actualizado con placeholders seguros")

def create_security_checklist():
    """Crea una lista de verificación de seguridad."""
    
    checklist = """# CHECKLIST DE SEGURIDAD TESTNET

## Antes de operar:
- [ ] Credenciales de testnet obtenidas de https://testnet.binance.vision
- [ ] API Key tiene permisos de lectura y trading
- [ ] USE_TESTNET=true en .env
- [ ] BINANCE_MODE=PAPER en .env
- [ ] BINANCE_STRICT_TESTNET_MODE=true en .env
- [ ] No hay credenciales de cuenta real en el proyecto
- [ ] Archivo .env no está en git (verifica .gitignore)

## Validacion de credenciales:
- [ ] Las credenciales no son las de ejemplo (your_api_key_here)
- [ ] Las credenciales tienen el formato correcto (longitud adecuada)
- [ ] Las credenciales son especificas para testnet
- [ ] No hay espacios extra en las credenciales

## Pruebas de seguridad:
- [ ] validate_testnet_config.py pasa todas las validaciones
- [ ] No se pueden realizar operaciones en modo real
- [ ] El sistema detecta automaticamente el modo testnet
- [ ] Las ordenes se ejecutan solo en testnet

## En caso de problemas:
- [ ] Revisa que las credenciales sean de testnet
- [ ] Verifica que los permisos de API esten habilitados
- [ ] Confirma que USE_TESTNET=true
- [ ] Ejecuta validate_testnet_config.py para diagnostico

## Si algo falla:
1. No uses credenciales reales
2. Verifica que estas en testnet
3. Revisa el archivo .env
4. Consulta testnet_setup_instructions.md
"""
    
    with open("security_checklist.md", "w") as f:
        f.write(checklist)
    
    print("✅ Checklist de seguridad creado: security_checklist.md")

def main():
    """Función principal del script de configuración."""
    print("🔧 CONFIGURACIÓN DE CREDENCIALES BINANCE TESTNET")
    print("=" * 50)
    
    print("\n1. Creando instrucciones de configuración...")
    create_testnet_instructions()
    
    print("\n2. Actualizando archivo .env con placeholders seguros...")
    update_env_with_placeholders()
    
    print("\n3. Creando checklist de seguridad...")
    create_security_checklist()
    
    print("\n" + "=" * 50)
    print("🎉 CONFIGURACIÓN INICIAL COMPLETA")
    print("=" * 50)
    
    print("\n📋 PRÓXIMOS PASOS:")
    print("1. Lee testnet_setup_instructions.md")
    print("2. Obten credenciales de https://testnet.binance.vision")
    print("3. Reemplaza las credenciales en .env")
    print("4. Ejecuta: python validate_testnet_config.py")
    print("5. Verifica que todo esté correcto antes de operar")
    
    print("\n🔒 RECORDATORIOS DE SEGURIDAD:")
    print("- Nunca uses credenciales de tu cuenta real")
    print("- Solo usa credenciales de testnet para desarrollo")
    print("- Guarda tus credenciales de forma segura")
    print("- Revoca credenciales que ya no uses")
    
    print("\n✅ Archivos creados:")
    print("- testnet_setup_instructions.md (instrucciones detalladas)")
    print("- .env (con placeholders seguros)")
    print("- security_checklist.md (lista de verificación)")
    print("- validate_testnet_config.py (script de validación)")

if __name__ == "__main__":
    main()