
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
