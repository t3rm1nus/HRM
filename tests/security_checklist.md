# CHECKLIST DE SEGURIDAD TESTNET

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
