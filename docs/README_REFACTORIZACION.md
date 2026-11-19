# HRM — Refactorización inicial de main

## Archivos

- `main.py`: conserva la orquestación del runtime y el ciclo de trading. Importa los módulos extraídos.
- `api/app.py`: contiene la aplicación FastAPI, middleware CORS, routers, modelos y endpoints.
- `system/main_helpers.py`: contiene las funciones auxiliares de actualización NAV, cierre auxiliar y comparación de cartera.

## Integración

Copia estos archivos sobre la raíz del repositorio respetando las rutas del ZIP. No sobrescribas otros archivos.

## Alcance de esta entrega

Se han movido bloques existentes sin cambiar sus cuerpos ni el orden del ciclo de trading, inicialización, recuperación o shutdown. La API mantiene sus rutas `/ws`, `/health`, `/status`, `/portfolio` y `/version`, además de los tres routers existentes y la configuración CORS.

## Validación realizada

- Compilación sintáctica con `python -m py_compile` de los tres módulos.
- Comparación AST de la función `main()` original y la refactorizada: cuerpo idéntico.
- Comparación AST de las tres funciones auxiliares extraídas: cuerpos idénticos.

No se ha ejecutado un arranque integral contra Binance/PAPER desde este entorno. Tras copiar, validar en tu máquina con el mismo entorno PAPER habitual.
