# HRM — informe de revisión y primera estabilización

Fecha: 3 de noviembre de 2026  
Base: `HRM-master (2).zip` recibido en esta conversación.

## Resultado

Se ha realizado una primera pasada de estabilización estática sobre backend y frontend. No se considera que HRM esté listo para operar en real ni que el build de frontend esté validado.

## Cambios aplicados

### Backend Python
- Corregidos los errores de sintaxis detectados en los módulos Python que impedían compilar el árbol completo.
- Eliminado el BOM que impedía importar `l1_operational/order_manager.py`.
- Reparada la indentación/importación rota de `l1_operational/signal_processor.py`.
- Reparado el final truncado de `backtesting/backtest_sensitivity_analysis.py`.
- Corregidos bloques sintácticos dañados en tests de `tests/`.
- Normalizada la codificación del módulo de integración FinRL.
- Reconstruido `api/servicios/servicio_websockets.py`, que estaba truncado.
- Añadido endpoint WebSocket `/ws` y CORS local para Vite.
- El endpoint `/status` deja de declarar falsamente que el trading está activo.
- El endpoint `/version` devuelve también la versión de Python.
- El capital inicial expuesto por API deja de inventar 500 USDT: usa `INITIAL_CAPITAL` o 0.
- Se ha retirado la limpieza destructiva automática del arranque y se conserva el historial paper entre reinicios.
- Se exige `HRM_ENABLE_LIVE_TRADING=true` además de las variables de modo para siquiera permitir la ruta live. Mantener paper por defecto.

### Frontend
- Reescrito el cliente HTTP: cabeceras Fetch correctas, timeout, token, validación Zod y normalización de errores.
- Corregidos los contratos Zod de salud, estado, cartera, versión y error.
- Corregida la URL base: la API FastAPI expone `/health`, `/status`, `/portfolio` y `/version` en raíz, no bajo `/api`.
- Reparadas las llamadas del cliente de API.
- Reescrito el router con páginas existentes del proyecto; retiradas referencias a páginas/componentes que no existían.
- Corregidas rutas del menú lateral y añadido acceso a Mercado.
- Reescrito el hook WebSocket para evitar reconexiones en bucle y dobles declaraciones.
- Eliminada la generación de cotizaciones aleatorias en la página Mercado. Ahora informa claramente que falta una fuente real conectada.
- Reparadas funciones de cálculo de rendimiento y fechas con errores de sintaxis/lógica.
- Corregidos varios imports, tipos y contratos del frontend.

### Configuración
- Añadido `requirements.txt` raíz de dependencias base.
- Añadido `.env.example` y `frontend/.env.example`.
- Añadida protección de `.env` al `.gitignore`.
- Añadida copia `l2_tactic/requirements.txt` con el nombre correcto del manifiesto que antes se llamaba `requeriments.txt`.

## Validaciones realizadas

- Compilación sintáctica de todos los archivos Python: **348 archivos revisados, 0 errores de sintaxis**.
- TypeScript: se ha ejecutado `tsc --noEmit`, pero no ha sido posible instalar dependencias con `npm ci` en el entorno de revisión (timeout). Por ello, no se puede certificar el build.
- La salida estática de TypeScript revela problemas adicionales pendientes en módulos de gráficas, utilidades, stores y páginas heredadas. Entre ellos hay componentes que usan una API de `lightweight-charts` incompatible con la versión declarada, imports faltantes y tests que no concuerdan con las implementaciones.
- No se han ejecutado pruebas de integración contra Binance, ni paper trading prolongado.
- No se ha probado el ciclo completo L1 → L2 → L3 → validación → ejecución → persistencia.

## Bloqueos que siguen abiertos

1. **Integración real de mercado:** no hay un endpoint de mercado conectado al frontend. La pantalla Mercado se deja sin cotizaciones inventadas hasta completar esa conexión.
2. **Gráficas:** hay componentes heredados con uso incorrecto de `lightweight-charts` (por ejemplo, referencias a `Widget`/`renderer()`), que requieren una migración coherente a la API instalada o a Recharts.
3. **Router y páginas heredadas:** se ha reparado el router principal, pero quedan archivos de una antigua interfaz de gestión/CRM y componentes no usados que aún generan errores de tipado.
4. **Contrato de cartera:** el endpoint de cartera sigue devolviendo valores vacíos porque no está conectado a un gestor de cartera compartido con el runtime.
5. **Estado del runtime:** `/status` es conservador y devuelve pausado; falta publicar el estado real del ciclo de trading en un almacén compartido.
6. **Eventos WebSocket:** existe canal de conexión, pero falta publicar desde los productores del runtime los eventos de mercado, señales, órdenes y cartera. No se debe interpretar una conexión WebSocket como prueba de que haya eventos reales.
7. **Dependencias:** el manifiesto raíz es una base de instalación, no un lock reproducible. Debe resolverse la matriz de versiones CPU/GPU para PyTorch, FinRL y Transformers.
8. **Modelos:** faltan artefactos de modelos y datasets en el ZIP; no pueden inferirse ni reconstruirse a partir de los nombres de los proveedores.
9. **Pruebas:** hace falta corregir los tests heredados y ejecutar unitarias, integración y E2E con dependencias instaladas.

## Estado del proyecto

**Fase actual: estabilización inicial de sintaxis, contratos y arranque seguro.**

Siguiente fase recomendada:
1. Instalar dependencias y cerrar el build TypeScript.
2. Terminar la migración de gráficas y eliminar páginas heredadas no usadas.
3. Conectar el estado real de runtime, cartera y datos de mercado a la API.
4. Publicar eventos desde el bus de mensajes al WebSocket.
5. Ejecutar pruebas de integración en paper con datos reales de mercado.
6. Auditar órdenes, riesgo, reinicios y persistencia antes de cualquier evaluación de live.

**No activar trading real con esta versión.** La corrección de sintaxis y contratos no demuestra rentabilidad ni seguridad financiera.
