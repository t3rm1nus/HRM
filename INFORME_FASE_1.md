# HRM — Informe de validación de Fase 1

**Fecha:** 5 de noviembre de 2026  
**Base examinada:** `HRM-master (1).zip` recibido en esta conversación.  
**Alcance:** estabilización técnica inicial de backend Python y frontend React/TypeScript.  
**Resultado global:** **FASE 1 INCOMPLETA — no cumple todavía los criterios de aceptación.**

## 1. Resumen ejecutivo

Se ha realizado una primera pasada reproducible sobre una copia de trabajo del ZIP. No se ha modificado el ZIP original ni se ha iniciado HRM, porque el entorno no dispone de todas las dependencias necesarias y el arranque podría interactuar con el estado persistente incluido en el archivo.

No se han modificado estrategias, señales, indicadores, dimensionamiento, gestión del riesgo, configuración PAPER/LIVE ni lógica de ejecución. No se han habilitado operaciones LIVE.

La compilación sintáctica de Python finaliza correctamente. Se han ejecutado 32 tests focalizados relacionados con perfiles PAPER, persistencia aislada y rutas de sesión/operaciones: todos pasan. La suite completa no puede considerarse ejecutable todavía: hay scripts de diagnóstico que realizan trabajo durante la importación y fallan la colección, y otras pruebas requieren dependencias opcionales no instaladas.

La instalación del frontend no se ha podido completar: `npm ci` agotó el tiempo disponible y el intento offline confirmó que falta al menos el paquete `zustand@5.0.16` en la caché. Por ello no se han ejecutado typecheck, tests ni build.

## 2. Validaciones ejecutadas

| Comprobación | Resultado | Evidencia / observación |
|---|---|---|
| Compilación sintáctica Python | **Correcta** | `python -m compileall -q -f .` terminó con código 0 sobre el árbol extraído. Se contabilizaron 369 archivos `.py` en el repositorio. |
| Tests PAPER focalizados | **Correctos** | `python -m pytest tests/test_preparar_sesion.py tests/test_perfil_datos.py tests/api/test_rutas_operaciones.py tests/api/test_rutas_modo_sesion.py -q --disable-warnings --maxfail=1` → 32 passed. |
| Suite Python completa | **Bloqueada** | La colección importa `test_l1_diagnostic.py`, que ejecuta diagnóstico y `sys.exit(1)` al no encontrar `backtesting/data/btcusdt_1m.csv`. |
| Colección `tests/` | **Bloqueada** | `tests/quick_log_test.py` ejecuta una consulta SQLite al importar y falla con `no such table: logs`. |
| Dependencias ML/sentimiento | **Incompletas en el entorno** | No están instalados `transformers`, `stable_baselines3`, `asyncpraw` ni `yfinance`. Una prueba falla al importar `transformers`. |
| Dependencias backend base | **Incompletas en el entorno** | No están instalados `ccxt` ni `dataclasses-json`; no se puede certificar el arranque de `main.py`. |
| Instalación frontend (`npm ci`) | **No completada** | La ejecución superó el límite de tiempo disponible. |
| Instalación frontend offline | **Fallida por caché incompleta** | `npm ci --offline` devuelve `ENOTCACHED` para `zustand@5.0.16`. |
| TypeScript typecheck | **No ejecutado** | No hay instalación completa de dependencias frontend. |
| Tests frontend | **No ejecutados** | No hay instalación completa de dependencias frontend. |
| Build frontend | **No ejecutado** | No hay instalación completa de dependencias frontend. |
| Arranque/apagado backend | **No ejecutado** | Faltan dependencias base y se evita ejecutar el arranque sobre el estado persistente aportado. |
| Arranque frontend | **No ejecutado** | No hay instalación completa de dependencias frontend. |

## 3. Dependencias y manifiestos

### 3.1 Python

Se han encontrado tres manifiestos:

- `requirements.txt` en la raíz.
- `l1_operational/requirements.txt`.
- `l2_tactic/requirements.txt`.

Los manifiestos L1 y L2 repiten dependencias ya presentes en la raíz. El manifiesto raíz utiliza principalmente rangos mínimos (`>=`) y no fija un conjunto de versiones transitivas. No existe un lock Python (`requirements-lock.txt`, `uv.lock`, `poetry.lock` o equivalente) ni una declaración inequívoca de la versión objetivo de Python.

No se han fijado versiones a ciegas: hacerlo sin resolver y probar la matriz de compatibilidad podría introducir incompatibilidades, especialmente con PyTorch, Stable-Baselines3, Transformers y FinRL.

### 3.2 ML, RL y sentimiento

Se han localizado imports directos de:

- PyTorch (`torch`).
- Stable-Baselines3.
- Transformers.
- AsyncPRAW.
- yfinance.

Hay usos en módulos de entrenamiento, inferencia y estrategia. No se puede declarar que todos sean opcionales para el runtime principal sin completar el grafo de imports y comprobar qué módulos se cargan durante el arranque. La instalación debe separar explícitamente extras de entrenamiento/inferencia cuando se demuestre que no son necesarios para el camino base.

### 3.3 Frontend

`frontend/package.json` declara `npm@10.9.2` y Node `>=20`. Existe `frontend/package-lock.json`, pero no se pudo validar mediante `npm ci`.

El `package-lock.json` de la raíz contiene únicamente `packages: {}`. No representa un lock de dependencias Python y no debe confundirse con el lock del frontend.

La configuración ESLint referencia reglas `@typescript-eslint/*`, pero `frontend/package.json` no declara `@typescript-eslint/parser` ni `@typescript-eslint/eslint-plugin`. El archivo de configuración tampoco registra esos plugins. La validación ESLint requerirá corregir la configuración y actualizar el lock del frontend en un entorno con acceso al registro npm.

## 4. Configuración y entorno

Existe `.env.example` en la raíz y otro en `frontend/`. Se han detectado variables utilizadas por el código que no están reflejadas en el ejemplo raíz, entre ellas:

- `HRM_INITIAL_BALANCE`.
- `HRM_ENABLE_COMMISSIONS`.
- `HRM_ENABLE_SLIPPAGE`.
- `HRM_PAPER_MODE`.
- `HRM_API_HOST`.
- `HRM_DATA_DIR` y `HRM_INFER_DIR`.
- `HRM_RUTA_RESET_PAPER`.
- `USE_TESTNET`.
- `ORDERFLOW_USE_POLLING` y `ORDERFLOW_POLL_INTERVAL`.
- `TWITTER_BEARER_TOKEN`, `GLASSNODE_API_KEY` y `SANTIMENT_API_KEY`.

Hay nombres de variables solapados o distintos para conceptos cercanos, por ejemplo `INITIAL_CAPITAL` frente a `HRM_INITIAL_BALANCE`, y `PAPER_MODE` frente a `HRM_PAPER_MODE`. No se han unificado automáticamente porque no se ha demostrado que sean alias equivalentes en todos los componentes; cambiar su precedencia podría alterar el comportamiento de sesión.

No se han encontrado rutas absolutas de Windows en la búsqueda estática básica de literales `C:\\`/`D:\\` en archivos Python. Esta búsqueda no descarta rutas construidas dinámicamente ni referencias en otros formatos.

## 5. Hallazgos de calidad y pruebas

### 5.1 Scripts que interfieren con pytest

- `test_l1_diagnostic.py` ejecuta un diagnóstico durante la importación y llama a `sys.exit(1)` si falta un CSV de datos.
- `tests/quick_log_test.py` ejecuta una prueba SQLite durante la importación y presupone que existe una tabla `logs`.

Estos scripts deben convertirse en funciones invocables bajo `if __name__ == "__main__":` o separarse del descubrimiento automático de pytest. No se han reescrito en esta pasada para evitar cambiar scripts heredados sin revisar sus usos.

### 5.2 TypeScript

`frontend/tsconfig.json` excluye los tests del `typecheck` (`src/**/__tests__/**`, `src/**/*.test.ts`, `src/**/*.test.tsx`, `src/test/**`). Esto puede ser una decisión intencional si Vitest transpila y ejecuta los tests por separado, pero no demuestra que los tests estén tipados. Debe documentarse o añadirse una configuración de TypeScript específica para tests.

### 5.3 Datos y persistencia

El ZIP contiene `.hrm_runtime/*.pid`, archivos de estado persistente y un ZIP anidado. Se han conservado en la copia de trabajo. No se ha ejecutado `system_cleanup`, `preparar_sesion` sobre el directorio real del repositorio ni ninguna rutina que pueda borrar o migrar datos persistentes. Los tests ejecutados utilizan `tmp_path` y datos aislados.

## 6. Riesgos que impiden certificar el arranque

El entorno no dispone de `ccxt` ni `dataclasses-json`, declaradas en los manifiestos base. Además, `core/config.py` contiene un `EnvironmentConfig` cuyo argumento predeterminado es `mode="live"`, aunque otras constantes fuerzan PAPER. No se ha demostrado que ese constructor sea utilizado en el arranque principal actual, por lo que se registra como riesgo de configuración y no se cambia en esta fase parcial.

`main.py` conserva una rama condicional asociada a `BINANCE_MODE=live`, `PAPER_MODE=false` y `HRM_ENABLE_LIVE_TRADING=true`. No se ha activado ni ejecutado. Las pruebas y validaciones futuras deberán imponer PAPER en el entorno aislado y comprobarlo antes de iniciar servicios.

## 7. Cambios realizados

- Se ha añadido este informe `INFORME_FASE_1.md` a la copia de trabajo.
- No se han modificado archivos de lógica de aplicación, configuración de trading, estrategias, señales, indicadores, riesgo ni persistencia.
- Los `__pycache__` generados por `compileall` son artefactos de validación local y se excluyen del paquete de entrega.

## 8. Acciones necesarias para completar Fase 1

1. Acordar y fijar la versión de Python objetivo y crear un lock reproducible con versiones resueltas en un entorno compatible.
2. Separar dependencias base de extras ML/RL/sentimiento tras comprobar el grafo de imports del runtime.
3. Corregir los dos scripts que ejecutan efectos durante la colección de pytest y ejecutar la suite completa.
4. Instalar frontend con `npm ci` en un entorno con acceso al registro npm.
5. Registrar correctamente los plugins ESLint TypeScript y actualizar `frontend/package-lock.json`.
6. Ejecutar typecheck, tests, build y lint del frontend; resolver errores sin eliminar funcionalidades ni contratos.
7. Completar el inventario de variables de entorno y documentar precedencia sin cambiarla accidentalmente.
8. Validar arranque y apagado backend/frontend en un directorio temporal o copia aislada, con PAPER forzado y sin acceso a rutas persistentes del usuario.
9. Repetir tests de regresión PAPER y comprobar que no se han modificado estrategias, señales, indicadores, dimensionamiento ni riesgo.
10. Actualizar este informe con resultados verificables y cerrar la Fase 1 solo cuando se cumplan los criterios de aceptación.

## 9. Conclusión

La Fase 1 **no está cerrada**. La sintaxis Python y los 32 tests focalizados de perfiles PAPER pasan en este entorno, pero la suite general, la instalación y compilación frontend, el entorno Python reproducible y el arranque/apagado todavía no están validados. No debe presentarse HRM como instalación reproducible ni como sistema funcionalmente certificado hasta completar los bloqueos anteriores.
