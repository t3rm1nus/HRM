# HRM — Informe de validación de Fase 1 (actualizado 16-nov-2026)

**Resultado: FASE 1 CASI CERRADA.** Frontend y backend cumplen casi todos los criterios medidos; quedan los puntos de la sección 5.
No se tocaron estrategias, señales, indicadores, riesgo ni dimensionamiento. PAPER sigue forzado; todas las ejecuciones se hicieron en copias aisladas con `PAPER_MODE=true`, `BINANCE_MODE=paper` y `HRM_ENABLE_LIVE_TRADING=false`.

## 1. Validaciones (medidas, Python 3.12.3 / Node 22 / npm 10.9)

| Comprobación | Resultado |
|---|---|
| `compileall` Python | Correcto |
| `pytest tests` | **40 passed** (antes 32 passed, 1 failed) |
| `import main` | Correcto con `requirements.txt` corregido |
| API FastAPI (`uvicorn main:app`) | Arranca, `/health`, `/status`, `/version`, `/sesion/modo`, `/mercado/estado` responden 200; apagado limpio con SIGTERM |
| `python main.py` (bot completo) | Llega hasta la carga del modelo L2; ahí aborta porque `models/` no está en el zip (ver §4) |
| `npm ci` | Correcto, 397 paquetes |
| `npm run typecheck` | Correcto, 0 errores |
| `npm test` | **171 passed** en 29 archivos (antes 168 passed, 5 failed) |
| `npm run build` | Correcto |

## 2. Cambios realizados

**Bug real corregido (bloqueaba el arranque):** `system/bootstrap.py::bootstrap_simulated_exchange` exigía `isinstance(config, dict)`, pero `main.py` le pasa `comms.config.config` (`HRMAppConfig`, que se comporta como dict sin serlo). Resultado: `python main.py` terminaba con `TypeError` en PAPER. Ahora se valida con la función nueva `admite_consulta_por_clave`, que solo exige `.get`, que es lo único que usa el código. Tests nuevos en `tests/unit/test_bootstrap_configuracion.py`.

**Dependencias:** `requirements.txt` ahora incluye lo que el arranque necesita de verdad: `psutil`, `python-binance`, `joblib`, `torch`, `transformers`, `asyncpraw` (se comprobó bloqueando cada paquete). Nuevos: `requirements-opcional.txt` (stable-baselines3, gymnasium, yfinance: sin ellos solo hay avisos), `requirements-desarrollo.txt` y `requirements-lock.txt` (registro del entorno validado). `scikit-learn` y `scipy` no son necesarios para arrancar.

**Logs:** en `l1_operational/simulated_exchange_client.py` se usaba formato `%s` con loguru, que no lo interpola: la línea `PAPER ORDER` salía sin lado, símbolo, cantidad ni precio. Corregido en 4 llamadas (solo texto de log).

**Tests:**
- Backend: `test_rechaza_simbolos_que_no_terminan_en_usdt` esperaba un mensaje antiguo; ahora coincide con el real (`Par no soportado`).
- Frontend: eliminado `resetDatosPaper.test.ts` (probaba `solicitarResetDatosPaper`, función que ya no existe; la sustituye `reiniciarDatosPaper`, que sí está probada). Añadido el caso de rechazo 403 a `cambiarModoSesion.test.ts`. En `fetchModoSesion` y `cambiarModoSesion`, el error HTTP 400 del backend ahora se comprueba por mensaje y estado, no por `RESPUESTA_INVALIDA` (ese código es para respuestas que violan el contrato).
- `pytest.ini`: fijado `asyncio_default_fixture_loop_scope`.

## 3. Verificado respecto a informes anteriores
- La imposibilidad de instalar el frontend era del entorno de revisión (sin acceso a npm), no del proyecto.
- Los tests `tests/api/*` citados en el informe del 5-nov **no existen** en este zip.

## 4. Límites de esta verificación
- No hay acceso a Binance desde el entorno de pruebas: el polling de OrderFlow falla por red, como era de esperar.
- `models/` está en `.gitignore` y no viene en el zip. `l2_tactic/finrl_processor.py` lanza `RuntimeError` si falta `models/L2/claude.zip`, sin degradación controlada. En tu máquina, con los modelos, habría que repetir el arranque completo.
- No se creó un venv limpio (el disco del entorno estaba al 95 %); la resolución se comprobó con `pip install --dry-run`.

## 5. Pendiente para cerrar la Fase 1
1. Arranque completo del bot en tu máquina (con `models/`), con PAPER forzado y apagado limpio.
2. **57 llamadas a `logger` en 10 archivos usan formato `%` y pierden sus datos** (`bootstrap.py` 15, `system/trading_pipeline_manager.py` 11, `runtime_loop.py` 8, `l1_operational/order_executors.py` 6, `l1_operational/risk_guard.py` 5, `execution_gateway.py` 4, `l1_operational/binance_client.py` 4, `main.py` 2, `system/system_cleanup.py` 1, `core/portfolio_manager.py` 1). Afecta a la trazabilidad de órdenes y riesgo (Fase 5).
3. Tests dispersos fuera de `testpaths` (`l2_tactic/tests/`, `core/test_portfolio_rebalancer.py`, `hacienda/`, etc.) y scripts de diagnóstico en la raíz.
4. Variables de entorno de la lista del informe anterior sin reflejar en `.env.example`.
5. Registrar los plugins `@typescript-eslint` en la configuración ESLint del frontend.
6. Manifiestos duplicados `l1_operational/requirements.txt` y `l2_tactic/requirements.txt`.
7. `bootstrap_simulated_exchange` conserva un capital por defecto de 500 USDT (`INITIAL_BALANCE`) si no hay configuración.
