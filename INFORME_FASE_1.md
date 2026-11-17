# HRM — Informe de validación de Fase 1 (cierre, 17-nov-2026)

**Resultado: FASE 1 CERRADA, con una verificación pendiente que solo puedes hacer tú** (arranque completo con `models/`, ver §4).
No se tocaron estrategias, señales, indicadores, riesgo ni dimensionamiento. PAPER sigue forzado. Todas las ejecuciones se hicieron en copias aisladas con `PAPER_MODE=true`, `BINANCE_MODE=paper`, `HRM_ENABLE_LIVE_TRADING=false`.

## 1. Validaciones medidas (Python 3.12.3, Node 22, npm 10.9)

| Comprobación | Resultado |
|---|---|
| `compileall` | Correcto |
| `pytest` (testpaths ampliado) | **53 passed** (antes 32 passed, 1 failed) |
| `import main` | Correcto con `requirements.txt` corregido |
| API (`uvicorn main:app`) | Arranca, endpoints 200, apagado limpio con SIGTERM |
| `python main.py` | Llega hasta cargar `models/L2/claude.zip` (ausente en el zip) |
| `npm ci` / typecheck / build | Correctos |
| `npm test` | **171 passed** en 29 archivos (antes 168 passed, 5 failed) |
| `npm run lint` | Ahora **funciona** (antes no podía ejecutarse): 387 problemas, ver §3 |

## 2. Cambios

**Bug que impedía arrancar:** `system/bootstrap.py::bootstrap_simulated_exchange` exigía `dict`, pero `main.py` entrega `HRMAppConfig`. Ahora valida con `admite_consulta_por_clave` (solo exige `.get`). Tests en `tests/unit/test_bootstrap_configuracion.py`.

**Logs (loguru no interpola `%s` ni usa `exc_info`):** 96 llamadas en 25 archivos perdían sus datos.
- 58 con `%s`/`%d`/`%.2f` pasan a placeholders `{}` (`bootstrap.py`, `runtime_loop.py`, `execution_gateway.py`, `risk_guard.py`, `order_executors.py`, `trading_pipeline_manager.py`, `binance_client.py`, etc.). Se usa `{}` sin formato numérico a propósito: con loguru, `{:.2f}` y llaves literales **lanzan excepción** en la llamada de log, y estas están dentro del bucle de trading. Contrapartida: los decimales ya no se redondean.
- 38 con `exc_info=True` pasan a `logger.opt(exception=True).<nivel>(...)`, y 3 en `trading_pipeline_manager.py` a `logger.exception(...)`. Antes, los errores del ciclo de trading, NAV y auto-learning se registraban sin traceback.
- Test de regresión `tests/unit/test_logs_loguru.py` falla si reaparece cualquiera de los dos patrones.
- `registro` y `log` (`logging` estándar) se dejaron intactos: sí interpolan.

**Dependencias:** `requirements.txt` con lo necesario para arrancar (`psutil`, `python-binance`, `joblib`, `torch`, `transformers`, `asyncpraw`, verificado bloqueando cada paquete); `requirements-opcional.txt` (stable-baselines3, gymnasium, yfinance, scipy, scikit-learn); `requirements-desarrollo.txt`; `requirements-lock.txt` (registro del entorno validado). `l1_operational/requirements.txt` y `l2_tactic/requirements.txt` ahora apuntan al raíz.

**Tests:** corregido el mensaje en `test_simulated_exchange_client.py`; frontend: eliminado `resetDatosPaper.test.ts` (función inexistente), añadido el caso 403 de `reiniciarDatosPaper` y ajustados los dos tests de error 400. `pytest.ini` incluye 4 archivos dispersos que funcionan (`core/test_portfolio_rebalancer.py`, `hacienda/test_tax_tracker.py`, `l2_tactic/test_grok_models.py`, `l2_tactic/test_tight_range_handler.py`).

**`.env.example`:** añadidas las variables que el código lee (`USE_TESTNET`, `ORDERFLOW_*`, claves de sentimiento) y las avanzadas, **comentadas**. `HRM_PAPER_MODE=false` desactiva PAPER en `core/config.py`; se avisa en el propio archivo.

**ESLint (frontend):** registrados `typescript-eslint`, `react-hooks` y `react-refresh` (se importaban y no se usaban), corregida la opción `IIFEs`, desactivadas las reglas base duplicadas, eximidos los tests de tamaño/magic numbers y añadido `npm run lint`.

## 3. Deuda conocida (no bloquea Fase 1, no se tocó para no alterar comportamiento)

- **Lint, 387 problemas:** 309 `no-magic-numbers` (todos en código de producción: `dominio/rendimiento/funciones.ts` 69, `dominio/indicadores/calculos.ts` 53…), 28 `rules-of-hooks`, 8 `max-lines-per-function`, 4 `consistent-type-imports`, 3 `react-refresh`, 2 `no-explicit-any`, 1 `no-unused-vars`.
- **Decisión pendiente (28 errores de hooks):** tus hooks se llaman `usarEstado`, `usarEtiquetaModo`…; React exige el prefijo `use` para reconocerlos. No son bugs. Opciones: renombrar a `useX` (convención pública de React) o desactivar la regla (no recomendado).
- **Tests dispersos rotos (no entran en la suite):** `l1_operational/test_clean_l1.py` (firma antigua de `OrderManager`), `l2_tactic/tests/*` (el `conftest` importa `l2_tactic.ai_model_integration.AIModelIntegration`, que no es un paquete), `l3_strategy/test_regime_classifier.py` (necesita `scipy`, opcional), `test_orderflow_fix.py` (test `async` sin marcador). `test_l1_diagnostic.py` no contiene tests y ejecuta código al importarse.
- **Falta degradación controlada:** `l2_tactic/finrl_processor.py` aborta todo el arranque si falta el modelo L2.
- `bootstrap_simulated_exchange` mantiene 500 USDT por defecto si no hay configuración.
- `main()` tiene 1.111 líneas (límite de `.clinerules`: 20). Es la primera tarea de la Fase 2.

## 4. Límites de esta verificación
- Sin acceso a Binance desde el entorno de pruebas.
- `models/` está en `.gitignore` y no viene en el zip: **falta que ejecutes `python main.py` en tu máquina, con PAPER forzado, y confirmes que arranca y se apaga con Ctrl+C.**
- Sin venv limpio por falta de disco; resolución comprobada con `pip install --dry-run`.

## 5. Archivos a limpiar en la raíz (decisión tuya; no se ha borrado nada)
Informes y volcados de fases anteriores, ya superados: `fase1_pytest.txt`, `fase1_dependencias_instaladas.txt`. Scripts y vacíos: `create_schemas.py` y `write_servicio.py` (0 bytes), `fix.py`, `fix3.py`, `correcciones_l3.patch`, `package-lock.json` de la raíz (82 bytes). `logs/`, `paper_trades/` y `temp/` son datos de ejecución y no deben versionarse.
