# Suite nueva de HRM

Esta es una primera base ejecutable, construida contra interfaces observadas en el repositorio. No certifica HRM completo ni sustituye la auditoría funcional.

## Instalación

```powershell
python -m pip install pytest
```

## Ejecución

Después de congelar la suite antigua y copiar `pytest.ini` y `tests/` a la raíz:

```powershell
python -m pytest --collect-only -q
python -m pytest -q
```

La configuración limita la recopilación a `tests/`; no importará los tests antiguos ni scripts de `scripts/`.

## Seguridad y aislamiento

- No inicia HRM ni conecta con Binance.
- No utiliza credenciales.
- No escribe en `data/`, `persistent_state/` ni `logs/`.
- Los tests de almacenamiento usan exclusivamente `tmp_path`.
- El singleton del exchange se reinicia antes y después de cada test.
- Un defecto conocido de migración está identificado como `xfail`.

## Cobertura inicial

- Exchange PAPER operacional: validaciones, balances, comisiones, slippage, historial y reset.
- Perfiles PAPER: rutas, aislamiento de limpieza y migración temporal.
- Modo de sesión: decisión pura de modo.

Queda pendiente ampliar la suite tras inspeccionar las interfaces reales de PortfolioManager, OrderManager, validadores, ExecutionGateway, fuentes de mercado, StateCoordinator, API REST, WebSockets, backtesting y frontend.
