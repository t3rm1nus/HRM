# Divergencias de Configuración Identificadas

## Tabla de Divergencias

| Parámetro | Archivo | Valor | Quién lo Lee |
|-----------|---------|-------|--------------|
| `capital_usdt` / `initial_balance` | `config_loader.py` (línea 17) | `500.0` | Módulo de carga de configuración |
| `capital_usdt` / `initial_balance` | `core/configuration_manager.py` (línea 27) | `500.0` | `TradingConfig` dataclass |
| `capital_usdt` / `initial_balance` | `core/config.py` (línea 49) | `500.0` | `TradingConfig` dataclass (fuerte typing) |
| `capital_usdt` / `initial_balance` | `scripts/start.ps1` (línea 16) | `3000.0` | Script de inicio - capital real |
| `capital_usdt` / `initial_balance` | `l1_operational/config.py` (línea 12-13) | "OPTIMIZADO PARA 3K USDT" | Límites de riesgo L1 |
| `mode` / `PAPER_MODE` | `config_loader.py` (línea 20) | `"paper"` | Carga de estado inicial |
| `mode` / `PAPER_MODE` | `core/config.py` (línea 53) | `True` | `TradingConfig.PAPER_MODE` |
| `mode` / `PAPER_MODE` | `core/configuration_manager.py` (línea 74) | `"simulated"` | `EnvironmentConfig.mode` |
| `mode` / `PAPER_MODE` | `l1_operational/config.py` (línea 10) | `"LIVE"` | Modo operación L1 |
| `mode` | `scripts/start.ps1` (línea 19) | `"paper"` | Script de inicialización |
| `mode` | `persistent_state/portfolio_state_live.json` (línea 20) | `"simulated"` | Estado de cartera persistente |
| `cycle_duration_seconds` | `main.py` (línea 804-807) | `3.0` (forzado) | Bucle de trading principal |
| `cycle_duration_seconds` | `core/configuration_manager.py` (línea 48) | `10` | `TradingConfig.cycle_duration_seconds` |
| `hrm_path_mode` | `core/configuration_manager.py` (línea 45) | `"PATH2"` | `TradingConfig.hrm_path_mode` |
| Rutas absolutas Windows | `l2_tactic/config.py` (línea 26) | `"C:/proyectos/HRM/models/L2/claude.zip"` | Configuración de modelo AI |
| Rutas absolutas Windows | `l2_tactic/config.py` (líneas 37-42) | `C:/proyectos/HRM/models/L2/*.zip` | Modelos disponibles |
| `MAX_PORTFOLIO_EXPOSURE_BTC` | `l1_operational/config.py` (línea 22) | `0.40` | `PORTFOLIO_LIMITS` - "OPTIMIZADO PARA 3K USDT" |
| `MAX_PORTFOLIO_EXPOSURE_ETH` | `l1_operational/config.py` (línea 23) | `0.40` | `PORTFOLIO_LIMITS` - "OPTIMIZADO PARA 3K USDT" |
| `MAX_PORTFOLIO_EXPOSURE_BTC` | `core/configuration_manager.py` (línea 28) | `0.40` | `TradingConfig.max_portfolio_exposure_btc` |
| `MAX_PORTFOLIO_EXPOSURE_ETH` | `core/configuration_manager.py` (línea 29) | `0.40` | `TradingConfig.max_portfolio_exposure_eth` |

## Divergencias Críticas Identificadas

### 1. Capital Inicial Discrepancy
- **3000 USDT**: `scripts/start.ps1` y `l1_operational/config.py` (optimizado para 3K)
- **500 USDT**: `config_loader.py`, `core/configuration_manager.py`, `core/config.py` (defaults)
- **Ausencia de 10000 USDT**: No encontrado en ningún archivo de configuración

### 2. Modo de Operación Discrepancy
- **paper**: `config_loader.py`, `scripts/start.ps1`
- **simulated**: `core/configuration_manager.py`, `persistent_state/portfolio_state_live.json`
- **live/LIVE**: `l1_operational/config.py`
- **Conflito**: `core/config.py` usa `PAPER_MODE = True` (forza paper)

### 3. Duración de Ciclo Discrepancy
- **3 segundos**: `main.py` línea 804-807 - "Enforce 3-second cycle timing"
- **10 segundos**: `core/configuration_manager.py` línea 48 - `cycle_duration_seconds: int = 10`
- **Impacto**: El main.py sobrescribe el valor de TradingConfig con hardcode de 3s

### 4. Rutas Absolutas de Windows
- Varios archivos en `l2_tactic/config.py` usan rutas `C:/proyectos/HRM/...`
- Esto rompe la portabilidad entre sistemas operativos
- Debe reemplazarse con rutas relativas o variables de entorno

### 5. Límites de Exposición
- `l1_operational/config.py` tiene `PORTFOLIO_LIMITS` optimizados para 3K USDT
- `core/configuration_manager.py` `TradingConfig` tiene exposiciones similares (0.40 BTC, 0.40 ETH)
- Posible redundancia o configuración duplicada

## Propuesta: Única Fuente de Verdad

Se propone centralizar toda la configuración en `core/configuration_manager.py::TradingConfig` como fuente única de verdad, con los siguientes ajustes:

1. **Capital**: Migrar `scripts/start.ps1` para que use 500 USDT (default) o parámetro configurable
2. **Modo**: Establecer `mode = "paper"` como default en `EnvironmentConfig` y eliminar `PAPER_MODE = True` duplicado de `core/config.py`
3. **Ciclo**: Remover el hardcode de 3s en `main.py` y usar `config.cycle_duration_seconds` desde TradingConfig
4. **Rutas**: Reemplazar rutas absolutas `C:/proyectos/HRM/` por rutas relativas o de entorno
5. **Exposición**: Consolidar `PORTFOLIO_LIMITS` de L1 con `TradingConfig.max_portfolio_exposure_*`

**Acción recomendada**: Migrar toda la configuración para usar `HRMConfigurationManager` desde `core.configuration_manager` y depurar los archivos `config/*.py` individuales para que solo lean, no redefinan, los valores centrales.