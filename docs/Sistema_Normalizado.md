# Sistema de Trading Normalizado

## Objetivos Alcanzados

### 1. Paper Trading con Datos Reales
- ✅ **Paper Mode Siempre Activo**: Mantiene `paper_mode = True` en todo momento
- ✅ **Datos de Mercado Reales**: Usa endpoints públicos reales de Binance (no testnet ni sandbox) para precios, velas, volumen y order book
- ✅ **Portfolio Simulado**: El portfolio es gestionado internamente y reacciona a precios reales
- ✅ **Sin Sincronización Real**: Nunca intenta sincronizar balances reales de Binance cuando está en modo paper

### 2. Separación Clara de Responsabilidades
- **MarketDataManager**: Obtiene y valida datos reales de Binance
- **PortfolioManager**: Gestiona el estado del portfolio simulado
- **OrderManager**: Genera órdenes simuladas (paper fills) con override para garantizar ejecución
- **OrderIntentBuilder**: Construye intenciones de orden desde señales con validación exhaustiva

### 3. Normalización del Flujo de Señales
- ✅ **Señales Válidas → Al Menos 1 Order Intent**: Se garantiza que cualquier señal válida (L3 = BUY, L2 = BUY/BUY_LIGHT, confidence ≥ threshold) genere al menos una intención de orden
- ✅ **Override de Paper Mode**: Si el cálculo normal devuelve tamaño 0, se usa un override de 10% del portfolio para BUY en paper mode
- ✅ **Umbrales Relaxados en Paper Mode**: Confidence ≥ 0.4 (o ≥ 0.3 en modo agresivo temporal)

### 4. Modo Agresivo Temporal
- ✅ **Flag Explícito**: `TEMPORARY_AGGRESSIVE_MODE`
- ✅ **Funcionalidad**: Reduce filtros conservadores y permite mayor frecuencia de operaciones sin alterar la gestión de riesgo base
- ✅ **Duración Limitada**: 
  - Por tiempo (defecto: 300 segundos = 5 minutos)
  - Por ciclos (defecto: 100 ciclos)
- ✅ **Desactivación Automática**: El modo se apaga automáticamente cuando se alcanza el límite de tiempo o ciclos
- ✅ **Logging Claro**: Muestra cuando el modo se activa, expira o se desactiva manualmente

### 5. Logging Mejorado
- ✅ **Logs Explícitos**:
  - Uso de datos reales de Binance
  - Generación de órdenes paper
  - Blocking de señales con razón concreta (ej: cooldown activo, confidence insuficiente)
- ✅ **Elimina Ambiguidades**: No hay más "signal valid but no intent generated" sin explicación

## Cambios Implementados

### 1. `core/config.py`
- **Modo Agresivo Temporal**: Implementado con duración y límite de ciclos
- **Umbrales de Confianza**: Min signal confidence = 0.35, L2 confidence = 0.40 (relaxados para paper mode)
- **Configuración de Paper Mode**: Habilitado por defecto

### 2. `l1_operational/order_intent_builder.py`
- **Logging Mejorado**: Más detalles sobre por qué una señal es rechazada
- **Modo Agresivo Temporal**: Umbral de confidence reducido a 0.3 en modo agresivo
- **Override de Paper Mode**: Garantiza que se genere una intención de orden incluso si el cálculo normal da 0

### 3. `core/portfolio_manager.py`
- **Sincronización Evitada**: Asegura que en paper mode nunca intente sincronizar con balances reales de Binance
- **Actualización Local**: En paper mode, siempre usa el cálculo local de balances para evitar errores

### 4. `system/trading_pipeline_manager.py`
- **Check de Modo Agresivo**: Verifica si el modo agresivo temporal debe ser desactivado al final de cada ciclo

## Resultado Esperado

En paper mode:
- ✅ Ver operaciones simuladas con paper trades
- ✅ Ver cambios en el portfolio basados en precios reales
- ✅ Ver PnL calculado con datos reales
- ✅ Sin llamadas fallidas a Binance por balances reales
- ✅ Sin ejecución real bajo ningún concepto

## Ejecución y Pruebas

### Instalar Dependencias
```bash
pip install pytest
```

### Correr Pruebas
```bash
python -m pytest test_paper_mode_order_intents.py test_temporary_aggressive_mode.py -v
```

Todas las pruebas deben pasar, incluyendo:
- **test_paper_mode_order_intents_from_valid_signals**: Verifica que las señales válidas generen intenciones de orden
- **test_paper_mode_override_for_zero_size**: Verifica el override de paper mode para tamaños de orden 0
- **test_temporary_aggressive_mode_basic**: Prueba la activación y desactivación básica
- **test_temporary_aggressive_mode_time_limit**: Prueba el límite de tiempo
- **test_temporary_aggressive_mode_cycle_limit**: Prueba el límite de ciclos
- **test_temporary_aggressive_mode_manual_disable**: Prueba la desactivación manual