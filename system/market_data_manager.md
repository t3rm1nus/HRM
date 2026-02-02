# Market Data Manager - Documentación

## Visión General

El `MarketDataManager` es un módulo centralizado que gestiona la obtención, validación y caché de datos de mercado para el sistema HRM. Implementa una estrategia robusta de fuentes primarias y fallbacks para garantizar la disponibilidad continua de datos.

## Arquitectura

### Componentes Principales

1. **Fuentes de Datos**
   - **ExternalAdapter**: Fuente primaria (preferida)
   - **RealTimeLoader**: Fallback 1 (datos en tiempo real)
   - **DataFeed**: Fallback 2 (último recurso)

2. **Validación**
   - **UnifiedValidator**: Validación centralizada
   - Reparación automática de datos inválidos
   - Validación por símbolo

3. **Caché**
   - Almacenamiento temporal de datos válidos
   - Control de expiración configurable
   - Acceso concurrente seguro

4. **Logging**
   - Registro detallado de decisiones de fallback
   - Métricas de rendimiento
   - Errores y recuperaciones

## Configuración

### Parámetros de Configuración

```python
config = {
    "SYMBOLS": ["BTCUSDT", "ETHUSDT"],           # Símbolos a monitorear
    "VALIDATION_RETRIES": 3,                     # Reintentos de validación
    "CACHE_VALID_SECONDS": 30,                   # Duración del caché
    "FALLBACK_STRATEGY": "external->realtime->datafeed"  # Estrategia de fallback
}
```

### Estrategias de Fallback

1. **EXTERNAL_TO_REALTIME_TO_DATAFEED** (por defecto)
   - ExternalAdapter → RealTimeLoader → DataFeed

2. **REALTIME_TO_DATAFEED**
   - RealTimeLoader → DataFeed

3. **DATAFEED_ONLY**
   - Solo DataFeed

## Uso

### Inicialización

```python
from system.market_data_manager import MarketDataManager

# Con configuración por defecto
manager = MarketDataManager()

# Con configuración personalizada
config = {
    "SYMBOLS": ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
    "CACHE_VALID_SECONDS": 60,
    "FALLBACK_STRATEGY": "realtime->datafeed"
}
manager = MarketDataManager(config)
```

### Obtención de Datos

```python
# Obtener datos con lógica de fallback
data = await manager.get_market_data()

# Forzar actualización (ignorar caché)
data = await manager.refresh_data()

# Validar y reparar datos externos
validated_data = await manager.validate_and_repair(raw_data)
```

### Función de Conveniencia

Para compatibilidad con el código existente:

```python
from system.market_data_manager import get_market_data_with_fallback

data = await get_market_data_with_fallback()
```

## Flujo de Operación

### 1. Búsqueda en Caché
- Verifica si hay datos válidos en caché
- Si el caché es válido, retorna datos inmediatamente
- Incrementa contador de cache hits

### 2. Intento de Fuentes
- Intenta obtener datos según la estrategia configurada
- Registra qué fuente se intentó y el resultado
- Cuenta los fallbacks utilizados

### 3. Validación y Reparación
- Valida la estructura general de los datos
- Valida cada símbolo individualmente
- Intenta reparar datos inválidos automáticamente
- Registra estadísticas de validación

### 4. Almacenamiento en Caché
- Almacena datos válidos en caché
- Registra la fuente de origen
- Actualiza estadísticas

## Validación de Datos

### Validación Estructural
- Verifica que los datos sean un diccionario
- Comprueba que no esté vacío
- Valida la presencia de símbolos requeridos

### Validación por Símbolo
- Convierte datos a DataFrame cuando sea necesario
- Valida columnas OHLCV
- Limpia valores no numéricos o negativos
- Repara formatos inconsistentes

### Reparación Automática
- Convierte dict a DataFrame
- Convierte listas a DataFrame
- Crea DataFrames vacíos con columnas estándar
- Maneja errores de conversión

## Caché

### Política de Expiración
- Los datos en caché tienen una duración configurable
- Por defecto: 30 segundos
- Se puede ajustar según necesidades de latencia

### Seguridad Concurrente
- Uso de asyncio.Lock para acceso concurrente
- Operaciones atómicas de lectura/escritura
- Prevención de condiciones de carrera

### Limpieza Automática
- Caché expirado se elimina automáticamente
- Limpieza forzada con `refresh_data()`
- Limpieza en cierre del gestor

## Logging y Métricas

### Información Registrada
- Qué fuente se intentó y resultado
- Razón del fallo (si falla)
- Qué fallback se utilizó
- Estado de la validación
- Resultado de la reparación

### Estadísticas de Operación
- Intentos totales
- Éxitos de validación
- Fallbacks utilizados
- Cache hits
- Fallos de validación
- Datos reparados

### Ejemplo de Logging

```
📡 Intentando obtener datos de ExternalAdapter (fuente primaria)
⚠️ ExternalAdapter retornó datos vacíos
📡 Intentando obtener datos de RealTimeLoader (fallback 1)
✅ RealTimeLoader exitoso: ['BTCUSDT', 'ETHUSDT']
💾 Caché actualizado desde realtime
```

## Manejo de Errores

### Errores de Conexión
- Excepciones de red se capturan y registran
- No interrumpen el flujo principal
- Se intentan fallbacks automáticamente

### Datos Inválidos
- Validación robusta con múltiples capas
- Reparación automática cuando sea posible
- Retorno de dict vacío si no se puede reparar

### Errores de Sistema
- Excepciones no controladas se registran con traceback
- El sistema continúa operando
- Se mantiene la disponibilidad del servicio

## Pruebas

### Cobertura de Pruebas
- Inicialización y configuración
- Estrategias de fallback
- Validación y reparación de datos
- Caché y expiración
- Manejo de errores
- Estadísticas y métricas

### Ejecución de Pruebas

```bash
# Ejecutar todas las pruebas
pytest test_market_data_manager.py -v

# Ejecutar pruebas específicas
pytest test_market_data_manager.py::TestMarketDataManager::test_get_market_data_with_fallback -v
```

## Integración con el Sistema

### Uso en main.py

El módulo está diseñado para reemplazar la lógica actual de obtención de datos en `main.py` (líneas 224-279):

```python
# Antes (main.py líneas 224-279)
logger.info("🔄 Attempting to get realtime market data...")
if external_adapter and external_adapter.get_component('realtime_loader'):
    market_data = await external_adapter.get_component('realtime_loader').get_market_data()
    # ... validación manual ...

# Después (con MarketDataManager)
from system.market_data_manager import MarketDataManager
manager = MarketDataManager()
market_data = await manager.get_market_data()
```

### Beneficios de la Integración
- **Código más limpio**: Elimina lógica duplicada
- **Mayor confiabilidad**: Fallbacks automáticos
- **Mejor mantenimiento**: Validación centralizada
- **Mejor observabilidad**: Logging detallado
- **Mayor testabilidad**: Componentes aislados

## Mejoras Futuras

### Posibles Extensiones
1. **Caché persistente**: Almacenamiento en disco para reinicios
2. **Balanceo de carga**: Distribuir solicitudes entre múltiples fuentes
3. **Circuit breaker**: Evitar fuentes que fallan repetidamente
4. **Métricas avanzadas**: Prometheus/Grafana integration
5. **Configuración dinámica**: Cambios en tiempo real de estrategias

### Optimizaciones
1. **Validación paralela**: Validar múltiples símbolos concurrentemente
2. **Compresión de caché**: Reducir uso de memoria
3. **Prefetching**: Cargar datos antes de que expiren
4. **Adaptación automática**: Ajustar estrategias según éxito/fallo histórico