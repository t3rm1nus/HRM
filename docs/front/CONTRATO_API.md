# Contrato de API FastAPI - HRM Trading System

Documenta los endpoints expuestos por el sistema de trading HRM a través de FastAPI. Estos endpoints están integrados en `main.py` y disponibles cuando se ejecuta el sistema.

## Base URL

```
http://localhost:8000
```

## Documentación Interactiva

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

## Endpoints

### 1. Chequeo de Salud del Sistema

**Endpoint**: `GET /health`

Propósito: Verifica que el sistema de trading esté operando correctamente.

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `status` | `string` | Estado salud (`ok`/`error`) |
| `timestamp` | `string` | Marca de tiempo ISO 8601 UTC |
| `version` | `string` | Versión de la API (`2.0.0`) |

**Respuesta exitosa** (200):
```json
{
  "status": "ok",
  "timestamp": "2024-01-15T14:30:00Z",
  "version": "2.0.0"
}
```

**Errores**: Ningún error específico, siempre retorna 200 con status "ok" o "error".

---

### 2. Estado Actual del Trading

**Endpoint**: `GET /status`

Propósito: Obtiene el estado operativo del bot de trading.

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `status` | `string` | Estado del sistema (`running`/`paused`) |
| `trading_active` | `bool` | ¿El trading está activo? |
| `cycle_count` | `int` | Número de ciclos completados |
| `uptime` | `string` | Tiempo de actividad |

**Respuesta exitosa** (200):
```json
{
  "status": "running",
  "trading_active": true,
  "cycle_count": 42,
  "uptime": "sistema activo"
}
```

---

### 3. Información de la Cartera

**Endpoint**: `GET /portfolio`

Propósito: Obtiene el estado actual de balances y NAV de la cartera de trading.

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `btc_balance` | `float` | Balance de BTC |
| `eth_balance` | `float` | Balance de ETH |
| `usdt_balance` | `float` | Balance de USDT |
| `total_nav` | `float` | Valor total del NAV |
| `initial_capital` | `float` | Capital inicial configurado |

**Respuesta exitosa** (200):
```json
{
  "btc_balance": 1.5234,
  "eth_balance": 12.45,
  "usdt_balance": 500.0,
  "total_nav": 1250.30,
  "initial_capital": 500.0
}
```

---

### 4. Información de Versión

**Endpoint**: `GET /version`

Propósito: Obtiene la versión actual del sistema HRM Trading.

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `app_version` | `string` | Versión de la aplicación |
| `framework` | `string` | Framework utilizado |
| `python_version` | `string` | Versión de Python requerida |

**Respuesta exitosa** (200):
```json
{
  "app_version": "2.0.0",
  "framework": "FastAPI",
  "python_version": "3.10+"
}
```

---

## Convenios y Consideraciones

### Type Hints Estrictos

Todos los endpoints utilizan type hints verificables siguiendo las reglas `.clinerules`:

- Respuestas definidas con `pydantic.BaseModel`
- Tipos explícitos en firma de funciones (`-> HealthCheckResponse`, `-> StatusResponse`, etc.)
- Sin uso de `any` tipo
- Validación automática de datos mediante Pydantic

### Integración con Trading Loop

- Los endpoints se ejecutan concurrentemente con el trading loop principal
- FastAPI se inicia en segundo plano al ejecutar `python main.py`
- El sistema permanece responsive mientras el trading loop corre
- Endpoints son read-only para datos del sistema (no modifican estado de trading)

### Seguridad

- Sin autenticación requerida para endpoints públicos
- Sin exposición de credenciales API en respuestas
- Logs sanitizados (sin keys secrets en output)
- Documentación automática en `/docs` y `/redoc`

### Desarrollo

Para iniciar con la API integrada:

```bash
cd c:\\proyectos\\HRM
python main.py
```

La API estará disponible en `http://0.0.0.0:8000` por defecto.

### Próximas Mejoras

- Agregar autenticación OAuth2 / API keys para endpoints sensibles
- Expandir endpoints de `/portfolio` con más métricas (P&L, riesgo, etc.)
- Agregar webhook endpoints para notificaciones externas
- Versionado de API en rutas futuras (`/v1/health`, `/v1/status`, etc.)