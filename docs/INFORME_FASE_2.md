# INFORME FASE 2 — ESTADO GLOBAL Y COORDINACIÓN DEL RUNTIME

**Proyecto:** HRM  
**Base revisada:** rama `master` del repositorio `t3rm1nus/HRM` y entrega anterior de separación de API/helpers.  
**Alcance:** coordinación del estado operativo. No se ha avanzado a otra fase.

## 1. Resultado

Se establece `StateCoordinator` como fuente única de verdad del ciclo de vida del runtime. La API ya no deduce que el trading está activo por la existencia del proceso, por la apertura de un WebSocket ni por el inicio anticipado del bucle.

El estado `RUNNING` se publica cuando `main.py` entra realmente en un ciclo de trading, después de completar la espera de cadencia. La finalización de cada ciclo, los errores, la recuperación y el apagado quedan registrados en el mismo coordinador.

## 2. Estados explícitos

| Estado | Significado | `trading_active` |
|---|---|---:|
| `STOPPED` | Runtime detenido | `false` |
| `STARTING` | Inicialización en curso; todavía no hay ciclo activo | `false` |
| `RUNNING` | Un ciclo real está ejecutándose o acaba de completarse dentro de la ventana de actividad | `true` |
| `DEGRADED` | Ha fallado un ciclo y el runtime está en recuperación | `false` |
| `STOPPING` | Se está cerrando el runtime y sus componentes | `false` |
| `FAILED` | Fallo fatal de arranque, ejecución o apagado | `false` |

Las transiciones están validadas y protegidas con `threading.RLock`. Un arranque solo se puede adquirir desde `STOPPED` o `FAILED`. Los intentos concurrentes adicionales se rechazan.

## 3. Fuente de verdad e integración

### `system/state_coordinator.py`

- Singleton protegido por lock de clase; la inicialización también es segura frente a concurrencia.
- Almacén del ciclo de vida, contador de ciclos, actividad del ciclo, timestamps y último error.
- Historial limitado a 200 eventos, cada uno con fecha UTC, estado, motivo y componente.
- Conserva las interfaces históricas `get_state("current")`, `update_state(...)`, `update_market_data`, `update_total_value`, `update_portfolio_state` y `update_l3_output`.
- `update_state` acepta tanto la forma histórica `update_state(diccionario)` como la forma de componente `update_state(nombre, estado, datos)`.

### `main.py`

- Adquiere el runtime de forma atómica antes de preparar la sesión y registrar manejadores de señales.
- Una segunda invocación no arranca otro bucle y, muy importante, tampoco ejecuta el apagado global del runtime que ya estaba funcionando.
- Registra el comienzo real de cada ciclo, los errores recuperables y los fallos fatales.
- Publica `STOPPING` antes de cerrar recursos y `STOPPED` solo después del cleanup.
- Atiende también la cancelación de la tarea asíncrona.
- Mantiene LIVE bloqueado durante esta fase: se fuerzan las variables de entorno de PAPER antes de inicializar componentes.

### `api/servicios/servicio_estado_vivo.py` y `api/app.py`

- El servicio de API utiliza la misma instancia de `StateCoordinator`; no mantiene un segundo estado de ciclo de vida.
- `registrar_inicio_bucle()` se conserva como compatibilidad, pero no publica `RUNNING`.
- El endpoint `/status` expone el estado público y añade `runtime_state`, `reason`, `component`, `updated_at` y `last_error`.
- El WebSocket solo transporta eventos. Conectarse a él no cambia el estado operativo.

## 4. MessageBus: decisión de arquitectura

Se conserva `MessageBus` para distribución asíncrona de eventos entre componentes. No se utiliza como fuente de verdad del runtime porque es un bus efímero de publicación/suscripción: no proporciona por sí mismo estado actual autoritativo, historial consultable ni exclusión mutua de arranques. Añadir ahí un segundo estado habría duplicado responsabilidades.

La fuente de verdad queda en `StateCoordinator`; `MessageBus` y WebSocket siguen siendo mecanismos de comunicación, no gestores de estado.

## 5. Seguridad y alcance funcional

- No se han modificado estrategias, señales, parámetros de riesgo ni reglas de ejecución PAPER.
- No se ha cambiado el flujo de órdenes ni el Gateway.
- LIVE permanece bloqueado en `main.py` durante esta fase.
- No se ha añadido un endpoint HTTP de arranque/parada que pudiera permitir controlar el runtime sin autenticación.
- La parada sigue usando el mecanismo de cierre existente y ahora refleja correctamente las fases `STOPPING` y `STOPPED`.

## 6. Pruebas

Archivo nuevo: `tests/test_runtime_lifecycle.py`.

| Prueba | Resultado |
|---|---|
| Arranque y `STARTING` antes del primer ciclo | OK |
| `RUNNING` al comenzar un ciclo real | OK |
| Parada ordenada `STOPPING` → `STOPPED` | OK |
| Reinicio desde `STOPPED` | OK |
| Error durante el arranque → `FAILED` y reintento | OK |
| Error durante el ciclo → `DEGRADED` y recuperación | OK |
| Dos solicitudes de arranque concurrentes | OK; solo una adquiere el runtime |
| Parada durante una operación; bloqueo de ciclos nuevos | OK |
| Consulta del endpoint `/status` | OK |
| Compatibilidad de las interfaces legacy de `StateCoordinator` | OK |

**Ejecución realizada:** `python -m pytest -q tests/test_runtime_lifecycle.py` → **10 passed**.  
**Validación sintáctica:** `python -m py_compile` sobre los módulos entregados → **OK**.

La prueba del endpoint invoca el handler real `/status` y aísla los routers ajenos a ese endpoint. No se ha ejecutado un arranque integral del proyecto ni una sesión PAPER completa desde este entorno; esa comprobación debe hacerse en el entorno local de HRM, donde están disponibles todos los componentes y la configuración del proyecto.

## 7. Archivos de esta entrega

- `main.py` — integración del ciclo de vida y protección contra arranques concurrentes.
- `system/state_coordinator.py` — fuente única de verdad y transiciones.
- `api/servicios/servicio_estado_vivo.py` — fachada de lectura y notificación conectada al coordinador.
- `api/app.py` — contrato ampliado del endpoint `/status`.
- `tests/test_runtime_lifecycle.py` — pruebas del ciclo de vida y concurrencia.

## 8. Criterio de cierre de Fase 2

La coordinación de estado y sus pruebas unitarias quedan implementadas. Antes de considerar validada la fase en ejecución real PAPER, hay que comprobar localmente: arranque completo, primera transición a `RUNNING`, consulta `/status`, parada con Ctrl+C y reinicio del proceso. No se declara esa prueba integral como realizada en este informe.
