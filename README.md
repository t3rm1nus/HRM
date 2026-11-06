# HRM — Sistema jerárquico de trading algorítmico

> **Estado de esta documentación:** actualización basada en la inspección estática del código y de las salidas de consola compartidas durante la auditoría de noviembre de 2026.
>
> **Estado del proyecto:** en auditoría técnica. La Fase 0 no está cerrada y la Fase 1 no se ha ejecutado ni validado completamente.
>
> **Importante:** este documento distingue entre código encontrado, código conectado al punto de entrada y comportamiento probado. La presencia de una clase o función no demuestra que el sistema funcione de extremo a extremo.

---

## 1. Qué es HRM

HRM es un proyecto Python con una interfaz web React/TypeScript, orientado al análisis y a la simulación de estrategias de trading de criptomonedas.

El diseño del proyecto contempla:

- Obtención y procesamiento de datos de mercado.
- Generación y agregación de señales.
- Análisis táctico y estratégico.
- Gestión de cartera, posiciones y riesgo.
- Gestión de órdenes y ejecución simulada.
- Persistencia de carteras y operaciones PAPER.
- API REST y comunicación con la interfaz web.
- Backtesting, entrenamiento y evaluación de modelos.

Estas capacidades describen el alcance del código. **No constituyen una certificación de que todos los componentes estén integrados, operativos o validados.**

HRM debe mantenerse en modo PAPER durante el desarrollo y las pruebas. Este README no documenta cómo habilitar operaciones con fondos reales.

## 2. Estado de validación

| Área | Estado conocido | Qué falta |
|---|---|---|
| Auditoría de arquitectura | Parcial | Completar el inventario y el flujo de todos los componentes. |
| Aislamiento PAPER PERSISTENTE/LIMPIA | Implementación presente; con riesgos detectados | Completar las pruebas de restauración de balances, migración y aislamiento de extremo a extremo. |
| Arranque del backend | No certificado | Validar en entorno reproducible, sin alterar datos existentes. |
| Frontend | No certificado en esta auditoría | `npm ci`, typecheck, tests y build. |
| Tests Python oficiales | `tests_nuevos/` es la única suite activa | Ejecutar y documentar sus resultados reales. |
| Tests históricos | `tests/` está congelada y fuera de la validación oficial | Conservar como referencia histórica, sin modificar ni ejecutar en la validación ordinaria. |
| Dependencias | No consolidadas de forma demostrada | Auditar imports y fijar versiones compatibles. |
| Backtesting | Código presente | Validar datos, costes, sesgos y reproducibilidad. |
| Integración completa L1-L2-L3 | No certificada | Trazar y probar el flujo real. |
| Trading real | Fuera del alcance de esta fase | No habilitar. |

### 2.1. Fases del proyecto

**Fase 0 — Auditoría y congelación:** en curso. Se han inspeccionado partes relevantes del arranque, configuración, perfiles PAPER, persistencia, exchange simulado, cartera, gestores de órdenes y limpieza. Todavía falta completar la revisión de todas las áreas y emitir el informe definitivo `INFORME_FASE_0.md`.

**Fase 1 — Estabilización técnica:** pendiente de validación completa. La validación automatizada de Python debe realizarse exclusivamente sobre `tests_nuevos/`. También quedan pendientes las comprobaciones de instalación, compilación, arranque y apagado. No se debe utilizar la suite histórica `tests/` como criterio de aceptación.

Hasta cerrar la Fase 0 y autorizar expresamente los cambios, no se deben modificar código, estrategias, señales, indicadores, parámetros de riesgo, configuración PAPER/LIVE ni datos persistentes.

## 3. Arquitectura del proyecto

La arquitectura se organiza conceptualmente en tres niveles:

| Nivel | Responsabilidad prevista |
|---|---|
| L1 — Operacional | Gestionar órdenes, validaciones, ejecución y sincronización de cartera. |
| L2 — Táctico | Analizar el mercado y producir señales tácticas. |
| L3 — Estratégico | Evaluar el contexto general y aportar decisiones estratégicas. |

Esta división es la arquitectura prevista. No debe asumirse que cada módulo participa en todos los ciclos del punto de entrada real.

### 3.1. Carpetas principales

| Ruta | Contenido identificado |
|---|---|
| `main.py` | Punto de entrada utilizado en el flujo principal inspeccionado. |
| `core/` | Cartera, estado, configuración, validación y utilidades. |
| `system/` | Preparación de sesión, bootstrap, coordinación y limpieza. |
| `comms/` | Configuración común, bus y esquemas de mensajes. |
| `l1_operational/` | Cliente de exchange, órdenes y ejecución operacional. |
| `l2_tactic/` | Señales, indicadores, características y modelos tácticos. |
| `l3_strategy/` | Régimen, sentimiento y análisis estratégico. |
| `storage/` | Rutas de datos y registro de operaciones PAPER. |
| `api/` | Servicios y rutas de API. |
| `frontend/` | Aplicación web; tecnología declarada React, TypeScript y Vite. |
| `backtesting/` | Herramientas de backtesting. |
| `ml_training/` | Scripts de entrenamiento offline. |
| `tests/` | Suite histórica congelada. No forma parte de la validación oficial actual. |
| `tests_nuevos/` | Suite oficial y activa de pruebas automatizadas Python. |
| `hacienda/` | Herramientas auxiliares de seguimiento fiscal. |
| `scripts/` | Scripts de soporte y control. |
| `persistent_state/` | Configuración y determinados estados persistentes. |
| `data/paper/` | Datos separados por perfil PAPER. |

La existencia de una carpeta no demuestra que sus módulos estén conectados al runtime.

## 4. Punto de entrada y flujo de arranque inspeccionado

El flujo principal inspeccionado parte de `main.py`.

A nivel general, el código:

1. Lee variables de entorno y determina el modo solicitado.
2. Registra el modo de sesión.
3. Llama a `preparar_sesion(...)` para seleccionar el perfil de datos.
4. Inicializa el contexto del sistema.
5. Inicializa manualmente componentes que no llegan creados por el contexto inicial.
6. Crea el cliente de exchange y el gestor de cartera.
7. Prepara el exchange simulado y el gestor de órdenes.
8. Sincroniza la cartera con el cliente simulado.

### 4.1. Observaciones sobre el bootstrap

En el camino inspeccionado, `main.py` importa `SystemBootstrap` desde `system/bootstrap.py`. Su método `initialize_system()` devuelve un contexto con un diccionario de componentes inicialmente vacío; después `main.py` realiza inicializaciones manuales.

También existe un `bootstrap.py` en la raíz con otra clase y otra ruta de inicialización. No deben confundirse ambas implementaciones.

En el flujo manual observado:

- `BinanceClient()` se instancia sin pasar explícitamente el modo.
- `PortfolioManager` se inicializa antes de llamar a `bootstrap_simulated_exchange(config)`.
- Posteriormente se obtiene el cliente simulado y se inyecta en `OrderManager`.
- Se realiza una sincronización de la cartera con el cliente simulado.

Este orden requiere una prueba de integración específica. La inspección estática no demuestra que todos los objetos compartan siempre la misma instancia ni que la restauración de estado sea correcta.

## 5. PAPER: perfiles PERSISTENTE y LIMPIA

### 5.1. Rutas previstas

El módulo `storage/perfil_datos.py` define dos perfiles PAPER:

```text
data/
└── paper/
    ├── persistente/
    │   ├── portfolio_state.json
    │   ├── portfolio_log.csv
    │   └── paper_trades/
    │       ├── paper_trades.json
    │       ├── paper_trades.csv
    │       └── summary.json
    └── limpia/
        ├── portfolio_state.json
        ├── portfolio_log.csv
        └── paper_trades/
            ├── paper_trades.json
            ├── paper_trades.csv
            └── summary.json
```

La estructura física observada en la instalación inspeccionada contiene ambos directorios y sus archivos principales. Esto confirma la existencia de los directorios, no la integridad de los datos ni el aislamiento completo en ejecución.

### 5.2. Componentes implicados

- `storage/perfil_datos.py`: selección del perfil y construcción de rutas.
- `storage/paper_trade_logger.py`: logger de operaciones PAPER, con caché de instancias diferenciada por directorio resuelto.
- `system/preparar_sesion.py`: migración, selección del perfil y limpieza al preparar la sesión.
- `core/portfolio_manager.py`: restauración y persistencia de la cartera.
- `api/servicios/servicio_operaciones.py`: acceso a operaciones del perfil activo.
- `system/system_cleanup.py`: limpieza técnica general.

### 5.3. Comportamiento del código inspeccionado

`preparar_sesion.py` lee el modo configurado y, cuando no está en LIVE, ejecuta la migración legacy y selecciona el perfil PAPER. Si el perfil seleccionado es LIMPIA, limpia el directorio de ese perfil. Si es PERSISTENTE y hay un marcador de reinicio pendiente, solicita la limpieza de ese perfil y después consume el marcador.

El módulo `storage/perfil_datos.py` valida los nombres de perfil admitidos y construye rutas distintas para `persistente` y `limpia`.

La limpieza técnica de `system/system_cleanup.py` no elimina directamente los historiales PAPER normales. Sí elimina determinados archivos temporales encontrados recursivamente bajo `data/`; esa regla debe mantenerse bajo revisión para evitar borrar temporales que pertenezcan a otros procesos.

### 5.4. Riesgos detectados que impiden dar el aislamiento por certificado

**A. Restauración de balances del exchange simulado**

En `l1_operational/simulated_exchange_client.py`, el método `force_reset()` inspeccionado actualiza `initial_balances`, limpia el historial en memoria y reinicia el contador de operaciones, pero no actualiza `self.balances`.

Por tanto, no se puede afirmar que ese método restaure los balances efectivos del exchange simulado. Es especialmente relevante porque `core/portfolio_manager.py` y `main.py` llaman a `force_reset()` durante la sincronización/restauración.

**B. Migración legacy marcada como completada aunque pueda ser parcial**

En `storage/perfil_datos.py`, la migración registra el marcador `persistent_state/migracion_paper_perfiles_v1.json`. Las funciones auxiliares de copia capturan errores de E/S y pueden devolver cero; el proceso puede terminar escribiendo el marcador aunque una o varias copias no se hayan realizado correctamente.

Esto puede impedir que una migración incompleta se reintente automáticamente. El marcador existente en una instalación concreta acredita que se registró una migración, pero no que todos los archivos se copiaran correctamente.

**C. Dos implementaciones de exchange simulado**

Se han identificado:

- `l1_operational/simulated_exchange_client.py`: implementación síncrona, con `get_instance()`, `get_balances()` y `execute_order()` síncrono. Es la que utiliza el flujo operacional inspeccionado.
- `core/simulated_exchange_client.py`: implementación distinta, con métodos asíncronos y una interfaz diferente.

No deben intercambiarse ni tratarse como si fueran la misma clase. La duplicidad aumenta el riesgo de imports equivocados y contratos incompatibles.

**D. Limpieza y reinicio de estado**

En `system/system_cleanup.py`, `memory_reset()` elimina entradas de `sys.modules`, pero no llama al método `PortfolioManager.reset_instance()` que existe en la clase. Eliminar el módulo de la caché de imports no garantiza que una instancia singleton ya creada desaparezca.

Además, `perform_full_cleanup(mode="paper")` no utiliza el argumento `mode` en el fragmento inspeccionado y no propaga necesariamente los resultados `success=False` de sus suboperaciones al resultado global.

**Conclusión PAPER:** existe una separación física de rutas y una implementación de selección de perfiles, pero hay defectos/riesgos en la restauración de balances y en la migración. El aislamiento PAPER PERSISTENTE/LIMPIA todavía no puede considerarse validado de extremo a extremo.

## 6. Exchange simulado y ejecución de órdenes

La implementación operacional identificada está en `l1_operational/simulated_exchange_client.py`.

La interfaz inspeccionada incluye:

- `get_instance(...)` para obtener la instancia singleton.
- `get_balances()` para consultar balances.
- `execute_order(...)` síncrono para simular una operación.
- `force_reset(...)` para reinicializar determinados datos de la instancia.

`l1_operational/order_executors.py` llama a `execute_order()` sin `await`, lo que coincide con la firma síncrona de la implementación L1 inspeccionada. Después consulta balances y llama a `PortfolioManager.sync_from_exchange_async(...)`.

Hay otra clase con el mismo nombre en `core/simulated_exchange_client.py`, cuya ejecución es asíncrona. Cualquier integración debe importar explícitamente la implementación adecuada.

**Pendiente:** comprobar con tests de integración que una orden PAPER actualiza de forma coherente balances, cartera, historial, identificadores, valoración y persistencia, incluyendo reinicios. No se ha certificado ese flujo completo.

## 7. Configuración y modos

Se han encontrado varias fuentes de configuración:

- `core/config.py`.
- `comms/config.py`.
- Variables de entorno.
- Configuraciones YAML.
- Parámetros pasados al bootstrap y a los gestores.

`core/config.py` reconoce los modos `live`, `testnet`, `backtest` y `simulated` en `EnvironmentConfig`; el modo literal `paper` no aparece entre los modos aceptados por ese cargador. También mantiene una variable global de modo forzado.

`comms/config.py` crea una instancia global `config` al importar el módulo y lee `PAPER_MODE` en ese momento. Si el entorno cambia después de la importación, esa instancia puede conservar el valor anterior.

En el flujo manual inspeccionado, `main.py` instancia `BinanceClient()` sin pasar `mode`; el constructor usa `paper` como valor predeterminado. Otros componentes consultan variables distintas para decidir el modo. Esto requiere una fuente única de verdad y pruebas específicas.

**Regla del proyecto durante esta auditoría:** PAPER debe seguir siendo el modo predeterminado. No habilitar LIVE ni introducir credenciales de trading real.

## 8. Cliente Binance

`l1_operational/binance_client.py` contempla el uso de CCXT, datos públicos y un cliente simulado para PAPER.

En el constructor inspeccionado, si se solicita `mode="live"` y faltan las claves, `paper_mode` puede pasar a `True` mientras `self.mode` conserva el valor `live` y se prepara el cliente CCXT con endpoints de mainnet. Esto es una configuración contradictoria y debe tratarse como un riesgo de diseño; no demuestra por sí solo que se envíen órdenes.

La respuesta correcta de seguridad debe ser fallar de forma explícita y segura ante una configuración incoherente, no depender de un fallback silencioso. Su corrección pertenece a una fase posterior y requiere autorización.

## 9. API y frontend

El proyecto contiene una carpeta `api/` y una aplicación en `frontend/`. La documentación previa identifica React, TypeScript y Vite para el frontend.

Se han localizado rutas relacionadas con sesión y operaciones, incluyendo `POST /sesion/reiniciar-datos`, y servicios que consultan el perfil activo.

No se ha completado en esta auditoría la revisión de todos los endpoints, contratos, autenticación, autorización, WebSockets, reconexión, validación de entradas ni integración frontend-backend.

Tampoco se han ejecutado `npm ci`, typecheck, tests ni build. Por tanto, no se afirma que el frontend instale o compile correctamente.

## 10. L2, L3, modelos y sentimiento

El repositorio contiene carpetas y módulos para análisis táctico y estratégico, indicadores, clasificación de régimen, sentimiento, modelos ML/RL y preparación de características.

La existencia de estos módulos no demuestra que estén conectados al flujo principal ni que dispongan de los artefactos necesarios.

En una inspección anterior del ZIP se documentaron referencias a modelos y rutas de artefactos que no estaban incluidos en ese paquete. Esos resultados deben volver a contrastarse con el árbol de trabajo actual antes de tratarlos como estado definitivo del repositorio local.

No se debe afirmar que proveedores comerciales de LLM estén integrados mediante API solo porque existan funciones o nombres de modelos que los mencionen.

## 11. Backtesting y entrenamiento

Las carpetas `backtesting/` y `ml_training/` contienen herramientas para pruebas históricas y entrenamiento offline.

No se ha ejecutado un backtest reproducible durante esta auditoría. Antes de considerar válidos sus resultados habrá que revisar, como mínimo:

- Procedencia y calidad de los datos.
- Separación temporal entre entrenamiento y evaluación.
- Look-ahead bias y leakage.
- Comisiones, spread, slippage y restricciones de ejecución.
- Tratamiento de órdenes no ejecutadas y liquidez.
- Reproducibilidad de semillas, parámetros y versiones.
- Comparación con benchmarks apropiados.

La presencia de scripts de entrenamiento tampoco demuestra que el autoaprendizaje esté conectado al runtime ni que los modelos resultantes se validen antes de usarse.

## 12. Tests y validación

### 12.1. Suite oficial y activa: `tests_nuevos/`

**`tests_nuevos/` es la única suite oficial de pruebas automatizadas Python de HRM.**

Toda validación ordinaria, ejecución de regresiones y comprobación de cambios debe realizarse sobre esta carpeta.

Los tests nuevos deben incorporarse a `tests_nuevos/`, respetando su organización y estructura existente.

**Comando oficial para ejecutar la suite Python:**

```powershell
python -m pytest tests_nuevos/ -v
```

Para ejecutar un archivo concreto:

```powershell
python -m pytest tests_nuevos/ruta/del_test.py -v
```

Para comprobar qué pruebas se van a recoger dentro de la suite activa:

```powershell
python -m pytest --collect-only tests_nuevos/ -q
```

La validación debe registrar:

- Comando exacto utilizado.
- Versión de Python.
- Número de tests recogidos.
- Número de tests superados y fallidos.
- Duración.
- Errores y dependencias ausentes.
- Resultado final.

Los tests de regresión de cualquier corrección aprobada deben añadirse a `tests_nuevos/`.

### 12.2. Suite histórica congelada: `tests/`

La carpeta `tests/` contiene pruebas históricas del proyecto y está **congelada**.

Esta carpeta no forma parte de la suite oficial actual de HRM.

Reglas obligatorias:

1. No ejecutar `tests/` como parte de la validación ordinaria.
2. No utilizar sus resultados para determinar si una modificación actual pasa o falla.
3. No adaptar sus pruebas a los cambios actuales de arquitectura.
4. No modificar sus archivos como parte del mantenimiento ordinario.
5. No borrar sus archivos.
6. No trasladar sus archivos a `tests_nuevos/` automáticamente.
7. Conservarlos en el repositorio como referencia histórica.
8. Consultarlos únicamente cuando sea necesario recuperar o estudiar un comportamiento anterior y exista una autorización expresa para ello.

La congelación significa que el contenido de `tests/` se conserva tal como está, sin participar en el ciclo normal de desarrollo y validación.

### 12.3. Scripts de diagnóstico manual

Los scripts de diagnóstico manual que puedan existir en el repositorio se consideran herramientas auxiliares.

No deben confundirse con la suite oficial de pruebas automatizadas.

Su ejecución debe ser explícita y voluntaria. No deben ejecutarse automáticamente durante la importación de módulos ni interferir con la recogida de tests de `tests_nuevos/`.

La presencia de un script de diagnóstico no implica que forme parte de la suite activa.

### 12.4. Separación obligatoria de las suites

La separación entre ambas carpetas es una decisión de organización del proyecto:

| Elemento | Función | Política |
|---|---|---|
| `tests_nuevos/` | Suite oficial de pruebas automatizadas | Activa. Se ejecuta, mantiene y amplía. |
| `tests/` | Suite histórica | Congelada. Se conserva sin modificaciones y fuera de la validación ordinaria. |
| Scripts de diagnóstico | Comprobaciones manuales específicas | Ejecución manual y explícita. |

**Importante:** los comandos oficiales de pytest deben indicar expresamente `tests_nuevos/`. No se debe utilizar `python -m pytest` sin especificar la ruta de la suite, porque podría recoger pruebas históricas u otros archivos de prueba fuera del ámbito autorizado.

No se deben borrar carpetas ni archivos históricos para resolver problemas de descubrimiento de pytest. La solución debe estar en la selección explícita de la suite activa y, cuando corresponda, en la configuración de descubrimiento.

Los mocks deben identificarse y no sustituir integraciones reales cuando se esté validando una integración.

## 13. Dependencias y entorno

Se han encontrado manifiestos parciales en el proyecto y dependencias de distintos ámbitos: cliente de exchange, datos, API, frontend, ML/RL y sentimiento.

No se ha certificado todavía una instalación limpia y reproducible. La Fase 1 deberá:

1. Inventariar imports reales y dependencias directas.
2. Separar dependencias obligatorias y extras opcionales.
3. Resolver manifiestos duplicados o incompletos.
4. Fijar versiones compatibles y documentar la versión de Python.
5. Evitar rutas absolutas de Windows.
6. Crear o completar `.env.example` sin secretos.
7. Validar instalación y pruebas en un entorno aislado.

No se deben instalar paquetes indiscriminadamente ni actualizar versiones sin comprobar compatibilidad.

## 14. Instalación y ejecución

**No se ofrece aquí un procedimiento de arranque certificado**, porque la Fase 1 aún no se ha completado y la Fase 0 sigue abierta.

No ejecutes scripts de inicio, reinicio o limpieza como método de diagnóstico mientras no se haya revisado su efecto sobre los datos. En particular, se ha identificado un script de Windows que puede reemplazar el estado inicial y borrar datos de estado/operaciones; debe auditarse antes de recomendar su uso.

Cuando la Fase 1 esté autorizada y completada, esta sección deberá incluir un procedimiento reproducible que cubra:

- Creación del entorno virtual.
- Instalación exacta del backend.
- Instalación del frontend mediante `npm ci`.
- Configuración PAPER sin credenciales privadas.
- Comprobaciones previas al arranque.
- Arranque y apagado controlados.
- Recuperación y conservación de los datos.
- Comandos de tests y validación, utilizando exclusivamente `tests_nuevos/` para la suite oficial Python.

## 15. Riesgos y bloqueantes conocidos

| Prioridad | Hallazgo | Archivo / área | Estado |
|---|---|---|---|
| Crítica para restauración PAPER | `force_reset()` no actualiza `self.balances`. | `l1_operational/simulated_exchange_client.py` | Defecto confirmado por inspección estática. |
| Alta | La migración puede escribir el marcador aunque alguna copia falle. | `storage/perfil_datos.py` | Riesgo confirmado por flujo de código. |
| Alta | Dos implementaciones distintas de `SimulatedExchangeClient`. | `l1_operational/` y `core/` | Duplicidad confirmada. |
| Alta | El reinicio de memoria no llama a `PortfolioManager.reset_instance()`. | `system/system_cleanup.py` | Defecto confirmado por inspección. |
| Alta | `perform_full_cleanup()` puede informar éxito global aunque una suboperación devuelva `success=False`. | `system/system_cleanup.py` | Riesgo confirmado por flujo de código. |
| Alta | Configuración de modo repartida entre módulos y variables. | `main.py`, `core/config.py`, `comms/config.py`, clientes | Riesgo confirmado; falta prueba integral. |
| Alta | Cliente Binance puede quedar con `mode=live` y `paper_mode=True` al faltar claves. | `l1_operational/binance_client.py` | Configuración contradictoria confirmada; no prueba envío de órdenes. |
| Media | `comms.config.config` se crea al importar el módulo. | `comms/config.py` | Riesgo de configuración desactualizada si cambia el entorno después. |
| Media | Limpieza recursiva de ciertos temporales dentro de `data/`. | `system/system_cleanup.py` | Comportamiento confirmado; alcance debe revisarse. |
| Pendiente | Integración real de WebSockets, contratos API y frontend. | `api/`, `frontend/` | No auditado por completo. |
| Pendiente | Compilación, suite oficial de tests y arranque completo. | Todo el repositorio | No ejecutado/validado completamente. |

Los hallazgos anteriores son resultados de inspección estática de los fragmentos y archivos revisados. No equivalen a una reproducción dinámica de todos los fallos.

## 16. Orden recomendado de trabajo

### Fase 0 — Completar auditoría y congelación

1. Inventariar todos los puntos de entrada y distinguir rutas activas de rutas legacy.
2. Completar el mapa de dependencias L1-L2-L3.
3. Trazar fuentes de mercado, bus, coordinación y estado.
4. Auditar validadores, controles de riesgo, `ExecutionGateway` y gestores de órdenes.
5. Completar el mapa de API, WebSockets y frontend.
6. Revisar backtesting, dependencias y organización de las pruebas.
7. Completar auditoría de persistencia, perfiles PAPER y protección de datos.
8. Registrar defectos confirmados, riesgos y puntos no comprobados.
9. Generar `INFORME_FASE_0.md`.
10. Presentar el resumen y esperar autorización.

### Fase 1 — Estabilización técnica, solo tras autorización

1. Consolidar dependencias y entorno reproducible.
2. Corregir imports y errores de sintaxis sin cambiar estrategias.
3. Unificar la lectura de configuración y documentar variables.
4. Crear `.env.example` sin secretos.
5. Resolver rutas absolutas.
6. Corregir los defectos técnicos aprobados, con tests de regresión en `tests_nuevos/`.
7. Ejecutar exclusivamente la suite oficial Python:

   ```powershell
   python -m pytest tests_nuevos/ -v
   ```

8. Comprobar la recogida de pruebas de la suite activa:

   ```powershell
   python -m pytest --collect-only tests_nuevos/ -q
   ```

9. Verificar que la validación ordinaria no ejecuta ni modifica la suite histórica `tests/`.
10. Ejecutar las comprobaciones estáticas.
11. Ejecutar `npm ci`, typecheck, tests y build del frontend.
12. Validar arranque y apagado controlados en PAPER.
13. Generar `INFORME_FASE_1.md` con comandos, versiones, resultados y fallos pendientes.

La carpeta `tests/` permanecerá congelada durante estas fases. No se deben modificar, borrar, trasladar ni adaptar sus pruebas como parte de la estabilización ordinaria.

**No se avanzará a otra fase sin cerrar y documentar la anterior.**

## 17. Convenciones de desarrollo

- Aplicar Clean Code y responsabilidad única.
- Mantener funciones pequeñas y con una sola responsabilidad.
- En código nuevo, usar nombres de funciones en español cuando no exista una convención pública o interfaz previa que deba preservarse.
- Mantener compatibilidad con contratos existentes, salvo cambio aprobado y documentado.
- No modificar estrategias, señales, indicadores ni parámetros de riesgo durante la estabilización técnica.
- No ocultar errores con datos ficticios.
- Usar mocks solo cuando corresponda y marcarlos claramente.
- No sustituir integraciones reales por implementaciones simuladas para hacer pasar tests.
- Incorporar las pruebas nuevas y de regresión exclusivamente en `tests_nuevos/`.
- No modificar la suite histórica congelada `tests/`.
- No registrar secretos, claves ni datos sensibles en logs.

## 18. Seguridad operativa

HRM es software experimental. La simulación PAPER debe permanecer aislada y ser el único ámbito de validación de esta etapa.

- No introducir claves privadas de trading.
- No habilitar operaciones con fondos reales.
- No conceder permisos de retirada a claves de API.
- No asumir que un indicador de modo o una variable de entorno constituye por sí sola una barrera de seguridad.
- No ejecutar limpiezas o migraciones sin comprender previamente qué rutas afectan.
- Mantener copias de seguridad verificadas antes de cualquier cambio autorizado sobre persistencia.

## 19. Resumen

HRM contiene una base de código amplia y una arquitectura modular, pero todavía no se ha certificado su funcionamiento integral. Se han identificado defectos relevantes en la restauración de balances PAPER, la migración de datos y el reinicio de singletons, además de duplicidad entre implementaciones del exchange simulado y configuración de modos repartida.

La política de pruebas queda establecida de forma explícita:

- **`tests_nuevos/` es la única suite oficial y activa de HRM.**
- **`tests/` es una suite histórica congelada que se conserva en el repositorio y queda fuera de la validación ordinaria.**
- Los tests nuevos y las regresiones deben añadirse exclusivamente a `tests_nuevos/`.
- Los comandos oficiales de pytest deben especificar la ruta `tests_nuevos/`.

**La Fase 0 está parcialmente realizada y debe completarse con un informe formal. La Fase 1 está pendiente de validación completa.** El siguiente paso correcto es terminar la auditoría sin modificar el repositorio y presentar `INFORME_FASE_0.md` para revisión y autorización.