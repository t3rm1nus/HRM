# HRM — Sistema jerárquico de trading algorítmico

> **Estado de la documentación:** revisión del `master` de HRM consultado el 7 de noviembre de 2026.
>
> **Estado del proyecto:** en auditoría y estabilización. La existencia de módulos no implica que el flujo completo esté integrado ni certificado.
>
> **Ámbito de esta documentación:** describe la arquitectura identificada, la organización de pruebas y los límites de validación conocidos. Los resultados dinámicos solo se consideran acreditados cuando se indica el comando ejecutado y su salida.

---

## 1. Qué es HRM

HRM es un proyecto Python con una interfaz web React/TypeScript, orientado al análisis de mercado, la simulación y la automatización de estrategias de trading de criptomonedas.

El alcance funcional del repositorio contempla:

- Obtención y procesamiento de datos de mercado.
- Generación, evaluación y agregación de señales.
- Análisis táctico y estratégico.
- Gestión de cartera, posiciones y riesgo.
- Gestión y ejecución de órdenes.
- Simulación de operaciones en modo PAPER.
- Persistencia de carteras, operaciones e historiales.
- API REST y comunicación con la interfaz web.
- Backtesting, entrenamiento y evaluación de modelos.

Esta lista describe el alcance del código. **No certifica que todos los componentes estén conectados, operativos o validados de extremo a extremo.**

### Principio de seguridad durante el desarrollo

El desarrollo y las pruebas de esta etapa deben realizarse en PAPER. No se deben introducir credenciales privadas ni habilitar operaciones con fondos reales como parte de las comprobaciones descritas en este documento.

---

## 2. Estado de validación

| Área | Estado conocido | Trabajo pendiente |
|---|---|---|
| Arquitectura general | Inspección parcial | Completar el mapa de dependencias y los flujos reales de ejecución. |
| Perfiles PAPER | Separación de rutas y preparación de sesión implementadas | Validar el ciclo completo de restauración, ejecución, persistencia y reinicio. |
| Migración de datos PAPER | Copia atómica y protección frente a migraciones incompletas presentes en el código inspeccionado | Ejecutar pruebas en el entorno real del proyecto y verificar instalaciones con datos históricos. |
| Exchange simulado L1 | `force_reset()` actualiza los balances y reinicia los historiales en la implementación inspeccionada | Validar operaciones, comisiones, slippage, persistencia y reinicios de extremo a extremo. |
| Backend/API | Código presente | Validar arranque, endpoints, contratos, errores y apagado controlado. |
| Frontend | Aplicación React/TypeScript/Vite identificada | Ejecutar instalación, comprobación de tipos, tests y build. |
| Suite Python activa | `tests/` | Ejecutar la suite completa en el entorno virtual del proyecto y registrar resultados. |
| Pruebas históricas | `DEPRECATED_tests_antiguos solo backuo/` | Conservar sin incluirlas en la validación ordinaria. |
| Dependencias | Manifiestos presentes | Comprobar instalación reproducible y compatibilidad. |
| Backtesting | Código presente | Validar datos, costes, sesgos y reproducibilidad. |
| Integración L1-L2-L3 | No certificada | Trazar y probar el flujo real completo. |
| Trading LIVE | Fuera del alcance de esta fase | No habilitar durante la estabilización PAPER. |

### 2.1. Criterio para declarar una tarea validada

Una tarea solo se considerará validada cuando se disponga de:

1. Código y configuración identificados en la rama de trabajo.
2. Pruebas adecuadas al comportamiento que se quiere garantizar.
3. Comando exacto ejecutado.
4. Resultado real, incluyendo errores y dependencias ausentes.
5. Evidencia de que no se han alterado datos ajenos al caso probado.

La inspección estática ayuda a detectar defectos, pero no sustituye la ejecución de pruebas.

### 2.2. Fases

**Fase de auditoría:** inventario de arquitectura, entradas, dependencias, persistencia, API, frontend, seguridad y pruebas. Debe terminar con un informe de hallazgos confirmados, riesgos y áreas no comprobadas.

**Fase de estabilización:** corrección controlada de defectos técnicos autorizados, incorporación de pruebas de regresión en `tests/` y validación reproducible del backend y frontend.

No se deben cambiar estrategias, señales, indicadores ni parámetros de riesgo como efecto colateral de una corrección técnica.

---

## 3. Arquitectura

La arquitectura se organiza conceptualmente en tres niveles:

| Nivel | Responsabilidad prevista |
|---|---|
| L1 — Operacional | Órdenes, ejecución, validaciones, exchange y sincronización de cartera. |
| L2 — Táctico | Análisis de mercado, indicadores, características y señales tácticas. |
| L3 — Estratégico | Contexto general, régimen de mercado y decisiones estratégicas. |

Esta separación describe la organización conceptual del proyecto. No implica que todos los módulos participen en cada ciclo del runtime.

### 3.1. Directorios principales

| Ruta | Responsabilidad identificada |
|---|---|
| `main.py` | Punto de entrada del flujo principal inspeccionado. |
| `core/` | Cartera, estado, configuración, validación y utilidades. |
| `system/` | Preparación de sesión, bootstrap, coordinación y limpieza. |
| `comms/` | Configuración común, bus y esquemas de mensajes. |
| `l1_operational/` | Cliente de exchange, órdenes y ejecución operacional. |
| `l2_tactic/` | Señales, indicadores, características y modelos tácticos. |
| `l3_strategy/` | Régimen, sentimiento y análisis estratégico. |
| `storage/` | Rutas de datos y persistencia de operaciones PAPER. |
| `api/` | Servicios, esquemas y rutas de API. |
| `frontend/` | Aplicación web basada en React, TypeScript y Vite. |
| `backtesting/` | Herramientas de backtesting. |
| `ml_training/` | Scripts de entrenamiento offline. |
| `sentiment/` | Componentes relacionados con análisis de sentimiento. |
| `hacienda/` | Herramientas auxiliares de seguimiento fiscal. |
| `scripts/` | Scripts de soporte y diagnóstico. |
| `persistent_state/` | Determinados estados y marcadores persistentes. |
| `data/paper/` | Datos separados por perfil PAPER. |
| `tests/` | Suite oficial y activa de pruebas Python. |
| `DEPRECATED_tests_antiguos solo backuo/` | Copia histórica de pruebas antiguas, congelada y fuera de la validación ordinaria. |

La existencia de un directorio o módulo no demuestra que esté conectado al punto de entrada ni que funcione correctamente.

---

## 4. Punto de entrada y arranque

El flujo principal inspeccionado parte de `main.py`.

A nivel general, el proceso contempla:

1. Leer variables de entorno y determinar el modo solicitado.
2. Registrar el modo de sesión.
3. Invocar la preparación de sesión para seleccionar el perfil de datos.
4. Inicializar el contexto del sistema.
5. Inicializar los componentes que no hayan sido creados por el bootstrap.
6. Crear o recuperar el cliente de exchange y el gestor de cartera.
7. Preparar el exchange simulado y el gestor de órdenes.
8. Sincronizar la cartera con el cliente simulado.

### 4.1. Observaciones sobre el bootstrap

En el flujo inspeccionado, `main.py` importa `SystemBootstrap` desde `system/bootstrap.py`. También existe un `bootstrap.py` en la raíz con otra implementación. Son rutas distintas y no deben confundirse.

El camino manual de inicialización requiere pruebas de integración que confirmen:

- Que los componentes reciben la configuración esperada.
- Que el cliente de exchange utilizado es el previsto.
- Que la cartera y el gestor de órdenes comparten el estado correcto.
- Que la sincronización no sobrescribe datos válidos.
- Que el apagado libera los recursos y conserva el estado esperado.

La lectura estática no acredita por sí sola esos comportamientos.

---

## 5. PAPER: perfiles PERSISTENTE y LIMPIA

### 5.1. Objetivo de los perfiles

HRM contempla dos perfiles de datos PAPER:

- **PERSISTENTE:** conserva cartera, operaciones e historial entre reinicios normales.
- **LIMPIA:** utiliza un espacio de datos independiente para comenzar una sesión PAPER sin reutilizar los datos del perfil persistente.

Los perfiles deben permanecer aislados. Seleccionar o limpiar uno no debe eliminar ni modificar los archivos del otro.

### 5.2. Estructura de almacenamiento

El módulo `storage/perfil_datos.py` define rutas diferenciadas para los perfiles:

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

La ruta de LIVE se mantiene separada de los directorios PAPER. La separación de rutas es una condición necesaria, pero no suficiente, para certificar el aislamiento: también deben revisarse todos los lectores, escritores, procesos de limpieza y puntos de arranque.

### 5.3. Componentes implicados

- `storage/perfil_datos.py`: perfiles, rutas, migración legacy, limpieza y marcadores de reinicio.
- `storage/paper_trade_logger.py`: registro de operaciones PAPER.
- `system/preparar_sesion.py`: lectura del modo, migración y selección del perfil.
- `core/portfolio_manager.py`: restauración y persistencia de cartera.
- `api/servicios/servicio_operaciones.py`: acceso a operaciones del perfil activo.
- `system/system_cleanup.py`: limpieza técnica general.

### 5.4. Comportamiento de preparación de sesión

El código inspeccionado en `system/preparar_sesion.py` sigue esta lógica general:

1. Lee el modo PAPER configurado.
2. Si el proceso no está en LIVE, ejecuta la migración legacy a PERSISTENTE.
3. Selecciona el perfil PAPER correspondiente.
4. Si el perfil es LIMPIA, limpia únicamente el directorio de ese perfil.
5. Si el perfil es PERSISTENTE y existe un reinicio diferido pendiente, limpia ese perfil y consume el marcador.
6. Si el proceso está en LIVE, selecciona el perfil persistente sin ejecutar la migración ni la limpieza PAPER.
7. Registra el modo efectivo de sesión.

El modo LIMPIA no debe provocar borrados en LIVE. Si LIMPIA se solicita mientras se ejecuta LIVE, el modo efectivo debe ser PERSISTENTE y no debe ejecutarse una limpieza de datos PAPER.

### 5.5. Migración de datos legacy

La migración desde las rutas antiguas hacia el perfil PERSISTENTE utiliza copias atómicas en el código inspeccionado:

- La copia se prepara en un archivo temporal dentro del directorio de destino.
- El archivo temporal se sustituye por el destino mediante una operación atómica.
- Los errores de copia se propagan.
- El marcador de migración se escribe únicamente después de completar las copias previstas.
- Si una copia falla, la migración no debe quedar marcada como completada.
- Los archivos ya existentes en destino no deben sobrescribirse durante un reintento parcial.

El marcador `persistent_state/migracion_paper_perfiles_v1.json` indica que el proceso de migración terminó según la lógica del programa. Aun así, en una instalación concreta debe comprobarse el contenido de los archivos; la presencia del marcador no sustituye una copia de seguridad ni una verificación de integridad.

### 5.6. Exchange simulado: reinicio de balances

En la implementación operacional inspeccionada, `l1_operational/simulated_exchange_client.py`, `force_reset(initial_balances)`:

- Actualiza `self.balances`.
- Actualiza `self.initial_balances`.
- Limpia el historial de operaciones y órdenes.
- Reinicia el contador de identificadores de trade.

Por tanto, el defecto descrito en versiones anteriores de esta documentación —que `force_reset()` no actualizaba `self.balances`— ya no debe considerarse vigente para el código inspeccionado.

Esto no certifica por sí solo el proceso completo de restauración de cartera. Deben probarse también la sincronización con `PortfolioManager`, la persistencia y el comportamiento después de reiniciar el proceso.

### 5.7. Riesgos y comprobaciones pendientes

- Comprobar que todos los lectores y escritores utilizan las rutas del perfil activo.
- Comprobar que no existen rutas legacy que sigan escribiendo en paralelo.
- Comprobar que los procesos de limpieza no alcanzan el perfil no seleccionado.
- Comprobar que los reinicios diferidos se consumen una sola vez y solo después de una limpieza correcta.
- Comprobar que los loggers y cachés no comparten instancias entre perfiles.
- Comprobar que LIVE no llama a funciones de migración o limpieza PAPER.
- Comprobar que un fallo durante la migración no cambia el perfil activo ni deja un marcador falso de finalización.

**Conclusión PAPER:** el código inspeccionado contiene separación de rutas, selección de perfiles, migración con copia atómica y protecciones frente a copias incompletas. La certificación de aislamiento completo requiere ejecutar pruebas de integración en el entorno de HRM y comprobar el ciclo real de cartera, exchange, logger y reinicio.

---

## 6. Exchange simulado y ejecución de órdenes

La implementación operacional identificada está en `l1_operational/simulated_exchange_client.py`.

La interfaz inspeccionada incluye:

- `get_instance(...)` para obtener la instancia singleton.
- `get_balances()` para consultar los balances.
- `execute_order(...)` síncrono para simular una operación.
- `force_reset(...)` para reinicializar balances e historiales.
- `cleanup()` para liberar el singleton.

También existe una implementación distinta en `core/simulated_exchange_client.py`, con una interfaz diferente. No deben intercambiarse ni tratarse como si fueran la misma clase.

La ruta operacional de `l1_operational/order_executors.py` utiliza la implementación síncrona L1. Cualquier integración debe importar explícitamente la clase adecuada.

### 6.1. Pruebas necesarias

La validación integral debe cubrir, al menos:

- Compra y venta.
- Rechazo por fondos insuficientes.
- Rechazo por cantidad o precio inválidos.
- Cálculo de comisiones.
- Aplicación de slippage.
- Actualización de balances.
- Actualización de historial de trades y órdenes.
- Reinicio de balances.
- Sincronización con la cartera.
- Persistencia y restauración después de reiniciar.

No se debe considerar certificado el flujo completo por el mero hecho de que `execute_order()` funcione de forma aislada.

---

## 7. Configuración y modos de ejecución

Se han identificado varias fuentes de configuración:

- `core/config.py`.
- `comms/config.py`.
- Variables de entorno.
- Archivos YAML.
- Parámetros enviados al bootstrap y a los gestores.

La existencia de varias fuentes exige comprobar que todos los componentes interpretan el modo de forma coherente.

### 7.1. Reglas operativas

- PAPER debe ser el modo de desarrollo y validación.
- La selección del perfil PAPER debe realizarse antes de inicializar componentes que lean o escriban datos.
- LIVE no debe ejecutar limpiezas ni migraciones de datos PAPER.
- Una configuración incoherente debe producir un error explícito y seguro.
- No debe utilizarse un fallback silencioso que deje indicadores de modo contradictorios.
- Las credenciales privadas no deben almacenarse en el repositorio ni en logs.

### 7.2. Fuente única de verdad

Queda pendiente completar la consolidación de la configuración de modo. Antes de modificarla deben trazarse todas las lecturas de variables de entorno y atributos de modo para no romper contratos existentes.

---

## 8. Cliente Binance y límites de seguridad

`l1_operational/binance_client.py` contempla el uso de CCXT, datos públicos y el cliente simulado para PAPER.

La configuración del cliente debe garantizar que el modo solicitado, el modo efectivo, el cliente seleccionado y los endpoints utilizados sean coherentes.

Cualquier discrepancia entre `mode`, `paper_mode`, credenciales y endpoints debe resolverse con un fallo explícito y seguro. No debe interpretarse un fallback a PAPER como autorización para conservar un estado interno que indique LIVE.

Esta área requiere pruebas específicas antes de considerar segura cualquier configuración distinta de PAPER. Este README no proporciona instrucciones para habilitar trading real.

---

## 9. API y frontend

El repositorio contiene una API en `api/` y una aplicación web en `frontend/`.

Se han identificado rutas relacionadas con sesión y operaciones, incluyendo `POST /sesion/reiniciar-datos`, además de servicios que consultan el perfil activo.

La auditoría integral de esta capa debe comprobar:

- Contratos de entrada y salida.
- Validación de parámetros.
- Gestión de errores.
- Autenticación y autorización.
- CORS y configuración de red.
- WebSockets y reconexión.
- Consistencia entre estado del backend y la interfaz.
- Acciones de reinicio y sus efectos sobre los datos.
- Build y comprobación de tipos del frontend.

La presencia de rutas y componentes no demuestra que la integración frontend-backend esté validada.

---

## 10. L2, L3, modelos y sentimiento

El repositorio contiene módulos para análisis táctico y estratégico, indicadores, clasificación de régimen, sentimiento, modelos ML/RL y preparación de características.

La revisión debe distinguir entre:

1. Módulos existentes.
2. Módulos importados por el runtime.
3. Módulos ejecutados durante un ciclo real.
4. Módulos cuyos resultados influyen efectivamente en decisiones.
5. Módulos validados con datos y pruebas reproducibles.

No debe afirmarse que un proveedor de modelos o un servicio externo está integrado mediante API únicamente porque aparezcan nombres, clases o funciones relacionadas.

También debe comprobarse la disponibilidad de artefactos, modelos, pesos y datos requeridos por cada componente.

---

## 11. Backtesting y entrenamiento

Las carpetas `backtesting/` y `ml_training/` contienen herramientas relacionadas con pruebas históricas y entrenamiento offline.

No se debe considerar válido un resultado de backtest sin revisar, como mínimo:

- Procedencia, calidad y cobertura temporal de los datos.
- Separación temporal entre entrenamiento, validación y evaluación.
- Look-ahead bias y data leakage.
- Comisiones, spread y slippage.
- Liquidez y restricciones de ejecución.
- Órdenes parciales o no ejecutadas.
- Reproducibilidad de semillas, parámetros y versiones.
- Comparación con benchmarks adecuados.
- Robustez fuera de muestra.

La presencia de scripts de entrenamiento no demuestra que el aprendizaje automático esté conectado al runtime ni que los modelos se validen antes de su uso.

---

## 12. Tests y validación

### 12.1. Suite oficial y activa: `tests/`

**`tests/` es la única suite oficial y activa de pruebas automatizadas Python de HRM.**

Toda validación ordinaria de Python debe ejecutarse sobre esta carpeta. Los tests nuevos y las regresiones deben incorporarse aquí, respetando la estructura existente.

Comando para ejecutar la suite activa:

```powershell
python -m pytest tests/ -v
```

Para ejecutar un archivo concreto:

```powershell
python -m pytest tests/unit/test_preparar_sesion.py -v
```

Para comprobar la recogida de pruebas:

```powershell
python -m pytest --collect-only tests/ -q
```

Para consultar la versión del intérprete:

```powershell
python --version
```

Los informes de validación deben registrar el comando, la versión de Python, el número de pruebas recogidas, los resultados, la duración y los errores.

### 12.2. Suite histórica congelada

La carpeta histórica se encuentra en:

```text
DEPRECATED_tests_antiguos solo backuo/
```

Esta carpeta no forma parte de la suite oficial actual.

Política:

1. No ejecutarla en la validación ordinaria.
2. No usar sus resultados como criterio de aceptación de cambios actuales.
3. No modificar ni adaptar sus pruebas durante el mantenimiento ordinario.
4. No borrar sus archivos.
5. No trasladar sus pruebas automáticamente a `tests/`.
6. Conservarla como referencia histórica.
7. Consultarla solo cuando sea necesario estudiar un comportamiento anterior.

La carpeta histórica debe permanecer congelada. Su nombre contiene la grafía `backuo` tal como aparece actualmente en el repositorio; no debe renombrarse como parte de una corrección documental.

### 12.3. Scripts de diagnóstico manual

Los scripts de diagnóstico que estén fuera de `tests/` no forman automáticamente parte de la suite oficial.

Su ejecución debe ser manual, explícita y justificada. No deben ejecutarse automáticamente durante la importación de módulos ni interferir con la recogida de pruebas.

### 12.4. Configuración de pytest

`pytest.ini` configura `tests/` como ruta de recogida oficial. Aun así, los comandos de validación deben indicar explícitamente esa ruta:

```powershell
python -m pytest tests/ -v
```

No se debe ejecutar `python -m pytest` sin ruta como sustituto del comando oficial, porque podría recoger pruebas ajenas al ámbito de validación establecido.

Los mocks deben utilizarse solo cuando sean apropiados y no deben sustituir integraciones reales en pruebas cuyo objetivo sea validar dichas integraciones.

---

## 13. Dependencias y entorno

El repositorio contiene manifiestos de dependencias para distintos ámbitos del proyecto.

Antes de declarar reproducible la instalación se debe:

1. Inventariar imports y dependencias directas.
2. Distinguir dependencias obligatorias de extras opcionales.
3. Revisar manifiestos duplicados o incompletos.
4. Fijar versiones compatibles.
5. Documentar la versión de Python soportada.
6. Evitar rutas absolutas específicas de una máquina.
7. Mantener `.env.example` libre de secretos.
8. Verificar la instalación en un entorno virtual limpio.

No se deben actualizar paquetes indiscriminadamente. Cada cambio de versión debe justificarse y comprobarse mediante tests.

---

## 14. Instalación y ejecución

La instalación y el arranque completo no se consideran certificados por esta documentación. Antes de publicar un procedimiento definitivo deben validarse los pasos en un entorno limpio y sin datos de usuario.

Como mínimo, el procedimiento futuro debe documentar:

- Creación y activación del entorno virtual.
- Instalación de dependencias Python.
- Instalación del frontend con el lockfile correspondiente.
- Configuración PAPER sin credenciales privadas.
- Comprobaciones previas al arranque.
- Arranque del backend y frontend.
- Comprobaciones de salud.
- Apagado controlado.
- Conservación y recuperación de datos.
- Ejecución de la suite Python oficial.

### 14.1. Precaución con scripts de Windows

Antes de utilizar scripts de inicio, reinicio o limpieza, hay que inspeccionar su contenido y confirmar qué directorios modifican. No se debe ejecutar un script que pueda reemplazar el estado inicial o borrar datos persistentes como método de diagnóstico.

No se debe probar una limpieza sobre los datos reales del usuario. Las pruebas de borrado deben utilizar directorios temporales aislados.

---

## 15. Riesgos y asuntos pendientes

La siguiente tabla separa los asuntos identificados de aquellos que requieren nueva comprobación. No presenta como defectos vigentes los problemas que ya no aparecen en el código inspeccionado.

| Prioridad | Área | Estado documental | Acción necesaria |
|---|---|---|---|
| Alta | Aislamiento PAPER | Implementación de perfiles y rutas separadas presente | Pruebas integrales con cartera, exchange, logger y reinicios. |
| Alta | Migración legacy PAPER | Copia atómica y protección frente a copias fallidas presentes | Verificar instalaciones reales y escenarios de migración parcial. |
| Alta | Restauración del exchange simulado | `force_reset()` actualiza `balances` e `initial_balances` en el código inspeccionado | Probar sincronización completa con `PortfolioManager`. |
| Alta | Doble implementación de exchange simulado | Clases distintas en `l1_operational/` y `core/` | Revisar imports y contratos; evitar intercambios accidentales. |
| Alta | Limpieza y singletons | Comportamiento de `system/system_cleanup.py` requiere revisión integral | Comprobar reset de instancias y propagación de errores. |
| Alta | Configuración de modo | Fuentes de configuración repartidas | Trazar y consolidar la fuente de verdad sin cambiar contratos inadvertidamente. |
| Alta | Cliente Binance | Coherencia de modo, endpoints y fallback requiere validación | Añadir pruebas de configuración segura; mantener PAPER. |
| Media | Configuración global importada | Puede conservar valores leídos al importar | Comprobar orden de carga y cambios de entorno. |
| Media | Limpieza de temporales | Hay limpieza recursiva en el área de datos que requiere delimitarse | Verificar que no alcance archivos de otros procesos o perfiles. |
| Pendiente | API y WebSockets | Auditoría incompleta | Validar contratos, seguridad y reconexión. |
| Pendiente | Frontend | Build y tests no acreditados en esta fase | Ejecutar instalación, typecheck, tests y build. |
| Pendiente | Backtesting | Resultados no certificados | Revisar sesgos, costes y reproducibilidad. |
| Pendiente | Integración L1-L2-L3 | Flujo completo no certificado | Trazar el ciclo real de decisión y ejecución. |

Un asunto marcado como pendiente no equivale a un fallo reproducido. Debe verificarse antes de corregirlo.

---

## 16. Plan de trabajo

### Fase A — Auditoría técnica

1. Inventariar puntos de entrada y rutas activas/legacy.
2. Completar el mapa de dependencias L1-L2-L3.
3. Trazar fuentes de mercado, bus, coordinación y estado.
4. Auditar validadores, controles de riesgo, gateway y gestores de órdenes.
5. Completar el mapa de API, WebSockets y frontend.
6. Revisar backtesting, entrenamiento y dependencias.
7. Completar la auditoría de persistencia y perfiles PAPER.
8. Registrar defectos confirmados, riesgos y áreas no comprobadas.
9. Emitir el informe de auditoría con evidencias y prioridades.

### Fase B — Estabilización técnica

Tras revisar y aprobar el alcance de los cambios:

1. Consolidar dependencias y entorno reproducible.
2. Corregir errores técnicos confirmados.
3. Unificar la lectura de configuración cuando esté aprobado.
4. Revisar rutas absolutas y configuración de entorno.
5. Corregir los defectos autorizados sin alterar estrategias.
6. Añadir pruebas de regresión exclusivamente en `tests/`.
7. Ejecutar la suite Python oficial.
8. Ejecutar comprobaciones estáticas.
9. Instalar y validar el frontend.
10. Validar arranque y apagado controlados en PAPER.
11. Emitir un informe con comandos, versiones, resultados y limitaciones.

Comandos mínimos de validación Python:

```powershell
python -m pytest --collect-only tests/ -q
python -m pytest tests/ -v
```

La carpeta `DEPRECATED_tests_antiguos solo backuo/` permanecerá congelada y fuera de la validación ordinaria.

---

## 17. Convenciones de desarrollo

- Aplicar Clean Code y responsabilidad única.
- Mantener funciones pequeñas y con una sola responsabilidad.
- En código nuevo, utilizar nombres de funciones en español cuando no exista una interfaz o convención previa que deba conservarse.
- Mantener compatibilidad con los contratos existentes, salvo cambio aprobado y documentado.
- No modificar estrategias, señales, indicadores ni parámetros de riesgo durante la estabilización técnica.
- No ocultar errores con datos ficticios.
- Utilizar mocks solo cuando sean apropiados y documentarlos.
- No sustituir integraciones reales por simulaciones para hacer pasar pruebas de integración.
- Incorporar pruebas nuevas y de regresión exclusivamente en `tests/`.
- No modificar la suite histórica congelada.
- No registrar secretos, claves ni datos sensibles en logs.
- No afirmar que una prueba ha pasado si no se ha ejecutado y comprobado su resultado.

---

## 18. Seguridad operativa

HRM es software experimental. PAPER debe ser el ámbito de desarrollo y validación.

- No introducir claves privadas de trading en el repositorio.
- No habilitar operaciones con fondos reales durante esta etapa.
- No conceder permisos de retirada a claves de API.
- No asumir que una variable de entorno constituye por sí sola una barrera de seguridad.
- No ejecutar limpiezas o migraciones sin conocer las rutas afectadas.
- Mantener copias de seguridad verificadas antes de cualquier cambio sobre datos persistentes.
- Realizar pruebas destructivas únicamente sobre directorios temporales.
- Mantener los perfiles PAPER y los datos LIVE separados.
- Ante una configuración ambigua, detener el proceso en lugar de continuar con un modo incierto.

---

## 19. Resumen

HRM dispone de una arquitectura modular para análisis, estrategia, ejecución simulada, persistencia, API y frontend. El repositorio contiene una implementación de perfiles PAPER separados, migración legacy con copia atómica y un exchange simulado cuya función `force_reset()` actualiza los balances en el código inspeccionado.

Estos elementos son una base técnica, no una certificación integral. Aún es necesario validar el flujo completo de cartera y exchange, la limpieza y los singletons, la configuración de modos, la integración L1-L2-L3, la API, el frontend y el backtesting.

La política de pruebas queda definida sin ambigüedad:

- **`tests/` es la única suite oficial y activa de pruebas Python.**
- **`DEPRECATED_tests_antiguos solo backuo/` es la copia histórica congelada.**
- Los tests nuevos y las regresiones se incorporan exclusivamente a `tests/`.
- Los comandos oficiales de pytest deben especificar `tests/`.
- Los resultados de pruebas deben documentarse únicamente después de ejecutarlas.

**Estado final:** la documentación queda corregida en cuanto a rutas de pruebas y a los hallazgos PAPER que ya no corresponden al código inspeccionado. La estabilidad integral de HRM sigue pendiente de validación dinámica completa.
