INFORME FASE 0 — HRM
Auditoría técnica y congelación del proyecto
Fecha: 5 de noviembre de 2026
Repositorio: https://github.com/t3rm1nus/HRM
Rama comunicada: master
Commit comunicado: b0b085a — Implementa aislamiento de perfiles PAPER y persistencia
Estado: AUDITORÍA PARCIAL — FASE 0 NO CERRADA


1. Resumen ejecutivo
HRM dispone de una base de código amplia, organizada en módulos Python y una interfaz web React/TypeScript. Su arquitectura contempla procesamiento de datos de mercado, análisis estratégico y táctico, gestión de órdenes, cartera, persistencia, backtesting y modelos de inteligencia artificial.

Durante la auditoría hemos inspeccionado parcialmente el arranque, la configuración, los perfiles PAPER, la persistencia, los clientes de exchange simulado, la gestión de cartera, los gestores de órdenes y los mecanismos de limpieza.

Hemos encontrado varios problemas técnicos relevantes, especialmente en la restauración de balances PAPER y en la migración de datos.

Sin embargo, todavía no hemos podido inspeccionar íntegramente el repositorio ni completar las pruebas de todas las áreas solicitadas.

Por tanto, no sería correcto declarar HRM completamente auditado ni dar por finalizada la Fase 0.
1.1. Estado general
Área	Estado
Arquitectura general	Parcialmente reconstruida
Arranque y bootstrap	Inspeccionados parcialmente
Arquitectura L1, L2 y L3	Parcial
Perfiles PAPER	Implementación inspeccionada; aislamiento no certificado
Exchange simulado	Defectos detectados
PortfolioManager	Inspección parcial
Persistencia y migración	Riesgos importantes detectados
Gestores de órdenes	Inspección parcial
MessageBus	Pendiente
Coordinación de estado	Pendiente
Validadores y controles de riesgo	Pendiente
ExecutionGateway	Pendiente
Fuentes de datos de mercado	Pendiente
API REST	Pendiente
WebSockets	Pendiente
Frontend	Pendiente
Backtesting	Pendiente
Tests	Identificados parcialmente, no ejecutados
Dependencias	Pendiente de auditoría completa
Seguridad	Riesgos identificados, revisión incompleta

1.2. Principales problemas detectados
1. force_reset() no actualiza los balances efectivos del exchange simulado.
2. La migración de datos PAPER puede marcarse como completada aunque alguna copia haya fallado.
3. Existen dos implementaciones diferentes de SimulatedExchangeClient.
4. El reinicio de memoria no reinicia correctamente el singleton de PortfolioManager.
5. La limpieza global puede informar de éxito aunque alguna operación interna haya fallado.
6. La configuración del modo está repartida entre varios módulos y variables de entorno.
7. El cliente Binance puede quedar en un estado contradictorio entre mode y paper_mode.
8. No se ha demostrado que la restauración de cartera y el exchange simulado mantengan balances coherentes.

2. Alcance de la auditoría
Se han utilizado:
- README del proyecto.
- Fragmentos de código compartidos desde C:\proyectos\HRM.
- Resultados de inspecciones mediante PowerShell.
- Tests identificados durante la revisión.
- Información del commit y estado Git comunicados anteriormente.
Es importante distinguir cuatro situaciones:
- Confirmado: aparece directamente en el código inspeccionado.
- Riesgo: el diseño permite un comportamiento problemático, pero no se ha reproducido dinámicamente.
- No verificado: no existe suficiente evidencia para emitir una conclusión.
- Probado: existe una ejecución identificable con resultado documentado.
No se ha ejecutado HRM durante esta auditoría.

Tampoco se han modificado archivos, estrategias, indicadores, señales, parámetros de riesgo ni configuración PAPER/LIVE.

3. Arquitectura real reconstruida
3.1. Estructura general
La arquitectura identificada es modular y contempla diferentes niveles de procesamiento.
Nivel	Responsabilidad
L1 — Operacional	Gestión de órdenes, ejecución, validaciones y sincronización de cartera
L2 — Táctico	Análisis de mercado y generación de señales
L3 — Estratégico	Análisis de contexto, régimen y decisiones estratégicas
L4 — Meta	Coordinación y procesamiento de decisiones superiores

No se ha demostrado todavía que todos los niveles estén conectados entre sí en el flujo real de ejecución.

La arquitectura conceptual no equivale a una integración funcional probada.
3.2. Directorios identificados
Directorio	Responsabilidad
main.py	Punto de entrada principal
core/	Cartera, configuración, estado y utilidades
system/	Bootstrap, preparación de sesión y limpieza
comms/	Comunicación, configuración y contratos
l1_operational/	Gestión operacional y ejecución
l2_tactic/	Análisis táctico
l3_strategy/	Análisis estratégico
l4_meta/	Coordinación de nivel superior
storage/	Persistencia y registro de operaciones
api/	API REST y servicios
frontend/	Aplicación web
backtesting/	Pruebas históricas
ml_training/	Entrenamiento de modelos
tests/	Pruebas automatizadas
persistent_state/	Estado persistente
data/paper/	Datos de los perfiles PAPER

La existencia de un directorio no demuestra que sus componentes estén conectados al runtime.

4. Punto de entrada y flujo de ejecución
4.1. Punto de entrada
El flujo principal inspeccionado comienza en:

main.py

Este archivo importa SystemBootstrap desde:

system/bootstrap.py

También existe otro archivo:

bootstrap.py

Este contiene una implementación diferente, con otra clase (HRMBootstrap).

Es importante no confundir ambos mecanismos de inicialización.
4.2. Flujo reconstruido
El flujo observado en main.py es:
1. Importación de configuración y componentes.
2. Lectura de variables de entorno.
3. Determinación del modo de sesión.
4. Registro del modo.
5. Ejecución de preparar_sesion().
6. Inicialización de SystemBootstrap.
7. Obtención del contexto inicial.
8. Inicialización manual de componentes.
9. Creación de BinanceClient.
10. Inicialización de PortfolioManager.
11. Inicialización del exchange simulado.
12. Creación de OrderManager.
13. Inyección del cliente simulado.
14. Sincronización de cartera.
Hallazgo: bootstrap parcialmente redundante
SystemBootstrap.initialize_system() devuelve un contexto cuyo diccionario de componentes está inicialmente vacío.

Posteriormente, main.py inicializa manualmente los componentes que necesita.

Esto significa que el bootstrap inspeccionado no constituye por sí solo una inicialización completa de la arquitectura.

5. Arquitectura PAPER
Esta es una de las partes más importantes de la auditoría.
5.1. Perfiles existentes
Archivo:

storage/perfil_datos.py

Se identifican dos perfiles:
- persistente
- limpia
La estructura física observada es:
data/
└── paper/
    ├── persistente/
    │   ├── portfolio_state.json
    │   └── paper_trades/
    │       ├── paper_trades.json
    │       ├── paper_trades.csv
    │       └── summary.json
    │
    └── limpia/
        ├── portfolio_state.json
        └── paper_trades/
            ├── paper_trades.json
            ├── paper_trades.csv
            └── summary.json


La estructura demuestra que existen directorios separados.

No demuestra por sí sola que el aislamiento esté garantizado en todos los componentes.
5.2. Funciones identificadas
En storage/perfil_datos.py:
- configurar_perfil_datos()
- obtener_directorio_paper()
- obtener_directorio_operaciones()
- obtener_ruta_cartera()
- obtener_ruta_log_cartera()
- migrar_datos_legacy_a_persistente()
- limpiar_perfil_paper()
- solicitar_reset_persistente()
- hay_reset_persistente_pendiente()
- consumir_reset_persistente()
El módulo valida los perfiles permitidos y construye rutas diferentes.
5.3. Preparación de sesión
Archivo:

system/preparar_sesion.py

Comportamiento inspeccionado:
- Lee el modo configurado.
- En sesiones no LIVE ejecuta la migración legacy.
- Selecciona el perfil PAPER.
- Si el perfil es LIMPIA, limpia ese directorio.
- Si el perfil es PERSISTENTE y hay un reset pendiente, limpia el perfil y consume el marcador.
- Si se arranca en LIVE, no consume el marcador de reset PAPER.
La migración se invoca antes de seleccionar el perfil, incluso cuando la sesión solicitada es LIMPIA.

Esto debe tenerse en cuenta en las pruebas de aislamiento.

6. Persistencia y migración
6.1. Migración legacy
Archivo:

storage/perfil_datos.py

La función migrar_datos_legacy_a_persistente():
1. Comprueba si existe el marcador de migración.
2. Si existe, no repite la migración.
3. Copia datos legacy de cartera.
4. Copia operaciones.
5. Copia el log de cartera.
6. Registra el resultado en un marcador JSON.
Defecto confirmado
 Marcador de migración prematuro
Las funciones auxiliares de copia capturan errores de E/S y pueden devolver cero.

El proceso puede escribir el marcador de migración aunque una o varias copias hayan fallado.

Como después el marcador impide repetir la migración, una migración incompleta puede quedarse bloqueada.

El marcador observado contiene:
{
  "version": 1,
  "migrado_en": "2026-11-05T17:01:18.217494+00:00",
  "archivos_copiados": {
    "cartera": 1,
    "operaciones": 2,
    "log_cartera": 0
  }
}


Esto demuestra que se registraron esos conteos, pero no que todos los datos estén íntegros.
6.2. Logger PAPER
Archivo:

storage/paper_trade_logger.py

Se ha comprobado que:
- Mantiene una caché de loggers.
- La clave de caché utiliza la ruta resuelta del directorio.
- Los loggers de distintos perfiles pueden tener instancias separadas.
- Puede cargar datos existentes.
- Dispone de una opción para limpiar archivos al inicializar.
Las llamadas de producción localizadas utilizan el logger sin activar la limpieza.

Es un diseño razonable para separar perfiles, pero falta probarlo en escenarios de reinicio y cambio de perfil.

7. Exchange simulado
7.1. Duplicidad de implementaciones
Se han encontrado dos clases diferentes.
Archivo	Características
l1_operational/simulated_exchange_client.py	Cliente síncrono, get_instance(), get_balances() y execute_order() síncrono
core/simulated_exchange_client.py	Implementación distinta con métodos asíncronos

El flujo operacional inspeccionado utiliza la implementación de l1_operational.

Esta duplicidad introduce riesgo de:
- Imports incorrectos.
- Interfaces incompatibles.
- Diferencias de comportamiento.
- Tests que validen una implementación distinta de la utilizada en el runtime.
7.2. Defecto crítico en force_reset()
Archivo:

l1_operational/simulated_exchange_client.py

El método inspeccionado realiza:
def force_reset(self, initial_balances=None):    if initial_balances:        self.initial_balances = initial_balances.copy()        self.trades.clear()        self.order_history.clear()        self._trade_id_counter = 1    else:        raise RuntimeError(...)



El problema es que no actualiza self.balances.

Esto significa que puede modificar los balances iniciales registrados, pero mantener los balances efectivos anteriores.
Consecuencias
- La restauración de cartera puede no trasladarse al exchange simulado.
- La sincronización posterior puede recuperar balances antiguos.
- El estado persistido y el estado efectivo pueden divergir.
- El historial en memoria se limpia durante el reset.
Este defecto es especialmente importante porque existen llamadas a force_reset() desde:
- main.py
- core/portfolio_manager.py
- system/bootstrap.py
No se ha ejecutado una prueba dinámica para reproducirlo, pero el defecto se observa directamente en el método.
7.3. Ejecución de órdenes
Archivo:

l1_operational/order_executors.py

Se ha observado este flujo:
1. Consulta de balances.
2. Cálculo de NAV.
3. Ejecución de orden simulada.
4. Consulta de balances posteriores.
5. Sincronización con PortfolioManager.
6. Verificación de valoración.
La llamada a execute_order() sin await es coherente con la implementación síncrona L1.

No se puede trasladar esa conclusión a la implementación asíncrona de core/.

Falta comprobar que los errores de sincronización detienen correctamente el procesamiento posterior.

8. PortfolioManager
Archivo:

core/portfolio_manager.py

Se han inspeccionado parcialmente:
- initialize_async()
- _sincronizar_cartera_con_exchange_simulado()
- _init_portfolio_from_simulated_client_async()
- sync_from_exchange_async()
- load_from_json()
Comportamiento identificado:
- Intenta restaurar la cartera desde JSON.
- Registra si se ha restaurado el estado PAPER.
- Intenta sincronizar los balances restaurados con el exchange simulado.
- Puede inicializar la cartera desde el cliente simulado.
- Actualiza balances y estado de sincronización.
Riesgos detectados
1. La restauración utiliza force_reset(), que no actualiza self.balances.
2. Falta verificar que todos los callers traten un resultado False de sincronización como un fallo bloqueante.
3. Algunos caminos inicializan total y peak_value utilizando initial_balance; debe comprobarse que el NAV se recalcula correctamente.
4. El ciclo de vida del singleton necesita revisión.
5. No se ha auditado toda la gestión de posiciones, órdenes pendientes, comisiones y persistencia concurrente.
El método de sincronización también rechaza el escenario en que BTC, ETH y USDT sean simultáneamente cero.

Esta guarda puede ser deliberada para proteger contra pérdida de estado; no se clasifica como defecto sin conocer las reglas de negocio completas.

9. Gestores de órdenes
Archivos:
- l1_operational/order_manager.py
- l1_operational/order_executors.py
Se ha confirmado que OrderManager inyecta el cliente simulado en OrderExecutors.

También se ha observado que OrderExecutors sincroniza posteriormente la cartera.

No se ha revisado completamente:
- Idempotencia.
- Reintentos.
- Cancelaciones.
- Órdenes parciales.
- Reconciliación.
- Duplicidad de órdenes.
- Estados de ejecución.
- Persistencia transaccional.
- Gestión de errores.
Por tanto, el flujo está parcialmente conectado, pero no certificado de extremo a extremo.

10. Bootstrap y limpieza
10.1. system/bootstrap.py
Se ha inspeccionado bootstrap_simulated_exchange(config).

La función:
- Obtiene balances de configuración.
- Recupera la instancia singleton.
- Comprueba si acaba de inicializarse.
- Puede conservar balances de una instancia anterior.
- Compara el saldo USDT con el valor esperado.
- Registra discrepancias.
Riesgo
Si el balance no coincide con la configuración, el código puede registrar un error y continuar devolviendo el cliente.

Esto puede permitir que el sistema continúe con un estado distinto del esperado.
10.2. system/system_cleanup.py
Se han inspeccionado:
- filesystem_cleanup()
- perform_full_cleanup()
- memory_reset()
- force_paper_mode()
- async_context_reset()
Defecto 1: singleton de PortfolioManager
memory_reset() reinicia StateCoordinator, pero no llama a:
PortfolioManager.reset_instance()



Eliminar el módulo de sys.modules no garantiza que desaparezca una instancia singleton ya creada.
Defecto 2: resultado global de limpieza
perform_full_cleanup() puede devolver éxito aunque alguna suboperación devuelva:
{"success": False}



si no se produce una excepción.
Defecto 3: argumento mode
El argumento mode no se utiliza en el fragmento inspeccionado.
Defecto 4: variable de entorno
force_paper_mode() establece:
os.environ["HRM_MODE"] = "paper"



Sin embargo, core/config.py utiliza HRM_PAPER_MODE para esa configuración.

No se ha demostrado que ambas variables sean equivalentes.
Riesgo 5: limpieza temporal
_clean_temp_data() elimina recursivamente archivos *.tmp y *_temp.* dentro de data/.

La regla tiene un alcance amplio y debe revisarse para evitar eliminar temporales pertenecientes a otros procesos.

11. Configuración y modos
Archivos:
- core/config.py
- comms/config.py
- main.py
Se han identificado diferentes fuentes de configuración:
- PAPER_MODE
- HRM_PAPER_MODE
- BINANCE_MODE
- Modo forzado global.
- Configuración común importada desde comms.config.
Problemas
core/config.py admite los modos:
- live
- testnet
- backtest
- simulated
El literal paper no aparece entre los modos admitidos por EnvironmentConfig.

Además, comms/config.py crea una configuración global al importar el módulo.

Si se modifica el entorno después de importar el módulo, la configuración global puede conservar valores antiguos.
BinanceClient
Archivo:

l1_operational/binance_client.py

Se ha detectado una configuración contradictoria:

Si se solicita mode="live" y faltan las claves API, el constructor puede establecer:
self.paper_mode = True



pero conservar:
self.mode = "live"



y preparar el cliente CCXT con endpoints de mainnet.

Esto no demuestra que se envíen órdenes, pero es una configuración incoherente que debe corregirse mediante un diseño fail-closed.

PAPER debe seguir siendo el modo predeterminado. LIVE no debe habilitarse.

12. MessageBus y coordinación de estado
Áreas identificadas:
- comms/
- system/state_coordinator.py
- core/
No se ha inspeccionado suficiente código para certificar:
- Topics.
- Contratos de mensajes.
- Orden de entrega.
- Reintentos.
- Backpressure.
- Correlación de ciclos.
- Idempotencia.
- Recuperación de desconexiones.
- Gestión de consumidores.
- Comunicación entre procesos.
Estado: pendiente de auditoría.

13. Fuentes de datos de mercado
Se han identificado referencias a:
- l1_operational/binance_client.py
- l1_operational/data_feed.py
- l1_operational/realtime_loader.py
- l3_strategy/data_provider.py
No se ha reconstruido completamente el recorrido de los datos desde el proveedor hasta L1, L2 y L3.

Pendiente comprobar:
- Fuente efectiva en runtime.
- Frecuencia y granularidad.
- Velas incompletas.
- Datos duplicados.
- Huecos de mercado.
- Timestamps.
- Reconexiones.
- Caché.
- Datos sintéticos.
- Validación de precios.
Estado: no certificado.

14. Validadores, riesgo y ExecutionGateway
El proyecto menciona componentes como validadores de órdenes, gestores de riesgo y ExecutionGateway.

No se ha recibido suficiente código para comprobar:
- Si todos los caminos de ejecución pasan por los validadores.
- Si las restricciones de riesgo son obligatorias.
- Si existen rutas alternativas.
- Si se controlan órdenes duplicadas.
- Si se rechazan cantidades inválidas.
- Si los fallos bloquean la ejecución.
- Si ExecutionGateway es realmente el punto único de ejecución.
Estado: pendiente y bloqueante para certificar la ejecución PAPER.

15. API REST y WebSockets
Se han identificado rutas API y servicios relacionados con sesiones y operaciones.

No se ha revisado íntegramente:
- Inventario de endpoints.
- Contratos de entrada y salida.
- Validación de datos.
- Autenticación.
- Autorización.
- CORS.
- Protección de acciones destructivas.
- Gestión de errores.
- WebSockets.
- Reconexión.
- Cierre de conexiones.
- Integración frontend-backend.
Estado: no certificado.

16. Frontend
La documentación identifica React, TypeScript y Vite.

No se ha inspeccionado completamente el código del frontend ni se han ejecutado sus comandos de validación.

Pendiente:
1. Revisar package.json.
2. Revisar package-lock.json.
3. Ejecutar npm ci.
4. Ejecutar TypeScript typecheck.
5. Ejecutar tests.
6. Ejecutar build de producción.
7. Revisar contratos API.
8. Revisar WebSockets.
9. Comprobar gestión de errores y estados de conexión.
Estado: no certificado.

17. Backtesting y modelos ML/RL
Se han identificado las áreas:
- backtesting/
- ml_training/
- l2_tactic/
- l3_strategy/
No se ha ejecutado ningún backtest reproducible durante esta auditoría.

Pendiente revisar:
- Calidad y procedencia de los datos.
- Separación temporal de entrenamiento y evaluación.
- Look-ahead bias.
- Data leakage.
- Comisiones.
- Spread.
- Slippage.
- Liquidez.
- Órdenes parciales.
- Reproducibilidad.
- Versiones de modelos.
- Artefactos disponibles.
- Dependencias opcionales.
- Conexión efectiva al runtime.
La existencia de scripts de entrenamiento no demuestra que los modelos estén conectados al sistema.

Estado: no certificado.

18. Tests
Se han identificado al menos los siguientes archivos:
- tests/test_preparar_sesion.py
- tests/test_perfil_datos.py
- tests/api/test_rutas_operaciones.py
Se han inspeccionado pruebas relacionadas con:
- Reset PERSISTENTE pendiente.
- Conservación del reset al arrancar en LIMPIA.
- No consumo del reset en LIVE.
- Conservación del historial PAPER durante limpieza técnica.
Sin embargo:
No ejecutados durante esta auditoría
No se ha ejecutado la suite completa de pytest.

Tampoco se han ejecutado:
- Tests de integración completos.
- Tests de frontend.
- Typecheck.
- Lint.
- Compilación Python.
- Build frontend.
- Arranque del backend.
- Apagado controlado.
No podemos afirmar que los tests pasen.

19. Dependencias y entorno
No se ha completado el inventario de imports ni la auditoría de dependencias.

La Fase 1 deberá revisar:
1. Dependencias directas.
2. Dependencias transitivas.
3. Versiones incompatibles.
4. Dependencias opcionales de ML/RL/sentimiento.
5. Manifiestos duplicados.
6. Versión de Python.
7. Variables de entorno.
8. Rutas absolutas.
9. .env.example.
10. Reproducibilidad de instalación.
No se deben actualizar paquetes indiscriminadamente.

20. Estado Git y congelación
En una inspección anterior se comunicó:

Commit:

b0b085a Implementa aislamiento de perfiles PAPER y persistencia

Rama:

master

Se comunicó que la rama local y origin estaban sincronizadas en ese momento.

También se observaron cambios locales no staged:
.hrm_runtime/backend.pid
.hrm_runtime/frontend.pid
core/__pycache__/portfolio_manager.cpython-310.pyc
hacienda/ganancias_realizadas.csv
hacienda/operaciones.csv
hacienda/posiciones_fifo.json
persistent_state/modo_sesion.json
persistent_state/portfolio_state_live.json


Y archivos no versionados:
HRM-master (1).zip
hacienda/Untitled
persistent_state/migracion_paper_perfiles_v1.json


Estos elementos no deben eliminarse ni sobrescribirse automáticamente.

Los CSV y JSON pueden contener información que el usuario necesita conservar.

El estado Git debe volver a comprobarse antes de autorizar cualquier modificación.

21. Matriz de hallazgos
ID	Prioridad	Hallazgo	Archivo	Clasificación
F0-01	Crítica	force_reset() no actualiza balances efectivos	l1_operational/simulated_exchange_client.py	Defecto confirmado
F0-02	Alta	Migración puede marcarse completada con copias fallidas	storage/perfil_datos.py	Defecto/riesgo confirmado
F0-03	Alta	Dos implementaciones distintas de exchange simulado	core/ y l1_operational/	Duplicidad confirmada
F0-04	Alta	No se reinicia el singleton de PortfolioManager	system/system_cleanup.py	Defecto confirmado
F0-05	Alta	Limpieza global puede ocultar fallos internos	system/system_cleanup.py	Riesgo confirmado
F0-06	Alta	Configuración de modo repartida	main.py, core/config.py, comms/config.py	Riesgo confirmado
F0-07	Alta	Estado contradictorio entre mode y paper_mode	l1_operational/binance_client.py	Defecto de configuración
F0-08	Alta	Desajuste de balances puede registrarse sin detener bootstrap	system/bootstrap.py	Riesgo confirmado
F0-09	Media	Configuración global creada durante importación	comms/config.py	Riesgo confirmado
F0-10	Media	Limpieza temporal recursiva bajo data/	system/system_cleanup.py	Comportamiento confirmado
F0-11	Media	NAV inicial puede depender de initial_balance	core/portfolio_manager.py	Requiere validación
F0-12	Media	No está confirmado que todos los callers detengan el flujo ante fallo de sincronización	core/portfolio_manager.py, l1_operational/order_executors.py	Pendiente
F0-13	Bloqueante	ExecutionGateway y controles de riesgo sin auditoría completa	l1_operational/, módulos de riesgo	No verificado
F0-14	Bloqueante	API, WebSockets y frontend sin auditoría completa	api/, frontend/	No verificado
F0-15	Bloqueante	Backtesting y ML/RL sin auditoría completa	backtesting/, ml_training/	No verificado
F0-16	Bloqueante	No hay validación integral ejecutada	Todo el proyecto	No ejecutado


22. Orden recomendado para completar la Fase 0
Antes de modificar código:
1. Obtener una copia íntegra del repositorio local exacto.
2. Confirmar commit, rama y git status.
3. Conservar todos los cambios locales y datos existentes.
4. Completar inventario de puntos de entrada.
5. Diferenciar módulos activos, alternativos y legacy.
6. Trazar L1, L2, L3 y L4 desde el punto de entrada real.
7. Auditar MessageBus y StateCoordinator.
8. Auditar fuentes de datos.
9. Auditar validadores, controles de riesgo y ExecutionGateway.
10. Completar auditoría de gestores de órdenes.
11. Completar auditoría PAPER PERSISTENTE/LIMPIA.
12. Revisar API REST y WebSockets.
13. Revisar contratos frontend-backend.
14. Auditar backtesting y modelos.
15. Revisar dependencias y configuración.
16. Relacionar tests con componentes y requisitos.
17. Documentar todos los riesgos.
18. Actualizar el informe con evidencias completas.
La Fase 0 solo se podrá declarar terminada cuando todas las áreas estén inspeccionadas o explícitamente excluidas con justificación.

23. Orden recomendado para la Fase 1
Una vez terminada y autorizada la Fase 0:
1. Corregir la restauración de balances PAPER.
2. Corregir el proceso de migración y su marcador.
3. Resolver la duplicidad del exchange simulado.
4. Corregir el ciclo de vida de los singletons.
5. Corregir la propagación de errores de limpieza.
6. Unificar la configuración de modo.
7. Mantener PAPER como modo predeterminado.
8. Consolidar dependencias.
9. Separar dependencias obligatorias y opcionales.
10. Corregir imports y errores de sintaxis.
11. Crear o actualizar .env.example.
12. Eliminar rutas absolutas innecesarias.
13. Añadir tests de regresión.
14. Ejecutar tests Python.
15. Ejecutar npm ci.
16. Ejecutar typecheck, tests y build frontend.
17. Validar arranque y apagado controlados.
18. Generar INFORME_FASE_1.md.
No se deben modificar estrategias, señales, indicadores ni parámetros de riesgo durante esta fase.

No se debe habilitar LIVE.

24. Conclusión final
La auditoría ha detectado problemas concretos que justifican no avanzar todavía a la estabilización.

El más importante es la restauración de balances del exchange simulado: el método force_reset() no actualiza los balances efectivos, lo que puede provocar divergencias entre cartera persistida y exchange PAPER.

También se han identificado problemas en la migración, el reinicio de singletons y la configuración de modos.

El proyecto tiene una base de código significativa, pero no se ha demostrado todavía que todos sus componentes estén conectados ni que el sistema completo funcione correctamente.
Dictamen definitivo de esta revisión
Fase	Estado
Fase 0 — Auditoría y congelación	Parcial. Pendiente de completar
Fase 1 — Estabilización técnica	No iniciada

No recomiendo iniciar todavía la Fase 1.

Primero debemos completar la inspección del repositorio exacto, terminar las áreas pendientes y aprobar el informe definitivo.
Garantías de esta auditoría
- No se ha arrancado HRM.
- No se han modificado archivos del proyecto.
- No se han eliminado datos.
- No se ha cambiado la configuración PAPER/LIVE.
- No se han modificado estrategias, señales, indicadores ni parámetros de riesgo.
- No se han declarado pruebas que no se hayan ejecutado.
La siguiente decisión debe ser completar la auditoría de las áreas pendientes y, después, solicitar autorización para iniciar la Fase 1.



INFORME DE AUDITORÍA — HRM
Revisión estática del flujo principal completada
Alcance: código fuente del repositorio, sin arrancar HRM ni modificar archivos.
1. Resumen ejecutivo
HRM tiene una base funcional considerable: gestión de cartera, exchange simulado, generación de señales, dimensionamiento de órdenes, persistencia por perfiles PAPER y API.

Sin embargo, he encontrado problemas de integración que impiden considerar el sistema validado.

Los más importantes son:
1. El verificador de señales puede rechazar una señal, pero el pipeline la sigue aceptando.
2. Los controles de riesgo y stop-loss de la ruta táctica principal contienen implementaciones incompletas.
3. El supuesto override de OrderFlow no llega al pipeline de ejecución.
4. La comprobación de balances de main.py utiliza un atributo que no corresponde al PortfolioManager actual.
5. El gestor de datos puede considerar válida una reparación que devuelve un DataFrame vacío.
6. Hay componentes de ejecución y comunicación que existen en el repositorio, pero no están conectados al arranque real.
No son problemas cosméticos. Afectan a la integridad de los datos, a la generación de órdenes y a la capacidad de confiar en las protecciones del sistema.
2. Hallazgos prioritarios
Prioridad	Componente	Hallazgo
Crítica	SignalVerifier	Se ignora el resultado de la validación de señales.
Crítica	L2 / RiskOverlay	Los controles de riesgo y stop-loss no están implementados en la ruta activa.
Alta	OrderFlow	El override se calcula, pero no se consume.
Alta	PortfolioManager	La detección de balances cero no está conectada correctamente.
Alta	MarketDataManager	Puede aceptar datos vacíos como reparados.
Alta	Persistencia	La sincronización puede fallar sin detener adecuadamente el flujo.
Alta	Configuración	La propagación del modo entre componentes es inconsistente.
Media	ExecutionGateway	Tiene errores de importación y construcción.
Media	MessageBus	No está integrado en el flujo principal de main.py.

F0-01. El verificador de señales no bloquea señales rechazadas
Crítico
En system/trading_pipeline_manager.py, el método _validate_signals hace lo siguiente:
- Envía la señal al verificador.
- Espera el resultado.
- Añade la señal a la lista de válidas sin comprobar result.is_valid.
El problema está confirmado al contrastarlo con core/incremental_signal_verifier.py, donde submit_signal_for_verification() devuelve un objeto SignalVerificationResult con el campo is_valid.

Consecuencia: una señal rechazada por el verificador puede llegar igualmente a la generación de órdenes.
F0-02. Las protecciones de riesgo L2 están incompletas
Crítico
En l2_tactic/tactical_signal_processor.py he comprobado que:
- _apply_risk_adjustment() devuelve la señal sin modificar.
- _apply_stop_loss_calculation() devuelve la señal sin modificar.
- _create_l1_composite_signal() devuelve None.
- RiskOverlay se instancia, pero no he encontrado su aplicación en el camino principal de procesamiento táctico.
Además, RiskControlManager aparece en componentes alternativos, pero no he encontrado su integración efectiva en el flujo principal de main.py.

Consecuencia: no podemos considerar implementados ni verificados los controles de riesgo que describe la arquitectura.
F0-03. OrderFlow no está conectado a la ejecución
Crítico
En main.py se generan señales OrderFlow y se establece:
state["orderflow_signals"] = ...state["orderflow_override"] = True



Pero TradingPipelineManager no consume esos campos para construir las señales que llegan al OrderManager.

Consecuencia: el mensaje de log indica que OrderFlow puede hacer un override, pero el código inspeccionado no demuestra que realmente lo haga.
F0-04. Comprobación incorrecta de balances
Alta
En main.py se comprueba:
hasattr(portfolio_manager, "client")



Sin embargo, el core.portfolio_manager.PortfolioManager actual almacena el cliente mediante exchange_client.

No he encontrado una asignación que proporcione el atributo client a esa instancia.

Por tanto, la comprobación de balances del bucle principal puede quedar completamente omitida.

Hay otro problema dentro de ese bloque: si se activara su recuperación de último recurso, utiliza balances fijos de BTC, ETH y USDT.

Eso no es una recuperación válida del estado: podría introducir capital ficticio en la simulación. Ante una pérdida de estado, HRM debe detenerse y exigir una recuperación verificable, nunca inventar balances.
F0-05. Validación de datos de mercado
Alta
En system/market_data_manager.py, _repair_symbol_data() puede devolver un DataFrame vacío.

Después, _validate_data() acepta cualquier resultado distinto de None como reparación válida, sin volver a validarlo.

También puede incrementar el contador de éxitos aunque el DataFrame esté vacío.

Consecuencia: el gestor puede considerar que ha conseguido datos válidos cuando realmente no dispone de velas utilizables.
F0-06. Sincronización de cartera
Alta
En core/portfolio_manager.py, sync_from_exchange_async() devuelve False cuando BTC, ETH y USDT están todos a cero.

Es una protección comprensible frente a la pérdida de estado, pero he comprobado que algunos llamadores no comprueban adecuadamente el resultado.

Riesgo: el sistema puede continuar con información anterior de cartera cuando la sincronización ha fallado.
F0-07. Configuración y modos
He encontrado una inconsistencia entre el modo calculado por main.py, el BinanceClient que se construye sin pasar explícitamente ese modo y el exchange simulado que se inicializa desde bootstrap_simulated_exchange().

Además, OrderExecutors rechaza la ejecución cuando no está en modo PAPER: el camino de ejecución real no está implementado en esa clase.

Por tanto, no se puede considerar que la arquitectura de modos esté validada ni que todos los componentes compartan un estado operativo coherente.

La configuración debe permanecer cerrada en PAPER mientras esto no esté resuelto.
F0-08. ExecutionGateway
En execution_gateway.py:
- Importa OrderValidator, pero el módulo l1_operational/order_validators.py define OrderValidators.
- Construye RiskGuard pasándole portfolio_manager, aunque el constructor inspeccionado no acepta ese argumento.
El componente tampoco aparece integrado en el camino activo de main.py.

Es una implementación alternativa que no puede considerarse operativa.
F0-09. Arquitectura de comunicación
El proyecto contiene MessageBus, L2BusAdapter y otros adaptadores.

Sin embargo, el flujo principal se ejecuta mediante llamadas directas desde TradingPipelineManager.

También he comprobado que SystemBootstrap.initialize_system() devuelve un contexto vacío: no inicializa los componentes que anuncia su documentación. main.py termina construyendo buena parte de ellos manualmente.

Esto no significa que el sistema no pueda funcionar, pero sí que la arquitectura documentada y el arranque efectivo no coinciden.
3. Persistencia PAPER: avances confirmados
Esta parte ha mejorado respecto a la revisión anterior.
Mejoras verificadas en código
He comprobado que:
- storage/perfil_datos.py separa los directorios persistente y limpia.
- PaperTradeLogger mantiene instancias independientes por directorio.
- PortfolioManager.initialize_async() intenta restaurar primero la cartera PAPER.
- El exchange simulado dispone ahora de un force_reset() que actualiza tanto balances como initial_balances.
- system/system_cleanup.py ya llama a PortfolioManager.reset_instance().
- El endpoint de reinicio diferido exige confirmación exacta y comprueba que el modo y perfil sean los adecuados.
- servicio_estado_vivo.normalizar_modo() traduce correctamente simulated a paper.
También existen pruebas focalizadas para perfiles, migración, persistencia y API.

Corrijo expresamente dos observaciones del informe anterior: el método force_reset() sí actualiza los balances en la versión actual y el reinicio del singleton PortfolioManager sí está implementado.
Riesgo pendiente de migración
La migración legacy crea el marcador de migración incluso cuando alguna copia falla.

Si ocurre un error de escritura, el marcador puede impedir que se vuelva a intentar automáticamente la migración.

Esto necesita una política de migración transaccional o, como mínimo, un registro fiable de migraciones parciales.
4. API y frontend
También hay avances que no aparecían en el informe anterior.

He comprobado que existen:
- /health
- /status
- /portfolio
- /version
- /mercado/estado
- /mercado/simbolos
- /mercado/velas/{simbolo}
- /sesion/modo
- /sesion/reiniciar-datos
- WebSocket /ws
La API de mercado está conectada a MarketDataManager mediante ServicioMercado, y existen pruebas para sus rutas y servicios.

El servicio de estado también registra la cartera y publica eventos de estado y cartera.

Lo que sigue incompleto: no he encontrado publicación efectiva de eventos de órdenes y señales al WebSocket. La conexión existe, pero no equivale a disponer de un flujo completo de eventos de trading.

Respecto al frontend, el informe de estabilización incluido en el repositorio, fechado el 3 de noviembre, indica que:
- Se revisaron sintácticamente 348 archivos Python sin errores.
- Se intentó ejecutar TypeScript.
- La instalación con npm ci no pudo completarse por timeout.
- Persistían errores de tipado y compatibilidad en componentes gráficos y módulos heredados.
No he repetido esas pruebas en este entorno, así que no las presento como resultados propios ni como validación del commit actual.
5. Pruebas y validación
Comprobación	Estado
Inspección estática de arranque	Revisada
Separación de perfiles PAPER	Revisada
Restauración de cartera	Revisada
Exchange simulado	Revisado
Gestores de órdenes	Revisados
Pipeline L1-L2-L3	Revisado
Validación de señales	Revisada
Controles de riesgo	Revisados
Datos de mercado	Revisados
API y WebSocket	Revisados
Pruebas unitarias	Existen, no ejecutadas
Pruebas de integración	No ejecutadas
Build de frontend	No certificado
Backtesting completo	No certificado
Ciclo completo de trading PAPER	No ejecutado

El estado combinado de comprobaciones del commit consultado no muestra resultados de CI.
6. Correcciones respecto al informe anterior
Estas son las principales actualizaciones:
Elemento	Estado actual
force_reset() no actualizaba balances	Corregido en el código actual
memory_reset() no reiniciaba PortfolioManager	Corregido
API de mercado inexistente	Ya existen rutas y servicio
Estado de runtime desconectado	Ahora existe registro de estado y ciclo
Cartera desconectada de la API	Ahora existe registro compartido
Eventos WebSocket inexistentes	Hay eventos de mercado, estado y cartera; faltan otros

El informe de estabilización del 3 de noviembre y el README todavía contienen afirmaciones que deben actualizarse para reflejar el código actual.

7. Estado de las fases
Fase 0 — Auditoría
Revisión estática completada
Camino crítico

Se ha revisado el flujo principal y documentado sus problemas. No equivale a certificar todos los módulos heredados ni el funcionamiento en ejecución.

Fase 1 — Estabilización
Parcial
Pendiente de completar

Ya existe trabajo previo de estabilización incorporado al repositorio, pero siguen pendientes las correcciones funcionales, el build y las pruebas de integración.

8. Orden recomendado para continuar
No empezaría por el frontend ni por añadir más funcionalidades. Primero hay que garantizar la integridad del núcleo.
1. Corregir la validación de señales y hacer que los rechazos bloqueen realmente las órdenes.
2. Completar los controles de riesgo y las protecciones de ejecución.
3. Corregir la sincronización de cartera y la detección de pérdida de estado.
4. Eliminar las recuperaciones mediante balances inventados.
5. Corregir la validación y reparación de datos de mercado.
6. Conectar correctamente OrderFlow al pipeline o retirar el falso override.
7. Unificar la inicialización y propagación del modo PAPER.
8. Resolver los errores de los componentes alternativos y legacy.
9. Completar el frontend y su compilación.
10. Ejecutar pruebas unitarias y de integración aisladas en PAPER.
Conclusión: la Fase 0 queda cerrada como revisión estática del camino crítico, con las limitaciones indicadas. HRM todavía no está validado funcionalmente y la Fase 1 no está terminada. No he modificado ningún archivo ni arrancado el sistema.