# Informe de correcciones del frontend HRM

## Resultado de tipado
- Errores iniciales reproducidos: 48 errores TypeScript en 19 archivos.
- Comando ejecutado: `npm run typecheck`.
- Resultado tras las correcciones: correcto, sin errores.

## Correcciones aplicadas
- Recharts: sustitución de `activePayload` por el índice activo y acceso validado a los datos.
- Gráficas: valores de color y etiquetas con fallback seguro.
- Modos: paso explícito del modo actual a `BadgeModo` y comparación con el enum `ModoAplicacion`.
- Validación de velas: entrada `unknown[]`, parseo con Zod y omisión de propiedades opcionales no definidas.
- React Query: adaptación de las cuatro consultas al contrato público `{ datos, error, isLoading, refetch }`.
- Fixtures y MSW: roles mediante el enum `Rol`, funciones mock que faltaban y validación del cuerpo de login como dato no confiable.
- Rutas: corrección del barrel de operaciones y de los imports diferidos para exportaciones nombradas.
- Lightweight Charts: migración a la API modular de la versión 5, con limpieza del gráfico y cálculo de drawdown acumulado.
- KPIs: valores numéricos y unidad separada.

## Validación pendiente por limitación del entorno
El ZIP recibido incluye `node_modules` preparado para Windows. En este entorno Linux:
- `npm test` no pudo arrancar porque el ejecutable `vitest` no tiene permiso de ejecución.
- La ejecución directa de Vitest falla porque falta el binario opcional `@rollup/rollup-linux-x64-gnu`.
- `npm run build` supera `typecheck`, pero no llega a Vite por el mismo problema de permisos del ejecutable.

Por tanto, el tipado está verificado; tests y bundle de producción deben ejecutarse en el entorno Windows del proyecto tras `npm ci`.

## Seguridad operativa
No se ha cambiado la lógica de ejecución de órdenes ni se ha habilitado LIVE. Estas correcciones no constituyen una validación del backend, de estrategias, de conectividad con exchanges ni de trading en producción.
