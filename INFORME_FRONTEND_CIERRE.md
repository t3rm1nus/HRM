# HRM — Informe de cierre de estabilización del frontend

Fecha: 3 de noviembre de 2026  
Base: `HRM-master (3).zip` recibido en esta conversación.

## Resultado ejecutivo

Se ha realizado una segunda pasada de reparación sobre el frontend. Se han corregido el manifiesto y el lockfile, reconstruido la configuración de Vite/Vitest, migrado las gráficas heredadas y retirado módulos de interfaz no conectados al router.

**El frontend todavía no puede declararse cerrado ni certificado:** el entorno de esta revisión no pudo acceder a `registry.npmjs.org`, por lo que no fue posible descargar las dependencias y ejecutar `npm run typecheck`, Vitest o `vite build` con el árbol instalado.

## Cambios aplicados

### Dependencias y configuración

- `package.json` y `package-lock.json` alineados: los bloques raíz de `dependencies` y `devDependencies` coinciden exactamente.
- React y React DOM alineados con las versiones fijadas por el lockfile: `^18.3.1` / `18.3.1`.
- MSW alineado con el lockfile: `^3.0.3` / `3.0.3`.
- Lockfile npm versión 3 conservado como fuente reproducible.
- Eliminado el `package-lock.json` raíz huérfano; el manifiesto de frontend y su lockfile viven juntos en `frontend/`.
- Fijados requisitos mínimos Node.js 20 y npm 10; `packageManager` fijado a npm 10.9.2.
- Añadido `.npmrc` con `legacy-peer-deps=true` por la dependencia peer opcional de MSW 2 declarada por `@vitest/mocker` de Vitest 2, mientras el árbol bloqueado utiliza MSW 3. Debe revisarse al actualizar Vitest/MSW conjuntamente.
- Corregido `vite.config.ts`: alias basado en `import.meta.url`, sin opciones de Vitest dentro de la configuración de Vite.
- Reconstruido `vitest.config.ts`, que tenía contenido duplicado y un bloque sobrante que rompía el parseo.
- Añadido `npm run typecheck`; `npm run build` ejecuta primero el typecheck y después Vite.
- Excluidos los tests del `tsconfig` de producción; Vitest conserva su propio descubrimiento de pruebas.

### Gráficas e indicadores

- Migrados a Recharts los gráficos de línea, área, barras, donut, histograma, serie temporal de regímenes y comparador BTC/ETH.
- Migrado el mapa de calor a una matriz de celdas con escala de color, en lugar de llamar a una serie heatmap que no existe en Lightweight Charts.
- Migrada la gráfica OHLC a la API real de Lightweight Charts 5 (`createChart`, `CandlestickSeries` y `HistogramSeries`).
- Eliminadas llamadas inexistentes como `renderer()`, `Widget`, `addMovingAverage`, `addPieSeries`, `addHeatmapSeries`, `setMarkup`, `set()` y `createPriceBand`.
- Añadidos cálculos puros de media móvil, RSI de Wilder y MACD (12, 26, 9).
- Los indicadores devuelven `null` cuando no existe histórico suficiente. El RSI de una serie plana devuelve 50, una serie estrictamente ascendente 100 y una estrictamente descendente 0.
- La comparación BTC/ETH usa base 100 para comparar rendimientos relativos, no precios nominales.
- La curva de equity suma el PnL de forma acumulada y no presupone un capital inicial arbitrario; el drawdown porcentual se oculta si no se proporciona capital inicial.

### Limpieza de módulos heredados

Eliminados módulos sin referencias desde el router o desde otros módulos activos:

- Panel antiguo (`src/paginas/panel/`) que contenía imports rotos y datos de ejemplo etiquetados como LIVE.
- `DashboardPage.tsx` y `devComponentes.tsx` sin referencias.
- `paginas/operaciones/index.tsx` y `paginas/secciones/index.tsx` con rutas/imports heredados.
- Páginas de CRM no conectadas al router: marketing, ventas, personal, usuarios, catálogo, monedas, portafolio, registro, soporte, comunicación y versión.
- Componentes `Monedas`, `TablaOperaciones` y `themeToggle` no utilizados. `themeToggle` además recorría los modos PAPER/TESTNET/LIVE y no era un control de tema apropiado.
- Corregido el barrel `src/componentes/index.ts` para exportar desde las carpetas reales `graficas` y `mercado`.

### Pruebas añadidas

- Añadida batería Vitest para RSI plano/ascendente/descendente, histórico insuficiente, media móvil y MACD.
- Añadido un lanzador Node portable para ejecutar Vitest con MSW desde Windows, Linux o macOS.
- Corregidos imports de API en cuatro pruebas de hooks (`../../api/usuarios`).

## Validaciones ejecutadas

| Comprobación | Resultado |
|---|---|
| Coincidencia `dependencies` package/lock | Correcta |
| Coincidencia `devDependencies` package/lock | Correcta |
| `npm ls --package-lock-only --depth=0` | Correcto; árbol raíz del lockfile reconocido |
| Resolución de imports relativos | 91 archivos de producción escaneados; 0 imports relativos sin resolver |
| Parseo/transpilación sintáctica TypeScript | 102 archivos TS/TSX; 0 errores sintácticos |
| Ejecución directa de funciones de indicadores | RSI plano 50, RSI subida 100, MACD y señal calculados, SMA comprobada |
| `npm ci` | Bloqueado por `EAI_AGAIN` al resolver `registry.npmjs.org` |
| `npm ci --offline` | Bloqueado: paquete Zustand no disponible en caché (`ENOTCACHED`) |
| `npm run typecheck` | No certificable sin dependencias instaladas |
| `npm test` | No ejecutado: Vitest no instalado |
| `npm run build` | No ejecutado: Vite y dependencias no instalados |

La transpilación sintáctica no sustituye al chequeo de tipos. La comprobación de imports relativos tampoco sustituye al build de Vite.

## Pasos necesarios para cerrar la fase en un equipo con acceso a npm

Desde `frontend/`:

```bash
npm ci
npm run typecheck
npm test
npm run build
```

Si el chequeo de tipos devuelve errores, corregirlos antes de considerar finalizada la fase. Después, ejecutar la aplicación y hacer una comprobación visual del dashboard, navegación, página Mercado y componentes de gráficas con datos de prueba identificados expresamente como tales.

## Estado

**Estado: frontend reparado estáticamente y lockfile alineado; build reproducible pendiente de verificación en un entorno con acceso al registro npm.**

No se ha habilitado trading real ni se ha conectado una fuente de cotizaciones inexistente. La estabilización del frontend no certifica la integración L1-L2-L3 ni el paper trading automatizado.
