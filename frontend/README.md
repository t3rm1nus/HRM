# HRM Frontend

Interfaz React + TypeScript de HRM.

## Requisitos

- Node.js 20 o superior.
- npm 10 o superior.

La versión exacta de cada dependencia se obtiene de `package-lock.json`.

## Instalación reproducible

Desde esta carpeta:

```bash
npm ci
```

No sustituir `npm ci` por `npm install` en instalaciones de despliegue: `npm ci` instala el árbol fijado en el lockfile y falla si el manifiesto y el lockfile divergen.

## Comandos

```bash
npm run dev        # servidor de desarrollo
npm run typecheck  # comprobación TypeScript
npm run build      # typecheck y build de producción
npm test           # pruebas Vitest
```

## Gráficas

- Recharts: líneas, áreas, barras, histogramas, distribución y comparaciones.
- Lightweight Charts 5: velas OHLC y volumen.
- Indicadores: RSI de Wilder, MACD (12, 26, 9) y medias móviles simples.

Los indicadores no dibujan valores de relleno cuando el histórico es insuficiente.

## Seguridad

El frontend no debe inferir que el trading está activo por el mero hecho de tener conexión con la API o con WebSocket. La habilitación de live debe depender del estado autoritativo del backend y de sus controles de seguridad. La pantalla Mercado permanece sin cotizaciones mientras no exista una fuente real conectada.

En Windows PowerShell, los tests con MSW también pueden ejecutarse de forma portable:

```powershell
npm run test:msw
```
