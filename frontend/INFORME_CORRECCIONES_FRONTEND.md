# Correcciones del frontend HRM

## Cambios aplicados

1. `src/index.css`
   - Corregida la declaración CSS inválida `--color-sup texto`.
   - Nombre resultante: `--color-superficie-texto`.

2. `src/test/setup.ts`
   - Eliminadas las sobrescrituras manuales de `navigator`, `document.doctype` y `document.documentElement`.
   - jsdom debe proporcionar esos objetos nativos. La sobrescritura de `navigator` eliminaba propiedades como `userAgent` que React necesita durante su inicialización.
   - Se conservan los mocks de `matchMedia` y `localStorage`, además de la configuración MSW.

3. `src/dominio/graficas/funciones.ts`
   - `formatearPrecio` ahora devuelve dos decimales con punto decimal, sin separador de miles, como espera el contrato actual del test y como resulta útil para representar precios de mercado.

4. `src/dominio/graficas/__tests__/funciones.test.ts`
   - Corregida la expectativa del índice del pico: el valor 105 está en el índice 1, no en el 2.

5. `src/dominio/rendimiento/__tests__/funciones.test.ts`
   - Corregidas las expectativas de comisiones: 1 + 1 + 2 + 1 = 5, no 6.
   - Corregido el comentario aritmético del equity: 100 + 19 - 11 + 28 + 24 = 160.

## Qué no se ha cambiado

- No se ha modificado la lógica de envío de órdenes.
- No se ha habilitado LIVE.
- No se han cambiado los controles de riesgo ni la configuración de exchanges.
- No se han actualizado dependencias ni se recomienda ejecutar `npm audit fix --force`.

## Validación

El ZIP se ha preparado sin `node_modules` para evitar trasladar binarios específicos de Windows/Linux. En este entorno no se ha podido completar `npm ci` ni ejecutar Vitest/Vite, por lo que **no se afirma que los tests o el bundle estén superados**.

## Cómo validar en Windows

Desde PowerShell, en la carpeta `frontend`:

```powershell
npm ci
npm run typecheck
npm test
npm run build
```

Si `npm test` todavía falla, comparte la salida completa, especialmente el primer error de cada suite. Si `npm run build` falla, comparte el primer error de PostCSS/Vite.

Las vulnerabilidades notificadas por `npm audit` deben revisarse por separado. No ejecutes `npm audit fix --force` sin evaluar antes las actualizaciones incompatibles que pueda introducir.
