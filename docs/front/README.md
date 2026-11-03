# Estructura Frontend

Esta carpeta contiene la aplicación frontend construida con Vite + TypeScript.

## Estructura de carpetas

- `src/` - Código fuente de la aplicación
  - `components/` - Componentes React (máximo 150 líneas cada uno)
  - `hooks/` - Hooks personalizados
  - `pages/` - Páginas de la aplicación
  - `constants/` - Constantes y configuraciones
  - `utils/` - Utilidades generales

- `api/` - Endpoints de la API (consumidos por el frontend)
- `tests/` - Tests unitarios
- `index.html` - Punto de entrada HTML
- `main.ts` - Punto de entrada de TypeScript

## Convenios de código

- Nombres en español, descriptivos y con verbo
- Funciones máx. 20 líneas, máximo 3 parámetros
- Componentes React máx. 150 líneas
- Sin números ni textos mágicos (usar constantes)
- Tipado estricto TypeScript (sin `any`)
- Tests unitarios para funciones de dominio

## Desarrollo

Ejecutar `npm install` y luego `npm run dev` para iniciar el servidor de desarrollo.