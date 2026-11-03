import js from '@eslint/js';
import reactHooks from 'eslint-plugin-react-hooks';
import reactRefresh from 'eslint-plugin-react-refresh';
import globals from 'globals';

export default [
  js.configs.recommended,
  {
    files: ['**/*.{ts,tsx}'],
    languageOptions: {
      ecmaVersion: 2020,
      sourceType: 'module',
      parserOptions: {
        project: './tsconfig.json',
      },
      globals: {
        ...globals.browser,
        ...globals.es2020,
      },
    },
    rules: {
      // Español en nombres
      'no-shadow': 'off',
      '@typescript-eslint/no-shadow': 'error',
      '@typescript-eslint/consistent-type-imports': 'error',
      '@typescript-eslint/no-explicit-any': 'error',
      '@typescript-eslint/explicit-module-boundary-types': 'off',

      // Funciones: máximo 25 líneas en lógica, 80 en componentes
      'max-lines-per-function': ['error', { max: 25, skipBlankLines: true, skipComments: true }],

      // Sin números mágicos
      'no-magic-numbers': 'error',
      '@typescript-eslint/no-magic-numbers': ['error', { ignoreNumericLiteralTypes: true, ignoreEnums: true, ignoreReadonlyClassProperties: true }],

      // React hooks
      'react-hooks/rules-of-hooks': 'error',
      'react-hooks/exhaustive-deps': 'warn',

      // React refresh
      'react-refresh/only-export-components': ['warn', { allowConstantExport: true }],
    },
  },
  {
    files: ['**/*.{ts,tsx}'],
    rules: {
      'max-lines-per-function': ['error', { max: 80, skipBlankLines: true, skipComments: true, Iife: true }],
    },
  },
  {
    files: ['vite.config.ts', 'vitest.config.ts', 'eslint.config.mjs', '.eslintrc.cjs', 'tailwind.config.js', 'postcss.config.js'],
    rules: {
      'max-lines-per-function': 'off',
      '@typescript-eslint/no-magic-numbers': 'off',
    },
  },
];