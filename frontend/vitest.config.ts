/// <reference types="vitest" />
/// <reference types="@testing-library/jest-dom" />

module.exports = {
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    css: false,
    transformMode: {
      web: [/\.[jt]sx$/],
    },
    exclude: ['vite.config.ts'],
  },
};