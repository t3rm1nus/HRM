/// <reference types="vitest" />
/// <reference types="@testing-library/jest-dom" />

import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    css: false,
    transformMode: {
      web: [/\\.[jt]sx$/],
    },
    exclude: ['vite.config.ts'],
  },
});
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    css: false,
    transformMode: {
      web: [/\.[jt]sx$/],
    },
    exclude: ['vite.config.ts'],
    include: ['src/**/*.test.{ts,tsx}'],
  },
};