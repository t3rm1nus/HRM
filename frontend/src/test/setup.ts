import '@testing-library/jest-dom';
import { vi } from 'vitest';
import { mswSetup, mswTeardown } from '../mocks';

Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation((query) => ({
    matches: false,
    media: query,
    onchange: null,
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
});

Object.defineProperty(window, 'localStorage', {
  value: {
    getItem: vi.fn(),
    setItem: vi.fn(),
    removeItem: vi.fn(),
    clear: vi.fn(),
  },
});

Object.defineProperty(window, 'navigator', {
  value: {
    languages: ['es-ES'],
  },
});

Object.defineProperty(document, 'doctype', {
  value: document.implementation.createDocumentType('html', '', ''),
});

Object.defineProperty(document, 'documentElement', {
  value: document.createElement('div'),
});

beforeEach(() => {
  localStorage.clear();
});

beforeAll(() => {
  if (import.meta.env.VITE_USAR_MOCKS === 'true') {
    mswSetup();
  }
});

afterAll(() => {
  if (import.meta.env.VITE_USAR_MOCKS === 'true') {
    mswTeardown();
  }
});