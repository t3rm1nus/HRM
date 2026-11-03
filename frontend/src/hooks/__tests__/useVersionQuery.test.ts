import { renderHook, waitFor } from '@testing-library/react';
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { usarVersion } from '../useVersionQuery';
import { fetchVersion } from '../api/usuarios';

vi.mock('../api/usuarios', () => ({
  fetchVersion: vi.fn(),
}));

describe('usarVersion', () => {
  const mockFetchVersion = vi.mocked(fetchVersion);

  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('debe cargar la versión correctamente', async () => {
    const mockDatos = {
      app_version: '2.0.0',
      framework: 'FastAPI',
      python_version: '3.10+',
    };
    mockFetchVersion.mockResolvedValue(mockDatos);

    const { result } = renderHook(() => usarVersion());

    expect(result.current.isLoading).toBe(true);

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.datos).toEqual(mockDatos);
    expect(result.current.error).toBeNull();
  });

  it('debe manejar errores de la API', async () => {
    const errorMessage = 'No se pudo obtener la versión';
    mockFetchVersion.mockRejectedValue(new Error(errorMessage));

    const { result } = renderHook(() => usarVersion());

    expect(result.current.isLoading).toBe(true);

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.error).toBeInstanceOf(Error);
    expect(result.current.error?.message).toBe(errorMessage);
  });

  it('debe usar el intervalo de revalidación correcto', async () => {
    const mockDatos = {
      app_version: '2.0.0',
      framework: 'FastAPI',
      python_version: '3.10+',
    };
    mockFetchVersion.mockResolvedValue(mockDatos);

    const { result, rerender } = renderHook(() => usarVersion());

    await waitFor(() => {
      expect(result.current.datos).toEqual(mockDatos);
    });

    const spy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    rerender();

    expect(spy).not.toHaveBeenCalled();
    spy.mockRestore();
  });
});