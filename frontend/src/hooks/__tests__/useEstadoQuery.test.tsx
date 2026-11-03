import { renderHook, waitFor } from '@testing-library/react';
import { crearEnvolvedorConsultas } from '../../test/envolvedorConsultas';
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { usarEstado } from '../useEstadoQuery';
import { fetchStatus } from '../../api/usuarios';

vi.mock('../../api/usuarios', () => ({
  fetchStatus: vi.fn(),
}));

describe('usarEstado', () => {
  const mockFetchStatus = vi.mocked(fetchStatus);

  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('debe cargar el estado del sistema correctamente', async () => {
    const mockDatos = {
      status: 'running' as const,
      trading_active: true,
      cycle_count: 42,
      uptime: 'sistema activo',
    };
    mockFetchStatus.mockResolvedValue(mockDatos);

    const { result } = renderHook(() => usarEstado(), { wrapper: crearEnvolvedorConsultas() });

    expect(result.current.isLoading).toBe(true);

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.datos).toEqual(mockDatos);
    expect(result.current.error).toBeNull();
  });

  it('debe manejar errores de la API', async () => {
    const errorMessage = 'No se pudo obtener el estado';
    mockFetchStatus.mockRejectedValue(new Error(errorMessage));

    const { result } = renderHook(() => usarEstado(), { wrapper: crearEnvolvedorConsultas() });

    expect(result.current.isLoading).toBe(true);

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.error).toBeInstanceOf(Error);
    expect(result.current.error?.message).toBe(errorMessage);
  });

  it('debe usar el intervalo de revalidación configurable', async () => {
    const mockDatos = {
      status: 'running' as const,
      trading_active: true,
      cycle_count: 42,
      uptime: 'sistema activo',
    };
    mockFetchStatus.mockResolvedValue(mockDatos);

    const { result, rerender } = renderHook(() => usarEstado(), { wrapper: crearEnvolvedorConsultas() });

    await waitFor(() => {
      expect(result.current.datos).toEqual(mockDatos);
    });

    expect(mockFetchStatus).toHaveBeenCalledTimes(1);

    const spy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    rerender();

    expect(spy).not.toHaveBeenCalled();
    spy.mockRestore();
  });
});