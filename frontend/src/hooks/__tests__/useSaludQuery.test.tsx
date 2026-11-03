import { renderHook, waitFor } from '@testing-library/react';
import { crearEnvolvedorConsultas } from '../../test/envolvedorConsultas';
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { usarSalud } from '../useSaludQuery';
import { fetchHealth } from '../../api/usuarios';

vi.mock('../../api/usuarios', () => ({
  fetchHealth: vi.fn(),
}));

describe('usarSalud', () => {
  const mockFetchHealth = vi.mocked(fetchHealth);

  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('debe cargar los datos de salud correctamente', async () => {
    const mockDatos = {
      status: 'ok' as const,
      timestamp: '2026-02-23T10:00:00Z',
      version: '2.0.0',
    };
    mockFetchHealth.mockResolvedValue(mockDatos);

    const { result } = renderHook(() => usarSalud(), { wrapper: crearEnvolvedorConsultas() });

    expect(result.current.isLoading).toBe(true);

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.datos).toEqual(mockDatos);
    expect(result.current.error).toBeNull();
  });

  it('debe manejar errores de la API', async () => {
    const errorMessage = 'Servicio no disponible';
    mockFetchHealth.mockRejectedValue(new Error(errorMessage));

    const { result } = renderHook(() => usarSalud(), { wrapper: crearEnvolvedorConsultas() });

    expect(result.current.isLoading).toBe(true);

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.error).toBeInstanceOf(Error);
    expect(result.current.error?.message).toBe(errorMessage);
  });

  it('debe usar el intervalo de revalidación configurado', async () => {
    const mockDatos = {
      status: 'ok' as const,
      timestamp: '2026-02-23T10:00:00Z',
      version: '2.0.0',
    };
    mockFetchHealth.mockResolvedValue(mockDatos);

    const { result, rerender } = renderHook(() => usarSalud(), { wrapper: crearEnvolvedorConsultas() });

    await waitFor(() => {
      expect(result.current.datos).toEqual(mockDatos);
    });

    const spy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    rerender();

    expect(spy).not.toHaveBeenCalled();
    spy.mockRestore();
  });
});