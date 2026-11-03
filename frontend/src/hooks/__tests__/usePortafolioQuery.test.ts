import { renderHook, waitFor } from '@testing-library/react';
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { usarPortafolio } from '../usePortafolioQuery';
import { fetchPortfolio } from '../api/usuarios';

vi.mock('../../../api/clienteApi', () => ({
  fetchPortfolio: vi.fn(),
}));

describe('usarPortafolio', () => {
  const mockFetchPortfolio = vi.mocked(fetchPortfolio);

  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('debe cargar el portafolio correctamente', async () => {
    const mockDatos = {
      btc_balance: 0.002345,
      eth_balance: 0.004567,
      usdt_balance: 3000.0,
      total_nav: 3000.0,
      initial_capital: 3000.0,
    };
    mockFetchPortfolio.mockResolvedValue(mockDatos);

    const { result } = renderHook(() => usarPortafolio());

    expect(result.current.isLoading).toBe(true);

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.datos).toEqual(mockDatos);
    expect(result.current.error).toBeNull();
  });

  it('debe manejar errores de la API', async () => {
    const errorMessage = 'No se pudo obtener el portafolio';
    mockFetchPortfolio.mockRejectedValue(new Error(errorMessage));

    const { result } = renderHook(() => usarPortafolio());

    expect(result.current.isLoading).toBe(true);

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.error).toBeInstanceOf(Error);
    expect(result.current.error?.message).toBe(errorMessage);
  });

  it('debe usar el intervalo de revalidación correcto', async () => {
    const mockDatos = {
      btc_balance: 0.002345,
      eth_balance: 0.004567,
      usdt_balance: 3000.0,
      total_nav: 3000.0,
      initial_capital: 3000.0,
    };
    mockFetchPortfolio.mockResolvedValue(mockDatos);

    const { result, rerender } = renderHook(() => usarPortafolio());

    await waitFor(() => {
      expect(result.current.datos).toEqual(mockDatos);
    });

    const spy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    rerender();

    expect(spy).not.toHaveBeenCalled();
    spy.mockRestore();
  });
});