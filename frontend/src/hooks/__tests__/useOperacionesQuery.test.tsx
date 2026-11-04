import { renderHook, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { crearEnvolvedorConsultas } from '../../test/envolvedorConsultas';
import { usarOperaciones } from '../useOperacionesQuery';
import { fetchOperaciones } from '../../api/operaciones';
import { FILTROS_ORDENES_INICIALES } from '../../dominio';
import { crearLoteOrdenesMock } from '../../mocks/ordenesMock';

vi.mock('../../api/operaciones', () => ({ fetchOperaciones: vi.fn() }));

describe('usarOperaciones', () => {
  const simularFetch = vi.mocked(fetchOperaciones);

  beforeEach(() => vi.clearAllMocks());

  it('carga el lote con los filtros recibidos', async () => {
    const lote = crearLoteOrdenesMock({ pagina: 1, tamano: 50, simbolo: null, lado: null });
    simularFetch.mockResolvedValue(lote);
    const filtros = { ...FILTROS_ORDENES_INICIALES, lado: 'buy' as const };

    const { result } = renderHook(() => usarOperaciones(filtros), { wrapper: crearEnvolvedorConsultas() });
    expect(result.current.isLoading).toBe(true);
    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.datos).toEqual(lote);
    expect(simularFetch).toHaveBeenCalledWith(filtros);
  });

  it('expone el error de la API', async () => {
    simularFetch.mockRejectedValue(new Error('No se pudo leer el archivo'));
    const { result } = renderHook(() => usarOperaciones(FILTROS_ORDENES_INICIALES), {
      wrapper: crearEnvolvedorConsultas(),
    });
    await waitFor(() => expect(result.current.error).toBeInstanceOf(Error));
    expect(result.current.error?.message).toBe('No se pudo leer el archivo');
  });
});
