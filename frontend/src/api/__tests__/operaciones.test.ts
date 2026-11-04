import { afterEach, describe, expect, it, vi } from 'vitest';
import { fetchOperaciones } from '../operaciones';
import { FILTROS_ORDENES_INICIALES } from '../../dominio';
import { crearLoteOrdenesMock } from '../../mocks/ordenesMock';

const simularFetch = (cuerpo: unknown, estado = 200) => {
  const funcion = vi.fn().mockResolvedValue(new Response(JSON.stringify(cuerpo), { status: estado }));
  vi.stubGlobal('fetch', funcion);
  return funcion;
};

describe('fetchOperaciones', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('pide /operaciones con los filtros en la query', async () => {
    const funcion = simularFetch(crearLoteOrdenesMock({ pagina: 1, tamano: 50, simbolo: null, lado: null }));
    await fetchOperaciones({ ...FILTROS_ORDENES_INICIALES, simbolo: 'BTCUSDT', lado: 'buy' });
    expect(String(funcion.mock.calls[0]?.[0])).toMatch(/\/operaciones\?pagina=1&tamano=50&simbolo=BTCUSDT&lado=buy$/);
  });

  it('devuelve el lote validado', async () => {
    simularFetch(crearLoteOrdenesMock({ pagina: 1, tamano: 3, simbolo: null, lado: null }));
    const lote = await fetchOperaciones({ ...FILTROS_ORDENES_INICIALES, tamano: 3 });
    expect(lote.ordenes).toHaveLength(3);
    expect(lote.has_more).toBe(true);
  });

  it('propaga el 503 del backend con su detalle', async () => {
    simularFetch({ detail: 'No se pudo leer el archivo' }, 503);
    await expect(fetchOperaciones(FILTROS_ORDENES_INICIALES)).rejects.toMatchObject({
      statusCode: 503,
      message: 'No se pudo leer el archivo',
    });
  });

  it('falla si la respuesta no cumple el contrato', async () => {
    simularFetch({ total: 'x' });
    await expect(fetchOperaciones(FILTROS_ORDENES_INICIALES)).rejects.toMatchObject({
      code: 'RESPUESTA_INVALIDA',
    });
  });
});
