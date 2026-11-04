import { describe, expect, it } from 'vitest';
import { crearLoteOrdenesMock, leerParametrosOrdenes } from '../ordenesMock';

describe('crearLoteOrdenesMock', () => {
  it('ordena de la más reciente a la más antigua', () => {
    const { ordenes } = crearLoteOrdenesMock({ pagina: 1, tamano: 50, simbolo: null, lado: null });
    const marcas = ordenes.map((orden) => orden.timestamp);
    expect(marcas).toEqual([...marcas].sort().reverse());
  });

  it('filtra por símbolo sin distinguir mayúsculas', () => {
    const { ordenes } = crearLoteOrdenesMock({ pagina: 1, tamano: 50, simbolo: 'ethusdt', lado: null });
    expect(ordenes.length).toBeGreaterThan(0);
    expect(ordenes.every((orden) => orden.simbolo === 'ETHUSDT')).toBe(true);
  });

  it('filtra por lado y pagina', () => {
    const lote = crearLoteOrdenesMock({ pagina: 2, tamano: 2, simbolo: null, lado: 'buy' });
    expect(lote.pagina).toBe(2);
    expect(lote.ordenes).toHaveLength(2);
    expect(lote.ordenes.every((orden) => orden.lado === 'buy')).toBe(true);
  });

  it('devuelve lote vacío con una sola página si nada coincide', () => {
    const lote = crearLoteOrdenesMock({ pagina: 1, tamano: 50, simbolo: 'XRPUSDT', lado: null });
    expect(lote).toMatchObject({ total: 0, ordenes: [], has_more: false, pagina_total: 1 });
  });
});

describe('leerParametrosOrdenes', () => {
  it('usa valores por defecto ante parámetros inválidos', () => {
    const url = new URL('http://localhost:8000/operaciones?pagina=-1&lado=hold');
    expect(leerParametrosOrdenes(url)).toEqual({ pagina: 1, tamano: 50, simbolo: null, lado: null });
  });
});
