import { describe, it, expect } from 'vitest';
import {
  FILTROS_ORDENES_INICIALES,
  cambiarFiltros,
  construirParametrosConsulta,
  describirRango,
  esquemaLoteOrdenes,
  etiquetaLado,
  formatearFechaOrden,
  irAPagina,
} from '../ordenes';

const loteDelBackend = {
  total: 1,
  ordenes: [
    {
      order_id: 'BTCUSDT_1',
      simbolo: 'BTCUSDT',
      lado: 'buy',
      cantidad: 0.001,
      precio_llenado: 50000,
      comision: 0.25,
      status: 'filled',
      timestamp: '2026-02-12T16:29:21.561813',
      estrategia: null,
      metadata: { ciclo: 12, motivo: null, condiciones_mercado: {} },
    },
  ],
  has_more: false,
  pagina: 1,
  pagina_total: 1,
};

describe('esquemaLoteOrdenes', () => {
  it('acepta el contrato real del backend, con nulos incluidos', () => {
    expect(esquemaLoteOrdenes.safeParse(loteDelBackend).success).toBe(true);
  });

  it('rechaza un lado desconocido', () => {
    const lote = { ...loteDelBackend, ordenes: [{ ...loteDelBackend.ordenes[0], lado: 'short' }] };
    expect(esquemaLoteOrdenes.safeParse(lote).success).toBe(false);
  });

  it('rechaza un lote sin paginación', () => {
    const { pagina_total: _omitido, ...incompleto } = loteDelBackend;
    expect(esquemaLoteOrdenes.safeParse(incompleto).success).toBe(false);
  });
});

describe('construirParametrosConsulta', () => {
  it('omite símbolo y lado cuando están vacíos', () => {
    expect(construirParametrosConsulta(FILTROS_ORDENES_INICIALES)).toBe('pagina=1&tamano=50');
  });

  it('incluye símbolo recortado y lado', () => {
    const filtros = { ...FILTROS_ORDENES_INICIALES, simbolo: ' ethusdt ', lado: 'sell' as const };
    expect(construirParametrosConsulta(filtros)).toBe('pagina=1&tamano=50&simbolo=ethusdt&lado=sell');
  });
});

describe('cambios de filtros', () => {
  it('vuelve a la primera página al cambiar un filtro', () => {
    const enPaginaTres = irAPagina(FILTROS_ORDENES_INICIALES, 3);
    expect(cambiarFiltros(enPaginaTres, { lado: 'buy' })).toMatchObject({ pagina: 1, lado: 'buy' });
  });

  it('nunca baja de la página 1', () => {
    expect(irAPagina(FILTROS_ORDENES_INICIALES, -4).pagina).toBe(1);
  });
});

describe('formato', () => {
  it('traduce el lado', () => {
    expect(etiquetaLado('buy')).toBe('Compra');
    expect(etiquetaLado('sell')).toBe('Venta');
  });

  it('devuelve el texto original si la fecha no es válida', () => {
    expect(formatearFechaOrden('no-es-fecha')).toBe('no-es-fecha');
  });

  it('muestra la hora guardada sin desplazarla de zona', () => {
    expect(formatearFechaOrden('2026-02-12T16:29:21.561813')).toMatch(/16:29:21/);
  });

  it('describe el rango del lote', () => {
    const lote = esquemaLoteOrdenes.parse({ ...loteDelBackend, total: 120, pagina: 2, pagina_total: 3 });
    expect(describirRango(lote)).toBe('Página 2 de 3 · 120 órdenes');
  });
});
