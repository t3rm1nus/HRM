import { describe, it, expect } from 'vitest';
import { calcularDesviacion } from '../cartera';

describe('calcularDesviacion', () => {
  it('calcula desviacion media', () => {
    expect(calcularDesviacion([{ activo: 'BTC', objetivo: 0.5, real: 0.4 }, { activo: 'ETH', objetivo: 0.5, real: 0.6 }])).toBe(0.1);
  });
  it('retorna 0 si vacio', () => {
    expect(calcularDesviacion([])).toBe(0);
  });
});
