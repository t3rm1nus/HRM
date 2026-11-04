import { describe, it, expect } from 'vitest';
import { derivarEstadoConexion } from '../conexion';
import { calcularRentabilidad } from '../cartera';

describe('derivarEstadoConexion', () => {
  it('es pendiente mientras no hay datos ni error', () => {
    expect(derivarEstadoConexion({ hayError: false, hayDatos: false })).toBe('pendiente');
  });

  it('es conectado cuando hay datos', () => {
    expect(derivarEstadoConexion({ hayError: false, hayDatos: true })).toBe('conectado');
  });

  it('es desconectado cuando hay error, aunque queden datos antiguos', () => {
    expect(derivarEstadoConexion({ hayError: true, hayDatos: true })).toBe('desconectado');
  });
});

describe('calcularRentabilidad', () => {
  it('calcula pnl y porcentaje', () => {
    expect(calcularRentabilidad(3300, 3000)).toEqual({ pnl: 300, porcentaje: 10 });
  });

  it('no calcula porcentaje sin capital inicial', () => {
    expect(calcularRentabilidad(100, 0)).toEqual({ pnl: 100, porcentaje: null });
  });
});
