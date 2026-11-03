import { describe, it, expect } from 'vitest';
import {
  calcularDrawdownSubmarino,
  calcularTasaExito,
  validarDatosVela,
  calcularMediaMovil,
  formatearPrecio,
  obtenerColorTendencia,
} from '../funciones';

const preciosEjemplo = [100, 102, 101, 98, 95, 97, 103, 105, 102, 100];

describe('funciones dominio graficas', () => {
  describe('calcularDrawdownSubmarino', () => {
    it('calcula correctamente el drawdown con pérdida mayor', () => {
      const precios = [100, 105, 103, 90, 85, 90, 95];
      const resultado = calcularDrawdownSubmarino(precios);
      expect(resultado.maxDrawdown).toBeGreaterThan(15);
      expect(resultado.precioAlto).toBe(105);
      expect(resultado.indiceAlto).toBe(2);
    });

    it('retorna 0 para precios monótonos', () => {
      const precios = [100, 101, 102, 103, 104];
      const resultado = calcularDrawdownSubmarino(precios);
      expect(resultado.maxDrawdown).toBe(0);
    });

    it('maneja array con un solo elemento', () => {
      const resultado = calcularDrawdownSubmarino([100]);
      expect(resultado.maxDrawdown).toBe(0);
    });
  });

  describe('calcularTasaExito', () => {
    it('calcula correctamente tasa de éxito', () => {
      const estados = [
        'pendiente',
        'completado',
        'completado',
        'fallido',
        'completado',
      ] as const;
      const resultado = calcularTasaExito(estados);
      expect(resultado).toBeCloseTo(60);
    });

    it('retorna 0 para array vacío', () => {
      const resultado = calcularTasaExito([] as const);
      expect(resultado).toBe(0);
    });

    it('retorna 100 cuando todas son completadas', () => {
      const estados = [
        'completado',
        'completado',
        'completado',
      ] as const;
      const resultado = calcularTasaExito(estados);
      expect(resultado).toBe(100);
    });
  });

  describe('validarDatosVela', () => {
    it('valida y transforma datos de vela correctamente', () => {
      const datosRaw = [
        { timestamp: '2024-01-01T00:00:00Z', open: 100, high: 105, low: 95, close: 102, volume: 1000 },
        { timestamp: '2024-01-02T00:00:00Z', open: 102, high: 108, low: 97, close: 105, volume: 1200 },
      ];
      const resultado = validarDatosVela(datosRaw);
      expect(resultado).toHaveLength(2);
      expect(resultado[0].open).toBe(100);
      expect(resultado[0].high).toBe(105);
    });

    it('lanza error para datos inválidos', () => {
      const datosInvalidos = [
        { timestamp: 'bad', open: -1, high: 105, low: 95, close: 102 },
      ];
      expect(() => validarDatosVela(datosInvalidos)).toThrow();
    });
  });

  describe('calcularMediaMovil', () => {
    it('calcula medias móviles simples', () => {
      const precios = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
      const resultado = calcularMediaMovil(precios, [3, 5]);
      expect(resultado.get(3)).toBeDefined();
      expect(resultado.get(5)).toBeDefined();
    });
  });

  describe('formatearPrecio', () => {
    it('formatea precio con 2 decimales', () => {
      expect(formatearPrecio(1234.56)).toBe('1234.56');
      expect(formatearPrecio(0.5)).toBe('0.50');
    });
  });

  describe('obtenerColorTendencia', () => {
    it('retorna verde para aumento', () => {
      expect(obtenerColorTendencia(105, 100)).toBe('bg-green-500');
    });

    it('retorna rojo para descenso', () => {
      expect(obtenerColorTendencia(95, 100)).toBe('bg-red-500');
    });

    it('retorna gris para sin cambio', () => {
      expect(obtenerColorTendencia(100, 100)).toBe('bg-gray-500');
    });
  });
});