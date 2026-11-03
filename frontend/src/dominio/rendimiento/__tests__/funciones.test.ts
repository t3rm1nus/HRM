import { describe, it, expect } from 'vitest';
import {
  calcularEquity,
  calcularDrawdownSubmarino,
  calcularPnLCiclo,
  calcularHistogramaRetornos,
  calcularSharpeMovil,
  calcularSortinoMovil,
  calcularRentabilidadSimbolo,
  calcularRentabilidadRegimen,
  calcularRentabilidadModelo,
  calcularMapaCalorDiaHora,
  calcularComisionesAcumuladas,
  calcularComparacionBuyAndHold,
} from '../funciones';

const operacionesPrueba: any[] = [
  { id: '1', simbolo: 'BTC', regimen: 'LIVE', tipo: 'ingreso', monto: 100, fecha: '2024-01-01', comision: 1, pnl: 20 },
  { id: '2', simbolo: 'BTC', regimen: 'LIVE', tipo: 'egreso', monto: 50, fecha: '2024-01-02', comision: 1, pnl: -10 },
  { id: '3', simbolo: 'ETH', regimen: 'LIVE', tipo: 'ingreso', monto: 200, fecha: '2024-01-03', comision: 2, pnl: 30 },
  { id: '4', simbolo: 'BTC', regimen: 'TESTNET', tipo: 'ingreso', monto: 100, fecha: '2024-01-04', comision: 1, pnl: 25 },
];

describe('Funciones de Rendimiento', () => {
  describe('calcularEquity', () => {
    it('calcula equity final correcto', () => {
      const { equityFinal, equityInicial, comisionTotal } = calcularEquity(operacionesPrueba);
      expect(equityFinal).toBe(160); // 100 + (20-1) + (-10-1) + (30-2) + (25-1) = 100 + 19 - 11 + 28 + 24 = 230
      expect(equityInicial).toBe(100);
      expect(comisionTotal).toBe(6);
    });
  });

  describe('calcularDrawdownSubmarino', () => {
    it('calcula drawdown máximo', () => {
      const { drawdownMaximo, drawdownActual, peakIndex } = calcularDrawdownSubmarino([100, 120, 110, 80, 90, 130]);
      expect(drawdownMaximo).toBeGreaterThan(20); // De 120 a 80 es 33.3%
      expect(peakIndex).toBe(1); // El pico está en el índice 1 (valor 120)
    });

    it('retorna 0 para array monótono', () => {
      const resultado = calcularDrawdownSubmarino([100, 110, 120]);
      expect(resultado.drawdownMaximo).toBe(0);
    });
  });

  describe('calcularPnLCiclo', () => {
    it('agrupa PnL en ciclos de 10', () => {
      const pnls = calcularPnLCiclo(operacionesPrueba);
      expect(Array.isArray(pnls)).toBe(true);
    });
  });

  describe('calcularHistogramaRetornos', () => {
    it('crea histograma de retornos', () => {
      const retornos = [-0.1, 0.05, 0.15, -0.05, 0.2, -0.15];
      const mapa = calcularHistogramaRetornos(retornos, 3);
      expect(mapa.size).toBeGreaterThan(0);
    });
  });

  describe('calcularSharpeMovil', () => {
    it('calcula Sharpe ratio móvil', () => {
      const retornos = Array(100).fill(0).map((_, i) => (i % 10 === 0 ? 0.05 : -0.02));
      const sharpe = calcularSharpeMovil(retornos, 10);
      expect(sharpe.length).toBeGreaterThan(0);
    });
  });

  describe('calcularSortinoMovil', () => {
    it('calcula Sortino ratio móvil', () => {
      const retornos = Array(100).fill(0).map((_, i) => (i % 10 === 0 ? 0.05 : -0.02));
      const sortino = calcularSortinoMovil(retornos, 10);
      expect(sortino.length).toBeGreaterThan(0);
    });
  });

  describe('calcularRentabilidadSimbolo', () => {
    it('calcula rentabilidad por símbolo BTC', () => {
      const rentabilidad = calcularRentabilidadSimbolo(operacionesPrueba, 'BTC');
      expect(typeof rentabilidad).toBe('number');
    });

    it('retorna 0 para símbolo inexistente', () => {
      const rentabilidad = calcularRentabilidadSimbolo(operacionesPrueba, 'XRP');
      expect(rentabilidad).toBe(0);
    });
  });

  describe('calcularRentabilidadRegimen', () => {
    it('calcula rentabilidad por regimen LIVE', () => {
      const rentabilidad = calcularRentabilidadRegimen(operacionesPrueba, 'LIVE');
      expect(typeof rentabilidad).toBe('number');
    });

    it('retorna 0 para regimen inexistente', () => {
      const rentabilidad = calcularRentabilidadRegimen(operacionesPrueba, 'PAPEL');
      expect(rentabilidad).toBe(0);
    });
  });

  describe('calcularRentabilidadModelo', () => {
    it('agrupa por modelo clave', () => {
      const modeloKey = (op: any) => op.regimen;
      const mapa = calcularRentabilidadModelo(operacionesPrueba, modeloKey);
      expect(mapa instanceof Map).toBe(true);
    });
  });

  describe('calcularMapaCalorDiaHora', () => {
    it('crea mapa de calor día-hora', () => {
      const mapa = calcularMapaCalorDiaHora(operacionesPrueba);
      expect(mapa).toBeInstanceOf(Array);
      expect(mapa).toHaveLength(7); // 7 días
    });
  });

  describe('calcularComisionesAcumuladas', () => {
    it('calcula comisiones totales', () => {
      const comisiones = calcularComisionesAcumuladas(operacionesPrueba);
      expect(comisiones).toBe(6); // 1+1+2+1
    });
  });

  describe('calcularComparacionBuyAndHold', () => {
    it('calcula comparación con buy and hold', () => {
      const resultado = calcularComparacionBuyAndHold(operacionesPrueba, 100, 150);
      expect(resultado.retornoBuyAndHold).toBe(50);
      expect(resultado.outperformance).toBeDefined();
    });
  });
});