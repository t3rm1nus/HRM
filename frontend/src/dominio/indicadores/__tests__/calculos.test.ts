import { describe, expect, it } from 'vitest';
import { calcularMacd, calcularMediaMovil, calcularRsi } from '../calculos';

describe('indicadores técnicos', () => {
  it('calcula RSI 50 en una serie sin variaciones', () => {
    const precios = Array.from({ length: 20 }, () => 100);
    expect(calcularRsi(precios).at(-1)).toBe(50);
  });

  it('calcula RSI 100 en una subida continua', () => {
    const precios = Array.from({ length: 30 }, (_, indice) => 100 + indice);
    expect(calcularRsi(precios).at(-1)).toBe(100);
  });

  it('calcula RSI 0 en una caída continua', () => {
    const precios = Array.from({ length: 30 }, (_, indice) => 200 - indice);
    expect(calcularRsi(precios).at(-1)).toBe(0);
  });

  it('devuelve null cuando no hay suficientes observaciones para RSI', () => {
    expect(calcularRsi([100, 101, 102])).toEqual([null, null, null]);
  });

  it('calcula una media móvil simple alineada con la serie original', () => {
    expect(calcularMediaMovil([1, 2, 3, 4, 5], 3)).toEqual([null, null, 2, 3, 4]);
  });

  it('calcula MACD, señal e histograma con suficiente histórico', () => {
    const precios = Array.from({ length: 50 }, (_, indice) => 100 + indice);
    const resultado = calcularMacd(precios);
    expect(resultado.macd.at(-1)).not.toBeNull();
    expect(resultado.senal.at(-1)).not.toBeNull();
    expect(resultado.histograma.at(-1)).not.toBeNull();
  });

  it('no inventa valores MACD si el histórico es insuficiente', () => {
    const resultado = calcularMacd([100, 101, 102]);
    expect(resultado.macd).toEqual([null, null, null]);
    expect(resultado.senal).toEqual([null, null, null]);
    expect(resultado.histograma).toEqual([null, null, null]);
  });
});
