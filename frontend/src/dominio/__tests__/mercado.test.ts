import { describe, expect, it } from 'vitest';
import {
  MAX_VELAS_EN_MEMORIA,
  fusionarEstadoConActualizacion,
  fusionarUltimaVela,
  interpretarActualizacion,
  velaADatosCandle,
  type LoteVelas,
  type Vela,
} from '../mercado';

const crearVela = (minuto: number, cierre = 100): Vela => ({
  instante: new Date(Date.UTC(2026, 10, 4, 10, minuto)).toISOString(),
  apertura: 99, maximo: 101, minimo: 98, cierre, volumen: 5,
});

const crearLote = (minutos: number[]): LoteVelas => ({
  simbolo: 'BTCUSDT', cantidad: minutos.length, frescura: 'actual',
  velas: minutos.map((m) => crearVela(m)),
});

describe('fusionarUltimaVela', () => {
  it('reemplaza la última vela si tiene el mismo instante', () => {
    const lote = fusionarUltimaVela(crearLote([0, 1]), crearVela(1, 150), 'actual');
    expect(lote.cantidad).toBe(2);
    expect(lote.velas[1]?.cierre).toBe(150);
  });

  it('añade una vela posterior', () => {
    expect(fusionarUltimaVela(crearLote([0, 1]), crearVela(2), 'actual').cantidad).toBe(3);
  });

  it('ignora una vela anterior a la última', () => {
    expect(fusionarUltimaVela(crearLote([5]), crearVela(1), 'actual').cantidad).toBe(1);
  });

  it('propaga la frescura recibida', () => {
    expect(fusionarUltimaVela(crearLote([0]), crearVela(0), 'obsoleto').frescura).toBe('obsoleto');
  });

  it('acota el histórico en memoria', () => {
    const minutos = Array.from({ length: MAX_VELAS_EN_MEMORIA }, (_, i) => i);
    const lote = fusionarUltimaVela(crearLote(minutos), crearVela(MAX_VELAS_EN_MEMORIA), 'actual');
    expect(lote.cantidad).toBe(MAX_VELAS_EN_MEMORIA);
  });

  it('parte de un lote vacío', () => {
    expect(fusionarUltimaVela(crearLote([]), crearVela(0), 'actual').cantidad).toBe(1);
  });
});

describe('interpretarActualizacion', () => {
  it('acepta un mensaje válido', () => {
    const datos = { frescura: 'actual', ultimas_velas: { BTCUSDT: crearVela(0) } };
    expect(interpretarActualizacion(datos)?.frescura).toBe('actual');
  });

  it('rechaza mensajes malformados', () => {
    expect(interpretarActualizacion({ frescura: 'rara' })).toBeNull();
    expect(interpretarActualizacion(null)).toBeNull();
  });
});

describe('velaADatosCandle', () => {
  it('traduce los nombres de campo', () => {
    expect(velaADatosCandle(crearVela(0, 120))).toMatchObject({ close: 120, open: 99, high: 101, low: 98 });
  });
});

describe('fusionarEstadoConActualizacion', () => {
  it('incorpora los símbolos nuevos aunque el estado llegase vacío', () => {
    const estado = { disponible: false, frescura: 'sin_datos', fuente: null, edad_segundos: null,
      validez_segundos: null, simbolos: [], modo: 'paper' } as const;
    const actualizacion = { frescura: 'actual', ultimas_velas: { ETHUSDT: crearVela(0), BTCUSDT: crearVela(0) } } as const;
    const resultado = fusionarEstadoConActualizacion({ ...estado, simbolos: [] }, actualizacion);
    expect(resultado).toMatchObject({ disponible: true, frescura: 'actual', simbolos: ['BTCUSDT', 'ETHUSDT'] });
  });
});
