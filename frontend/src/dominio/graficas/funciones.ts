import { z } from 'zod';
import { EstadoOperacion } from '../tipos';

export interface DatosCandle {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

export interface DatosLinea {
  timestamp: string;
  valor: number;
}

export interface DatosArea {
  timestamp: string;
  valor: number;
}

export interface DatosBarras {
  timestamp: string;
  valor: number;
  categoria?: string;
}

export interface DatosDonut {
  nombre: string;
  valor: number;
  color?: string;
}

export interface DatosHistograma {
  rango: string;
  count: number;
  monto?: number;
}

export interface DatosMapaCalor {
  x: string;
  y: string;
  valor: number;
}

export interface DatosMedidor {
  min: number;
  max: number;
  actual: number;
  umbral?: number;
}

export interface DatosTemporalRegimenes {
  timestamp: string;
  regimen: 'PAPEL' | 'TESTNET' | 'LIVE';
  activo: boolean;
}

/** Calcula el drawdown submarino máximo (pico a valle) */
export const calcularDrawdownSubmarino = (precios: readonly number[]): {
  maxDrawdown: number;
  precioAlto: number;
  precioBajo: number;
  indiceAlto: number;
  indiceBajo: number;
} => {
  if (precios.length < 2) {
    return { maxDrawdown: 0, precioAlto: 0, precioBajo: 0, indiceAlto: 0, indiceBajo: 0 };
  }

  let maxDrawdown = 0;
  let precioAlto = precios[0] ?? 0;
  let precioBajo = precios[0] ?? 0;
  let indiceAlto = 0;
  let indiceBajo = 0;

  for (let i = 1; i < precios.length; i++) {
    const precio = precios[i] ?? precioAlto;
    if (precio > precioAlto) {
      precioAlto = precio;
      indiceAlto = i;
    } else {
      const drawdown = ((precioAlto - precio) / precioAlto) * 100;
      if (drawdown > maxDrawdown) {
        maxDrawdown = drawdown;
        precioBajo = precio;
        indiceBajo = i;
      }
    }
  }

  return { maxDrawdown, precioAlto, precioBajo, indiceAlto, indiceBajo };
};

/** Calcula la tasa de éxito (operaciones completadas / total) */
export const calcularTasaExito = (estados: readonly EstadoOperacion[]): number => {
  const total = estados.length;
  if (total === 0) return 0;
  const completadas = estados.filter(
    (e) => e === EstadoOperacion.COMPLETADO
  ).length;
  return (completadas / total) * 100;
};

/** Valida y transforma datos crudos a formato vela */
export const validarDatosVela = (datosRaw: unknown[]): DatosCandle[] => {
  const schema = z.object({
    timestamp: z.string().datetime(),
    open: z.number().positive(),
    high: z.number().positive(),
    low: z.number().positive(),
    close: z.number().positive(),
    volume: z.number().optional(),
  });

  return datosRaw.map((dato) => {
    const vela = schema.parse(dato);
    return {
      timestamp: vela.timestamp,
      open: vela.open,
      high: vela.high,
      low: vela.low,
      close: vela.close,
      ...(vela.volume !== undefined ? { volume: vela.volume } : {}),
    };
  });
};

/** Calcula medias móviles simples */
export const calcularMediaMovil = (precios: number[], periodos: number[]): Map<number, number[]> => {
  const resultado = new Map<number, number[]>();
  periodos.forEach((p) => {
    const medias: number[] = [];
    for (let i = p - 1; i < precios.length; i++) {
      const slice = precios.slice(i - p + 1, i + 1);
      const suma = slice.reduce((a, b) => a + b, 0);
      medias.push(suma / p);
    }
    resultado.set(p, medias);
  });
  return resultado;
};

/** Formatea precio a cadena con 2 decimales */
export const formatearPrecio = (precio: number): string => precio.toFixed(2);

/** Obtiene color según tendencia */
export const obtenerColorTendencia = (actual: number, anterior: number): string =>
  actual > anterior ? 'bg-green-500' : actual < anterior ? 'bg-red-500' : 'bg-gray-500';

/** Función pura sin efectos secundarios - exportada para tests */
