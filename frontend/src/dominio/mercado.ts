import { z } from 'zod';
import type { DatosCandle } from './graficas/funciones';

export const MAX_VELAS_EN_MEMORIA = 1000;

export const esquemaFrescura = z.enum(['sin_datos', 'actual', 'obsoleto']);

export const esquemaVela = z.object({
  instante: z.string(),
  apertura: z.number(),
  maximo: z.number(),
  minimo: z.number(),
  cierre: z.number(),
  volumen: z.number(),
});

export const esquemaEstadoMercado = z.object({
  disponible: z.boolean(),
  frescura: esquemaFrescura,
  fuente: z.string().nullable(),
  edad_segundos: z.number().nullable(),
  validez_segundos: z.number().nullable(),
  simbolos: z.array(z.string()),
  modo: z.literal('paper'),
});

export const esquemaLoteVelas = z.object({
  simbolo: z.string(),
  cantidad: z.number().int().nonnegative(),
  frescura: esquemaFrescura,
  velas: z.array(esquemaVela),
});

export const esquemaActualizacionMercado = z.object({
  frescura: esquemaFrescura,
  ultimas_velas: z.record(z.string(), esquemaVela),
});

export type Frescura = z.infer<typeof esquemaFrescura>;
export type Vela = z.infer<typeof esquemaVela>;
export type EstadoMercado = z.infer<typeof esquemaEstadoMercado>;
export type LoteVelas = z.infer<typeof esquemaLoteVelas>;
export type ActualizacionMercado = z.infer<typeof esquemaActualizacionMercado>;

const tiempoDe = (vela: Vela): number => new Date(vela.instante).getTime();

export const interpretarActualizacion = (datos: unknown): ActualizacionMercado | null => {
  const resultado = esquemaActualizacionMercado.safeParse(datos);
  return resultado.success ? resultado.data : null;
};

const reemplazarOAnadir = (velas: Vela[], nueva: Vela): Vela[] => {
  const ultima = velas[velas.length - 1];
  if (ultima === undefined) return [nueva];
  if (tiempoDe(nueva) < tiempoDe(ultima)) return velas;
  if (tiempoDe(nueva) === tiempoDe(ultima)) return [...velas.slice(0, -1), nueva];
  return [...velas, nueva].slice(-MAX_VELAS_EN_MEMORIA);
};

export const fusionarUltimaVela = (lote: LoteVelas, nueva: Vela, frescura: Frescura): LoteVelas => {
  const velas = reemplazarOAnadir(lote.velas, nueva);
  return { ...lote, velas, cantidad: velas.length, frescura };
};

export const fusionarEstadoConActualizacion = (
  estado: EstadoMercado,
  actualizacion: ActualizacionMercado,
): EstadoMercado => {
  const simbolos = [...new Set([...estado.simbolos, ...Object.keys(actualizacion.ultimas_velas)])].sort();
  return { ...estado, disponible: true, frescura: actualizacion.frescura, simbolos };
};

export const velaADatosCandle = (vela: Vela): DatosCandle => ({
  timestamp: vela.instante,
  open: vela.apertura,
  high: vela.maximo,
  low: vela.minimo,
  close: vela.cierre,
  volume: vela.volumen,
});

export const describirFrescura = (frescura: Frescura): string => {
  const textos: Record<Frescura, string> = {
    actual: 'Datos actuales',
    obsoleto: 'Datos obsoletos: la fuente no se actualiza',
    sin_datos: 'Sin datos de mercado',
  };
  return textos[frescura];
};
