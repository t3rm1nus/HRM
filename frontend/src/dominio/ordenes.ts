import { z } from 'zod';

export const TAMANO_PAGINA_POR_DEFECTO = 50;
export const TAMANO_PAGINA_MAXIMO = 200;

export const esquemaLado = z.enum(['buy', 'sell', 'hold']);
export const esquemaEstadoOrden = z.enum(['filled', 'partial', 'canceled', 'rejected']);

export const esquemaMetadataOrden = z.object({
  ciclo: z.union([z.string(), z.number()]).nullable().optional(),
  motivo: z.string().nullable().optional(),
  condiciones_mercado: z.record(z.string(), z.unknown()).optional(),
});

export const esquemaOrden = z.object({
  order_id: z.string(),
  simbolo: z.string(),
  lado: esquemaLado,
  cantidad: z.number(),
  precio_llenado: z.number(),
  comision: z.number(),
  status: esquemaEstadoOrden,
  timestamp: z.string(),
  estrategia: z.string().nullable().optional(),
  metadata: esquemaMetadataOrden.nullable().optional(),
});

export const esquemaLoteOrdenes = z.object({
  total: z.number().int().nonnegative(),
  ordenes: z.array(esquemaOrden),
  has_more: z.boolean(),
  pagina: z.number().int().positive(),
  pagina_total: z.number().int().positive(),
});

export type LadoOrden = z.infer<typeof esquemaLado>;
export type EstadoOrden = z.infer<typeof esquemaEstadoOrden>;
export type Orden = z.infer<typeof esquemaOrden>;
export type LoteOrdenes = z.infer<typeof esquemaLoteOrdenes>;

export type LadoFiltrable = Exclude<LadoOrden, 'hold'>;

export interface FiltrosOrdenes {
  pagina: number;
  tamano: number;
  simbolo: string;
  lado: LadoFiltrable | '';
}

export const FILTROS_ORDENES_INICIALES: FiltrosOrdenes = {
  pagina: 1,
  tamano: TAMANO_PAGINA_POR_DEFECTO,
  simbolo: '',
  lado: '',
};

export const construirParametrosConsulta = (filtros: FiltrosOrdenes): string => {
  const parametros = new URLSearchParams({
    pagina: String(filtros.pagina),
    tamano: String(filtros.tamano),
  });
  const simbolo = filtros.simbolo.trim();
  if (simbolo) parametros.set('simbolo', simbolo);
  if (filtros.lado) parametros.set('lado', filtros.lado);
  return parametros.toString();
};

export const cambiarFiltros = (
  filtros: FiltrosOrdenes,
  cambios: Partial<Pick<FiltrosOrdenes, 'simbolo' | 'lado'>>,
): FiltrosOrdenes => ({ ...filtros, ...cambios, pagina: 1 });

export const irAPagina = (filtros: FiltrosOrdenes, pagina: number): FiltrosOrdenes => ({
  ...filtros,
  pagina: Math.max(1, pagina),
});

const ETIQUETAS_LADO: Record<LadoOrden, string> = { buy: 'Compra', sell: 'Venta', hold: 'Mantener' };
const CLASES_LADO: Record<LadoOrden, string> = {
  buy: 'bg-green-100 text-green-800',
  sell: 'bg-red-100 text-red-800',
  hold: 'bg-gray-100 text-gray-800',
};

const ETIQUETAS_ESTADO: Record<EstadoOrden, string> = {
  filled: 'Ejecutada',
  partial: 'Parcial',
  canceled: 'Cancelada',
  rejected: 'Rechazada',
};

export const etiquetaLado = (lado: LadoOrden): string => ETIQUETAS_LADO[lado];
export const claseLado = (lado: LadoOrden): string => CLASES_LADO[lado];
export const etiquetaEstadoOrden = (estado: EstadoOrden): string => ETIQUETAS_ESTADO[estado];

export const formatearPrecioOrden = (valor: number): string =>
  valor.toLocaleString('es-ES', { minimumFractionDigits: 2, maximumFractionDigits: 2 });

export const formatearCantidadOrden = (valor: number): string =>
  valor.toLocaleString('es-ES', { minimumFractionDigits: 0, maximumFractionDigits: 8 });

/** El logger guarda la hora sin zona horaria: se muestra tal cual, sin convertirla. */
export const formatearFechaOrden = (marcaTemporal: string): string => {
  const fecha = new Date(marcaTemporal);
  return Number.isNaN(fecha.getTime())
    ? marcaTemporal
    : fecha.toLocaleString('es-ES', { dateStyle: 'short', timeStyle: 'medium' });
};

export const describirRango = (lote: LoteOrdenes): string =>
  `Página ${lote.pagina} de ${lote.pagina_total} · ${lote.total} órdenes`;
