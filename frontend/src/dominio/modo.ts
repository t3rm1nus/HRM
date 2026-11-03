/**
 * Tipos del sistema de modos de operación.
 *
 * PAPER: entorno de simulación sin capital real (verde)
 * TESTNET: entorno de pruebas con capital simulado (ámbar)
 * LIVE: entorno de producción con capital real (rojo)
 *
 * El modo determina la visibilidad del banner fijo, el color de la insignia
 * y el nivel de restricción de acciones peligrosas.
 */

export enum ModoAplicacion {
  PAPERA = 'papel',
  TESTNET = 'testnet',
  LIVE = 'live',
}

export type ModoValor = `${ModoAplicacion}`;

export interface ModoConfig {
  valor: ModoAplicacion;
  etiqueta: string;
  color: string;
  fondo: string;
  esProduccion: boolean;
}

/** Mapeo de cada modo a su configuración visual y de comportamiento. */
export const MODOS_CONFIG: Record<ModoAplicacion, ModoConfig> = {
  [ModoAplicacion.PAPERA]: {
    valor: ModoAplicacion.PAPERA,
    etiqueta: 'PAPER',
    color: '#16A34A',
    fondo: 'bg-green-50',
    esProduccion: false,
  },
  [ModoAplicacion.TESTNET]: {
    valor: ModoAplicacion.TESTNET,
    etiqueta: 'TESTNET',
    color: '#F59E0B',
    fondo: 'bg-amber-50',
    esProduccion: false,
  },
  [ModoAplicacion.LIVE]: {
    valor: ModoAplicacion.LIVE,
    etiqueta: 'LIVE',
    color: '#DC2626',
    fondo: 'bg-red-50',
    esProduccion: true,
  },
} as const;

/** Constante única para el banner activo (visualmente distinto del resto). */
export const BANNER_LIVE_ID = 'banner-fijo-vivo';

/** Palabras de verificación exigidas por el modal de confirmación fuerte. */
export const PALABRAS_VERIFICACION = {
  LIVE: 'LIVE',
  CONFIRMAR: 'CONFIRMAR',
} as const;

export type PalabraVerificacion = typeof PALABRAS_VERIFICACION[keyof typeof PALABRAS_VERIFICACION];
