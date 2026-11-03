import { ModoAplicacion } from '../dominio/modo';

/** Constantes de configuración de los modos de operación. */

export const MODO_POR_DEFECTO = ModoAplicacion.PAPERA as ModoAplicacion;

export const MODO_NOMBRES = {
  [ModoAplicacion.PAPERA]: 'PAPER',
  [ModoAplicacion.TESTNET]: 'TESTNET',
  [ModoAplicacion.LIVE]: 'LIVE',
} as const;

export const COLOR_ESQUELETO = '#111827';
export const COLOR_BANNER_FONDO = '#DC2626';
export const COLOR_BANNER_TEXTO = '#FFFFFF';
export const COLOR_BANNER_BITACORA = '#FFFFFFCC';
export const COLOR_BANNER_BOTON = '#F97316';
export const COLOR_BANNER_BOTON_TEXTO = '#FFFFFF';
