import { obtenerApiClient } from './clienteApi';
import {
  esquemaEstadoMercado,
  esquemaLoteVelas,
  type EstadoMercado,
  type LoteVelas,
} from '../dominio/mercado';

const RUTA_MERCADO = '/mercado';
const LIMITE_VELAS_POR_DEFECTO = 500;

export const fetchEstadoMercado = async (): Promise<EstadoMercado> => {
  const respuesta = await obtenerApiClient().get<EstadoMercado>(
    `${RUTA_MERCADO}/estado`,
    esquemaEstadoMercado,
  );
  return respuesta.data;
};

export const fetchVelas = async (
  simbolo: string,
  limite: number = LIMITE_VELAS_POR_DEFECTO,
): Promise<LoteVelas> => {
  const respuesta = await obtenerApiClient().get<LoteVelas>(
    `${RUTA_MERCADO}/velas/${encodeURIComponent(simbolo)}?limite=${limite}`,
    esquemaLoteVelas,
  );
  return respuesta.data;
};
