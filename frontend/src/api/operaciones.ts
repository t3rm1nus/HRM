import { obtenerApiClient } from './clienteApi';
import {
  APIS,
  construirParametrosConsulta,
  esquemaLoteOrdenes,
  type FiltrosOrdenes,
  type LoteOrdenes,
} from '../dominio';

const construirEndpointOperaciones = (filtros: FiltrosOrdenes): string =>
  `${APIS.OPERACIONES}?${construirParametrosConsulta(filtros)}`;

export const fetchOperaciones = async (filtros: FiltrosOrdenes): Promise<LoteOrdenes> => {
  const respuesta = await obtenerApiClient().get<LoteOrdenes>(
    construirEndpointOperaciones(filtros),
    esquemaLoteOrdenes,
  );
  return respuesta.data;
};
