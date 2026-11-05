import { obtenerApiClient } from './clienteApi';
import { APIS, esquemaEstadoModoSesion } from '../dominio';
import type { EsModoSesion } from '../dominio';

export const fetchModoSesion = async (): Promise<EsModoSesion> => {
  const respuesta = await obtenerApiClient().get<EsModoSesion>(
    APIS.MODO_SESION,
    esquemaEstadoModoSesion,
  );
  return respuesta.data;
};

export const cambiarModoSesion = async (modo: 'persistente' | 'limpia'): Promise<EsModoSesion> => {
  const respuesta = await obtenerApiClient().post<EsModoSesion>(
    APIS.MODO_SESION,
    { modo },
    esquemaEstadoModoSesion,
  );
  return respuesta.data;
};

export interface RespuestaReinicioDatosPaper {
  estado: 'pendiente';
  mensaje: string;
}

export const reiniciarDatosPaper = async (confirmacion: string): Promise<RespuestaReinicioDatosPaper> => {
  const respuesta = await obtenerApiClient().post<RespuestaReinicioDatosPaper>(
    APIS.REINICIAR_DATOS_PAPER,
    { confirmacion },
  );
  return respuesta.data;
};
