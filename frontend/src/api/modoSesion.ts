import { obtenerApiClient } from './clienteApi';
import { APIS, esquemaEstadoModoSesion, EsModoSesion } from '../dominio';

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