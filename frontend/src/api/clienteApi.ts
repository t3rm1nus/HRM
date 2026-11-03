import { z } from 'zod';
import {
  esquemaError,
  APIS,
  CODIGOS_ESTADO_API,
  type HeadersRespuesta,
} from '../dominio';

export class ErrorApi extends Error {
  constructor(
    mensaje: string,
    public readonly statusCode: number,
    public readonly code?: string,
    public readonly path?: string,
    public readonly details?: Record<string, string[]>,
  ) {
    super(mensaje);
    this.name = 'ErrorApi';
  }
}

export interface HeadersRequest {
  'content-type'?: string;
  authorization?: string;
  'x-request-id'?: string;
}

export interface ConfigApi {
  baseURL: string;
  authToken: string | null;
  headers: HeadersRequest;
  timeout: number;
}

interface RespuestaCliente<T> {
  data: T;
  headers: HeadersRespuesta;
}

interface FetchApiClient {
  get<T>(endpoint: string, esquema?: z.ZodType<T>): Promise<RespuestaCliente<T>>;
  post<T>(endpoint: string, cuerpo: unknown, esquema?: z.ZodType<T>): Promise<RespuestaCliente<T>>;
}

export const crearConfigApi = (config: Partial<ConfigApi> = {}): ConfigApi => ({
  baseURL: config.baseURL ?? APIS.API_BASE,
  authToken: config.authToken ?? null,
  headers: { 'content-type': 'application/json', ...config.headers },
  timeout: config.timeout ?? 30_000,
});

const obtenerToken = (tokenConfigurado: string | null): string | null => {
  if (tokenConfigurado) return tokenConfigurado;
  if (typeof window === 'undefined') return null;
  return window.localStorage.getItem('token');
};

const obtenerCabeceras = (cabeceras: Headers): HeadersRespuesta => ({
  'content-type': cabeceras.get('content-type') ?? '',
  'x-request-id': cabeceras.get('x-request-id') ?? '',
  'x-response-time': cabeceras.get('x-response-time') ?? '',
});

const convertirError = (error: unknown): ErrorApi => {
  if (error instanceof ErrorApi) return error;
  if (error instanceof DOMException && error.name === 'AbortError') {
    return new ErrorApi('La petición ha superado el tiempo límite', 408, 'TIMEOUT');
  }
  if (error instanceof Error) {
    return new ErrorApi(error.message || 'Error de red', CODIGOS_ESTADO_API.SERVER_ERROR, 'RED');
  }
  return new ErrorApi('Error de red desconocido', CODIGOS_ESTADO_API.SERVER_ERROR, 'RED');
};

const analizarRespuesta = async <T>(
  respuesta: Response,
  esquema?: z.ZodType<T>,
): Promise<RespuestaCliente<T>> => {
  const cabeceras = obtenerCabeceras(respuesta.headers);
  const texto = await respuesta.text();
  const datos: unknown = texto ? JSON.parse(texto) : null;

  if (!respuesta.ok) {
    const errorValidado = esquemaError.safeParse(datos);
    if (errorValidado.success) {
      throw new ErrorApi(
        errorValidado.data.detail,
        respuesta.status,
        errorValidado.data.code,
        errorValidado.data.path,
        errorValidado.data.details,
      );
    }
    throw new ErrorApi(
      `Error HTTP ${respuesta.status}`,
      respuesta.status,
    );
  }

  if (!esquema) return { data: datos as T, headers: cabeceras };
  const resultado = esquema.safeParse(datos);
  if (!resultado.success) {
    const detalles = resultado.error.issues.reduce<Record<string, string[]>>((acumulado, incidencia) => {
      const campo = incidencia.path.join('.') || 'respuesta';
      (acumulado[campo] ??= []).push(incidencia.message);
      return acumulado;
    }, {});
    throw new ErrorApi('La respuesta del servidor no cumple el contrato', 502, 'RESPUESTA_INVALIDA', undefined, detalles);
  }
  return { data: resultado.data, headers: cabeceras };
};

export const obtenerApiClient = (configuracion: Partial<ConfigApi> = {}): FetchApiClient => {
  const config = crearConfigApi(configuracion);
  const token = obtenerToken(config.authToken);

  const solicitar = async <T>(
    metodo: 'GET' | 'POST',
    endpoint: string,
    cuerpo?: unknown,
    esquema?: z.ZodType<T>,
  ): Promise<RespuestaCliente<T>> => {
    const controlador = new AbortController();
    const temporizador = setTimeout(() => controlador.abort(), config.timeout);
    const cabeceras = new Headers(Object.entries(config.headers).filter(([, valor]) => valor !== undefined) as [string, string][]);
    if (token) cabeceras.set('authorization', `Bearer ${token}`);

    try {
      const respuesta = await fetch(`${config.baseURL}${endpoint}`, {
        method: metodo,
        headers: cabeceras,
        signal: controlador.signal,
        ...(metodo === 'POST' ? { body: JSON.stringify(cuerpo) } : {}),
      });
      return await analizarRespuesta(respuesta, esquema);
    } catch (error) {
      throw convertirError(error);
    } finally {
      clearTimeout(temporizador);
    }
  };

  return {
    get: <T>(endpoint: string, esquema?: z.ZodType<T>) => solicitar<T>('GET', endpoint, undefined, esquema),
    post: <T>(endpoint: string, cuerpo: unknown, esquema?: z.ZodType<T>) => solicitar<T>('POST', endpoint, cuerpo, esquema),
  };
};

export const validarRespuesta = <T>(
  respuesta: { data: unknown; headers: HeadersRespuesta },
  esquema: z.ZodType<T>,
): RespuestaCliente<T> => {
  const resultado = esquema.safeParse(respuesta.data);
  if (!resultado.success) {
    throw new ErrorApi(
      'La respuesta del servidor no cumple el contrato',
      CODIGOS_ESTADO_API.BAD_REQUEST,
      'RESPUESTA_INVALIDA',
    );
  }
  return { data: resultado.data, headers: respuesta.headers };
};
