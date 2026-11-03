import {
  esquemaHealthCheck,
  esquemaStatus,
  esquemaPortfolio,
  esquemaVersion,
  esquemaError,
  HeadersRespuesta,
  ApiError,
  APIS,
  CODIGOS_ESTADO_API,
} from '../dominio';

export class ErrorApi extends Error {
  constructor(
    public readonly message: string,
    public readonly statusCode: number,
    public readonly code?: string,
    public readonly path?: string,
    public readonly details?: Record<string, string[]>
  ) {
    super(message);
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

export const crearConfigApi = (config: Partial<ConfigApi> = {}): ConfigApi => ({
  baseURL: config.baseURL ?? APIS.API_BASE,
  authToken: config.authToken ?? null,
  headers: {
    'content-type': 'application/json',
    ...config.headers,
  },
  timeout: config.timeout ?? 30000,
});

const eventosMap: Map<string, Set<(data: unknown) => void>> = new Map();

export const suscribirEvento = (evento: string, listener: (data: unknown) => void): () => void => {
  if (!eventosMap.has(evento)) {
    eventosMap.set(evento, new Set());
  }
  const listeners = eventosMap.get(evento);
  if (listeners) {
    listeners.add(listener);
  }
  return () => {
    const listenersActual = eventosMap.get(evento);
    if (listenersActual) {
      listenersActual.delete(listener);
      if (listenersActual.size === 0) {
        eventosMap.delete(evento);
      }
    }
  };
};

export const emitirEvento = (evento: string, data: unknown): void => {
  const listeners = eventosMap.get(evento);
  if (listeners) {
    listeners.forEach((listener) => {
      try {
        listener(data);
      } catch (error) {
        console.error('Error al procesar evento:', error);
      }
    });
  }
};

export const obtenerApiClient = (config: Partial<ConfigApi> = {}): FetchApiClient => {
  const apiConfig = crearConfigApi(config);
  const token = apiConfig.authToken ?? localStorage.getItem('token');
  return obtenerFetchClient(config);
};

interface FetchApiClient {
  get<T>(endpoint: string, esquema?: z.ZodType<T>): Promise<{ data: T; headers: HeadersRespuesta }>;
  post<T>(endpoint: string, body: unknown, esquema?: z.ZodType<T>): Promise<{ data: T; headers: HeadersRespuesta }>;
}
const obterFetchClient = (config: Partial<ConfigApi> = {}): FetchApiClient => {
  const apiConfig = criarConfigApi(config);
  const token = apiConfig.authToken ?? localStorage.getItem('token');
  return {
    async get<T>(endpoint: string, esquema?: z.ZodType<T>): Promise<{ data: T; headers: HeadersRespuesta }> {
      const url = \`\${apiConfig.baseURL}\${endpoint}\`;

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), apiConfig.timeout);

      try {
        const response = await fetch(url, {
          method: 'GET',
          headers: {
            'content-type': 'application/json',
            ...(token ? { authorization: \`Bearer \${token}\` } : {}),
            ...apiConfig.headers,
          },
          signal: controller.signal,
        });

        clearTimeout(timeoutId);

        const headers: HeadersRespuesta = {
          'content-type': response.headers['content-type'] ?? '',
          'x-request-id': response.headers['x-request-id'] ?? '',
          'x-response-time': response.headers['x-response-time'] ?? '',
        } as HeadersRespuesta;

        const datos = await response.json();

        if (!response.ok) {
          throw new ErrorApi(
            datos.detail || 'Error en la respuesta',
            response.status,
            undefined,
            undefined,
            undefined
          );
        }

        if (esquema) {
          const parsed = esquema.safeParse(datos);
          if (!parsed.success) {
            throw new ErrorApi(
              'Datos de respuesta inválidos',
              CODIGOS_ESTADO_API.BAD_REQUEST,
              undefined,
              undefined,
              parsed.error.issues.reduce((acc, item) => {
                acc[item.path.join('.')] = [item.message];
                return acc;
              }, {} as Record<string, string[]>)
            );
          }
          return { data: parsed.data, headers };
        }

        return { data: datos as T, headers };
      } catch (error: any) {
        clearTimeout(timeoutId);
        throw obterErrorFetch(error);
      }
    },

    async post<T>(endpoint: string, body: unknown, esquema?: z.ZodType<T>): Promise<{ data: T; headers: HeadersRespuesta }> {
      const url = \`\${apiConfig.baseURL}\${endpoint}\`;

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), apiConfig.timeout);

      try {
        const response = await fetch(url, {
          method: 'POST',
          headers: {
            'content-type': 'application/json',
            ...(token ? { authorization: \`Bearer \${token}\` } : {}),
            ...apiConfig.headers,
          },
          signal: controller.signal,
          body: JSON.stringify(body),
        });

        clearTimeout(timeoutId);

        const headers: HeadersRespuesta = {
          'content-type': response.headers['content-type'] ?? '',
          'x-request-id': response.headers['x-request-id'] ?? '',
          'x-response-time': response.headers['x-response-time'] ?? '',
        } as HeadersRespuesta;

        const datos = await response.json();

        if (!response.ok) {
          throw new ErrorApi(
            datos.detail || 'Error en la respuesta',
            response.status,
            undefined,
            undefined,
            undefined
          );
        }

        if (esquema) {
          const parsed = esquema.safeParse(datos);
          if (!parsed.success) {
            throw new ErrorApi(
              'Datos de respuesta inválidos',
              CODIGOS_ESTADO_API.BAD_REQUEST,
              undefined,
              undefined,
              parsed.error.issues.reduce((acc, item) => {
                acc[item.path.join('.')] = [item.message];
                return acc;
              }, {} as Record<string, string[]>)
            );
          }
          return { data: parsed.data, headers };
        }

        return { data: datos as T, headers };
      } catch (error: any) {
        clearTimeout(timeoutId);
        throw obterErrorFetch(error);
      }
    }
  };
};

const obtenerErrorFetch = (error: any): ErrorApi | ErrorRedes => {
  if (error instanceof Error) {
    if (error.name === 'Error') {
      return new ErrorApi(error.message, CODIGOS_ESTADO_API.SERVER_ERROR, undefined, undefined, undefined);
    }
const mensaje = error.message ?? 'Error de red';
  const código = error.code;

  if (código === 'ECONNABORTED') {
    return new ErrorApi('La petición se ha cancelado por timeout', CODIGOS_ESTADO_API.SERVER_ERROR, 'TIMEOUT');
  }

  if (error.response) {
    const response = error.response;
    const datos = response.data;
    if (typeof datos === 'object' && datos !== null) {
      const parsed = esquemaError.safeParse(datos);
      if (parsed.success) {
        return new ErrorApi(
          parsed.data.detail,
          response.status,
          parsed.data.code,
          parsed.data.path,
          parsed.data.details
        );
      }
    }
    return new ErrorApi(mensaje, response.status, undefined, undefined, undefined);
  }

  return new ErrorApi(mensaje, CODIGOS_ESTADO_API.SERVER_ERROR, 'REDES', undefined, undefined);
};

export const validarRespuesta = <T>(
  respuesta: { data: unknown; headers: HeadersRespuesta },
  esquema: (z.ZodType<T> & { safeParse: (data: unknown) => z.ZodSafeParse<T> })
): { data: T; headers: HeadersRespuesta } => {
  const parsed = esquema.safeParse(respuesta.data);
  if (!parsed.success) {
    throw new ErrorApi(
      'Datos de respuesta inválidos',
      CODIGOS_ESTADO_API.BAD_REQUEST,
      undefined,
      undefined,
      parsed.error.issues.reduce((acc, item) => {
        acc[item.path.join('.')] = [item.message];
        return acc;
      }, {} as Record<string, string[]>)
    );
  }
  const headers = {
    'content-type': respuesta.headers['content-type'] ?? '',
    'x-request-id': respuesta.headers['x-request-id'] ?? '',
    'x-response-time': respuesta.headers['x-response-time'] ?? '',
  } as HeadersRespuesta;
  return { data: parsed.data, headers };
};