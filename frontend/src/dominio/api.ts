import { z } from 'zod';

export type RespuestaAPI<T> = {
  datos: T;
  origen: 'API' | 'WS' | 'ARCHIVO';
  generado_en: string;
  es_sintetico: boolean;
  es_respaldo: boolean;
};

export const crearRespuesta = <T>(
  datos: T,
  opciones: {
    origen?: RespuestaAPI<T>['origen'];
    es_sintetico?: boolean;
    es_respaldo?: boolean;
  } = {},
): RespuestaAPI<T> => ({
  datos,
  origen: opciones.origen ?? 'API',
  generado_en: new Date().toISOString(),
  es_sintetico: opciones.es_sintetico ?? false,
  es_respaldo: opciones.es_respaldo ?? false,
});

export const esquemaHealthCheck = z.object({
  status: z.enum(['ok', 'error']),
  timestamp: z.string(),
  version: z.string(),
});
export const esquemaStatus = z.object({
  status: z.enum(['running', 'paused']),
  trading_active: z.boolean(),
  cycle_count: z.number().int().nonnegative(),
  uptime: z.string(),
  mode: z.enum(['paper', 'live', 'unknown']),
});
export const esquemaPortfolio = z.object({
  btc_balance: z.number().nonnegative(),
  eth_balance: z.number().nonnegative(),
  usdt_balance: z.number().nonnegative(),
  total_nav: z.number().nonnegative(),
  initial_capital: z.number().nonnegative(),
});
export const esquemaVersion = z.object({
  app_version: z.string(),
  framework: z.string(),
  python_version: z.string(),
});
export const esquemaError = z.object({
  detail: z.string(),
  code: z.string().optional(),
  path: z.string().optional(),
  details: z.record(z.string(), z.array(z.string())).optional(),
});

export type HealthCheckResponse = z.infer<typeof esquemaHealthCheck>;
export type StatusResponse = z.infer<typeof esquemaStatus>;
export type PortfolioResponse = z.infer<typeof esquemaPortfolio>;
export type VersionResponse = z.infer<typeof esquemaVersion>;
export type ApiError = {
  message: string;
  statusCode: number;
  code?: string;
  path?: string;
  details?: Record<string, string[]>;
};

export interface HeadersRespuesta {
  'content-type': string;
  'x-request-id': string;
  'x-response-time': string;
}

export const esquemaEstadoModoSesion = z.object({
  modo_activo: z.enum(['persistente', 'limpia']).nullable(),
  modo_configurado: z.enum(['persistente', 'limpia']),
  reinicio_pendiente: z.boolean(),
});

export type EsModoSesion = z.infer<typeof esquemaEstadoModoSesion>;
export type ModoSesion = EsModoSesion['modo_configurado'];

export const CODIGOS_ESTADO_API = {
  BAD_REQUEST: 400,
  UNAUTHORIZED: 401,
  FORBIDDEN: 403,
  NOT_FOUND: 404,
  SERVER_ERROR: 500,
} as const;

const obtenerBaseApi = (): string => {
  const baseConfigurada = import.meta.env.VITE_API_BASE_URL;
  if (baseConfigurada) return baseConfigurada.replace(/\/$/, '');
  if (typeof window !== 'undefined') return window.location.origin;
  return 'http://127.0.0.1:8000';
};

export const APIS = {
  API_BASE: obtenerBaseApi(),
  BASE_URL: obtenerBaseApi(),
  USUARIOS: '/usuarios',
  OPERACIONES: '/operaciones',
  REPORTES: '/reportes',
  MODO_SESION: '/sesion/modo',
} as const;
