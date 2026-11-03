import { z } from 'zod';
import { TipoOperacion, EstadoOperacion, Rol } from './tipos';

/**
 * Tipos basados en el Contrato de API FastAPI (docs/front/CONTRATO_API.md)
 *
 * Endpoints documentados:
 * - GET /health
 * - GET /status
 * - GET /portfolio
 * - GET /version
 */

export interface HealthCheckResponse {
  status: 'ok' | 'error';
  timestamp: string;
  version: string;
}

export interface StatusResponse {
  status: 'running' | 'paused';
  trading_active: boolean;
  cycle_count: number;
  uptime: string;
}

export interface PortfolioResponse {
  btc_balance: number;
  eth_balance: number;
  usdt_balance: number;
  total_nav: number;
  initial_capital: number;
}

export interface VersionResponse {
  app_version: string;
  framework: string;
  python_version: string;
}

export const esquemaHealthCheck = z.object({
  status: z.enum(['ok', 'error']),
  timestamp: z.string().datetime(),
  version: z.string(),
});

export const esquemaStatus = z.object({
  status: z.enum(['running', 'paused']),
  trading_active: z.boolean(),
  cycle_count: z.number().int().nonnegative(),
  uptime: z.string(),
});

export const esquemaPortfolio = z.object({
  btc_balance: z.number().positive(),
  eth_balance: z.number().positive(),
  usdt_balance: z.number().positive(),
  total_nav: z.number().positive(),
  initial_capital: z.number().positive(),
});

export const esquemaVersion = z.object({
  app_version: z.string(),
  framework: z.string(),
  python_version: z.string(),
});

export type HealthCheckResponse = z.infer<typeof esquemaHealthCheck>;
export type StatusResponse = z.infer<typeof esquemaStatus>;
export type PortfolioResponse = z.infer<typeof esquemaPortfolio>;
export type VersionResponse = z.infer<typeof esquemaVersion>;

export interface ApiErrorResponse {
  detail: string;
  code?: string;
  path?: string;
}

export interface ApiError {
  message: string;
  statusCode: number;
  code?: string;
  path?: string;
  details?: Record<string, string[]>;
}

export const esquemaError = z.object({
  detail: z.string(),
  code: z.string().optional(),
  path: z.string().optional(),
});

export type EsquemaError = z.infer<typeof esquemaError>;

export interface HeadersRespuesta {
  'content-type': string;
  'x-request-id': string;
  'x-response-time': string;
}

export enum MetodoHttp {
  GET = 'GET',
  POST = 'POST',
  PUT = 'PUT',
  DELETE = 'DELETE',
  PATCH = 'PATCH',
}

export interface OpcionesSolicitud {
  metodo: MetodoHttp;
  headers?: Record<string, string>;
  cuerpo?: unknown;
  tiempoEspera?: number;
}

export const CODIGOS_ESTADO_API = {
  SUCCESS: 200,
  CREATED: 201,
  NO_CONTENT: 204,
  BAD_REQUEST: 400,
  UNAUTHORIZED: 401,
  FORBIDDEN: 403,
  NOT_FOUND: 404,
  SERVER_ERROR: 500,
} as const;

export const TIEMPOS_ESPERA = {
  REDUCCION_BASE: 1000,
  MAX_RECHECK: 30000,
  HEARTBEAT: 30000,
  REFRESCO_POR_DEFECTO: 30000,
} as const;