export type RespuestaAPI<T> = {
  datos: T;
  origen: 'API' | 'WS' | 'ARCHIVO';
  generado_en: string;
  es_sintetico: boolean;
  es_respaldo: boolean;
};

export const crearRespuesta = <T>(datos: T, opciones: { origen?: 'API' | 'WS' | 'ARCHIVO'; es_sintetico?: boolean; es_respaldo?: boolean } = {}): RespuestaAPI<T> => ({
  datos,
  origen: opciones.origen ?? 'API',
  generado_en: new Date().toISOString(),
  es_sintetico: opciones.es_sintetico ?? false,
  es_respaldo: opciones.es_respaldo ?? false,
});

// === ESTADO ===
export const esquemaEstado = z.object({
  key: z.string(),
  value: z.string(),
  timestamp: z.string(),
});

export type Estado = z.infer<typeof esquemaEstado>;

// === CARTERA ===
export const esquemaPosicion = z.object({
  activo: z.string(),
  cantidad: z.number(),
  precio_promedio: z.number(),
  valor_actual: z.number(),
  ganancia_pct: z.number(),
});

export type Posicion = z.infer<typeof esquemaPosicion>;

export const esquemaCartera = z.object({
  total_valor: z.number(),
  ganancia_pct: z.number(),
  posiciones: z.array(esquemaPosicion),
});

export type Cartera = z.infer<typeof esquemaCartera>;

// === OPERACIONES ===
export const esquemaOrden = z.object({
  id: z.string(),
  activo: z.string(),
  tipo: z.enum(['COMPRA', 'VENTA']),
  cantidad: z.number(),
  precio: z.number(),
  estado: z.enum(['PENDIENTE', 'EXECUTADA', 'CANCELADA']),
  timestamp: z.string(),
});

export type Orden = z.infer<typeof esquemaOrden>;

// === MERCADO ===
export const esquemaTicker = z.object({
  simbolo: z.string(),
  precio: z.number(),
  variacion_pct: z.number(),
  volumen: z.number(),
});

export type Ticker = z.infer<typeof esquemaTicker>;

// === SENIALES ===
export const esquemaSenal = z.object({
  id: z.string(),
  tipo: z.enum(['COMPRA', 'VENTA', 'SEÑAL_COMPRA', 'SEÑAL_VENTA']),
  activo: z.string(),
  precio: z.number(),
  timestamp: z.string(),
  metadata: z.object({
    fuerza: z.number(),
    confianza: z.number(),
  }),
});

export type Senal = z.infer<typeof esquemaSenal>;

// === ESTRATEGIA ===
export const esquemaEstrategia = z.object({
  id: z.string(),
  nombre: z.string(),
  descripcion: z.string(),
  parametros: z.record(z.unknown()),
  metricas: z.object({
    sharpe: z.number(),
    sortino: z.number(),
    maximo_drawdown: z.number(),
  }),
});

export type Estrategia = z.infer<typeof esquemaEstrategia>;

// === RIESGO ===
export const esquemaRiesgo = z.object({
  valor_at_risk: z.number(),
  volatilidad: z.number(),
  drawdown_maxim: z.number(),
  margem_uso: z.number(),
});

export type Riesgo = z.infer<typeof esquemaRiesgo>;

// === AUTOAPRENDIZAJE ===
export const esquemaAutoAprendizaje = z.object({
  id: z.string(),
  tipo: z.enum['MEJORA', 'ESTRATEGIA_NUEVA'],
  progreso: z.number(),
  ultima_actualizacion: z.string(),
});

export type AutoAprendizaje = z.infer<typeof esquemaAutoAprendizaje>;

// === MODELOS ===
export const esquemaModelo = z.object({
  id: z.string(),
  nombre: z.string(),
  tipo: z.enum['ARIMA', 'PROPHET', 'LSTM'],
  precision: z.number(),
  ultimo_treinamiento: z.string(),
});

export type Modelo = z.infer<typeof esquemaModelo>;

// === BACKTESTS ===
export const esquemaBacktest = z.object({
  id: z.string(),
  nombre: z.string(),
  fecha: z.string(),
  retorno: z.number(),
  maximo_drawdown: z.number(),
  sharpe: z.number(),
});

export type Backtest = z.infer<typeof esquemaBacktest>;

// === FISCAL ===
export const esquemaFiscal = z.object({
  periodo: z.string(),
  ingresos: z.number(),
  egresos: z.number(),
  utilidad_neta: z.number(),
});

export type Fiscal = z.infer<typeof esquemaFiscal>;

// === SALUD ===
export const esquemaSalud = z.object({
  status: z.enum['OK', 'ADVERTENCIA', 'ERROR'],
  uptime: z.string(),
  memoria_uso: z.number(),
  conexiones_activas: z.number(),
});

export type Salud = z.infer<typeof esquemaSalud>;

// === CONFIGURACIÓN ===
export const esquemaConfiguracion = z.object({
  api_base: z.string(),
  websocket_base: z.string(),
  modo: z.enum['CLARO', 'OSCURO'],
  timezone: z.string(),
});

export type Configuracion = z.infer<typeof esquemaConfiguracion>;

// === EVENTOS ===
export const esquemaEvento = z.object({
  id: z.string(),
  tipo: z.string(),
  datos: z.unknown(),
  timestamp: z.string(),
});

export type Evento = z.infer<typeof esquemaEvento>;
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