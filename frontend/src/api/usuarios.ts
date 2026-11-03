import { obtenerApiClient } from './clienteApi';
import {
  esquemaHealthCheck,
  esquemaStatus,
  esquemaPortfolio,
  esquemaVersion,
} from '../dominio';

export const obtenerApi = () => obtenerApiClient();

export const fetchHealth = async (): Promise<HealthCheckRespuesta> => {
  const cliente = obtenerApi();
  const respuesta = await cliente.get<HealthCheckRespuesta>('/health', esquemaHealthCheck);
  return respuesta.data;
};

export const fetchStatus = async (): Promise<StatusRespuesta> => {
  const cliente = obtenerApi();
  const respuesta = await cliente.get<StatusRespuesta>('/status', esquemaStatus);
  return respuesta.data;
};

export const fetchPortfolio = async (): Promise<PortfolioRespuesta> => {
  const cliente = obtenerApi();
  const respuesta = await cliente.get<PortfolioRespuesta>('/portfolio', esquemaPortfolio);
  return respuesta.data;
};

export const fetchVersion = async (): Promise<VersionRespuesta> => {
  const cliente = obtenerApi();
  const respuesta = await cliente.get<VersionRespuesta>('/version', esquemaVersion);
  return respuesta.data;
};

export interface HealthCheckRespuesta {
  status: 'ok' | 'error';
  timestamp: string;
  version: string;
}

export interface StatusRespuesta {
  status: 'running' | 'paused';
  trading_active: boolean;
  cycle_count: number;
  uptime: string;
}

export interface PortfolioRespuesta {
  btc_balance: number;
  eth_balance: number;
  usdt_balance: number;
  total_nav: number;
  initial_capital: number;
}

export interface VersionRespuesta {
  app_version: string;
  framework: string;
  python_version: string;
}