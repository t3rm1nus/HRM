import { useQuery } from '@tanstack/react-query';
import { fetchStatus } from '../api/usuarios';
import { TIEMPO_REFRESCO_STATUS } from './tiempos';

export const usarEstado = (): {
  datos: StatusRespuesta | undefined;
  error: Error | null;
  isLoading: boolean;
  refetch: () => void;
} => {
  const consulta = useQuery({
    queryKey: ['status'],
    queryFn: fetchStatus,
    refetchInterval: TIEMPO_REFRESCO_STATUS,
    staleTime: TIEMPO_REFRESCO_STATUS,
  });

  return {
    datos: consulta.data,
    error: consulta.error,
    isLoading: consulta.isLoading,
    refetch: () => { void consulta.refetch(); },
  };
};

export interface StatusRespuesta {
  status: 'running' | 'paused';
  trading_active: boolean;
  cycle_count: number;
  uptime: string;
  mode: 'paper' | 'live' | 'unknown';
}