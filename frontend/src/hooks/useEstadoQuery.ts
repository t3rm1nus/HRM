import { useQuery } from '@tanstack/react-query';
import { fetchStatus } from '../api/usuarios';
import { TIEMPO_REFRESCO_STATUS } from './tiempos';

export const usarEstado = (): {
  datos: StatusRespuesta | undefined;
  error: Error | null;
  isLoading: boolean;
  refetch: () => void;
} => {
  return useQuery({
    queryKey: ['status'],
    queryFn: fetchStatus,
    refetchInterval: TIEMPO_REFRESCO_STATUS,
    staleTime: TIEMPO_REFRESCO_STATUS,
    retry: 2,
  });
};

export interface StatusRespuesta {
  status: 'running' | 'paused';
  trading_active: boolean;
  cycle_count: number;
  uptime: string;
}