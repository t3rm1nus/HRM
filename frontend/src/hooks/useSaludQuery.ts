import { useQuery } from '@tanstack/react-query';
import { fetchHealth } from '../api/usuarios';
import { TIEMPO_REFRESCO_HEALTH } from './tiempos';

export const usarSalud = (): {
  datos: HealthCheckRespuesta | undefined;
  error: Error | null;
  isLoading: boolean;
  refetch: () => void;
} => {
  return useQuery({
    queryKey: ['health'],
    queryFn: fetchHealth,
    refetchInterval: TIEMPO_REFRESCO_HEALTH,
    staleTime: TIEMPO_REFRESCO_HEALTH,
    retry: 2,
  });
};

export interface HealthCheckRespuesta {
  status: 'ok' | 'error';
  timestamp: string;
  version: string;
}