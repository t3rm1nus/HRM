import { useQuery } from '@tanstack/react-query';
import { fetchVersion } from '../api/usuarios';
import { TIEMPO_REFRESCO_VERSION } from './tiempos';

export const usarVersion = (): {
  datos: VersionRespuesta | undefined;
  error: Error | null;
  isLoading: boolean;
  refetch: () => void;
} => {
  const consulta = useQuery({
    queryKey: ['version'],
    queryFn: fetchVersion,
    refetchInterval: TIEMPO_REFRESCO_VERSION,
    staleTime: TIEMPO_REFRESCO_VERSION,
  });

  return {
    datos: consulta.data,
    error: consulta.error,
    isLoading: consulta.isLoading,
    refetch: () => { void consulta.refetch(); },
  };
};

export interface VersionRespuesta {
  app_version: string;
  framework: string;
  python_version: string;
}