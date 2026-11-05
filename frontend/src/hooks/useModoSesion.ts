import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { fetchModoSesion, cambiarModoSesion } from '../api/modoSesion';
import type { EsModoSesion } from '../dominio';

export interface UseModoSesionResult {
  modoActual: EsModoSesion | undefined;
  cargando: boolean;
  error: Error | null;
  cambiarModo: (modo: 'persistente' | 'limpia') => Promise<void>;
  refetch: () => void;
}

export const useModoSesion = (): UseModoSesionResult => {
  const queryClient = useQueryClient();
  const INTERVALO_REINTENTO_MS = 3_000;

  const { data, isLoading, error, refetch } = useQuery<EsModoSesion, Error>({
    queryKey: ['modo-sesion'],
    queryFn: fetchModoSesion,
    staleTime: 60_000,
    retry: true,
    retryDelay: INTERVALO_REINTENTO_MS,
  });

  const { mutateAsync: cambiarModo, isPending: mutacionEnCurso } = useMutation<
    EsModoSesion,
    Error,
    'persistente' | 'limpia'
  >({
    mutationFn: (modo) => cambiarModoSesion(modo),
    onSuccess: () => {
      // After changing mode, refetch to get the updated state
      queryClient.invalidateQueries({ queryKey: ['modo-sesion'] });
    },
  });

  const cambiarModoWrapper = async (modo: 'persistente' | 'limpia') => {
    await cambiarModo(modo);
  };

  return {
    modoActual: data,
    cargando: isLoading || mutacionEnCurso,
    error,
    cambiarModo: cambiarModoWrapper,
    refetch,
  };
};