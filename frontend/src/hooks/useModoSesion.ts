import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { fetchModoSesion, cambiarModoSesion } from '../api/modoSesion';
import { EsModoSesion } from '../dominio';

export interface UseModoSesionResult {
  modoActual: EsModoSesion | undefined;
  cargando: boolean;
  error: Error | null;
  cambiarModo: (modo: 'persistente' | 'limpia') => Promise<void>;
  refetch: () => void;
}

export const useModoSesion = (): UseModoSesionResult => {
  const queryClient = useQueryClient();

  const { data, isLoading, error, refetch } = useQuery<EsModoSesion, Error>({
    queryKey: ['modo-sesion'],
    queryFn: fetchModoSesion,
    staleTime: 60_000,
  });

  const { mutateAsync: cambiarModo, isLoading: mutationIsLoading } = useMutation<
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
    cargando: isLoading || mutationIsLoading,
    error,
    cambiarModo: cambiarModoWrapper,
    refetch,
  };
};