import { keepPreviousData, useQuery } from '@tanstack/react-query';
import { fetchOperaciones } from '../api/operaciones';
import type { FiltrosOrdenes, LoteOrdenes } from '../dominio';
import { TIEMPO_REFRESCO_OPERACIONES } from './tiempos';

export const usarOperaciones = (
  filtros: FiltrosOrdenes,
): {
  datos: LoteOrdenes | undefined;
  error: Error | null;
  isLoading: boolean;
  isFetching: boolean;
  refetch: () => void;
} => {
  const consulta = useQuery({
    queryKey: ['operaciones', filtros],
    queryFn: () => fetchOperaciones(filtros),
    refetchInterval: TIEMPO_REFRESCO_OPERACIONES,
    staleTime: TIEMPO_REFRESCO_OPERACIONES,
    placeholderData: keepPreviousData,
  });

  return {
    datos: consulta.data,
    error: consulta.error,
    isLoading: consulta.isLoading,
    isFetching: consulta.isFetching,
    refetch: () => { void consulta.refetch(); },
  };
};
