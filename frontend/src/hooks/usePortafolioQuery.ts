import { useQuery } from '@tanstack/react-query';
import { fetchPortfolio } from '../api/usuarios';
import { TIEMPO_REFRESCO_PORTFOLIO } from './tiempos';

export const usarPortafolio = (): {
  datos: PortfolioRespuesta | undefined;
  error: Error | null;
  isLoading: boolean;
  refetch: () => void;
} => {
  const consulta = useQuery({
    queryKey: ['portfolio'],
    queryFn: fetchPortfolio,
    refetchInterval: TIEMPO_REFRESCO_PORTFOLIO,
    staleTime: TIEMPO_REFRESCO_PORTFOLIO,
  });

  return {
    datos: consulta.data,
    error: consulta.error,
    isLoading: consulta.isLoading,
    refetch: () => { void consulta.refetch(); },
  };
};

export interface PortfolioRespuesta {
  btc_balance: number;
  eth_balance: number;
  usdt_balance: number;
  total_nav: number;
  initial_capital: number;
}