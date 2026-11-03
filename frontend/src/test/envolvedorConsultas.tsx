import type { ReactNode } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

const crearClienteDePruebas = (): QueryClient =>
  new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
  });

export const crearEnvolvedorConsultas = () => {
  const cliente = crearClienteDePruebas();

  return ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={cliente}>{children}</QueryClientProvider>
  );
};
