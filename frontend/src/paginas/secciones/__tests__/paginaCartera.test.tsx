import { render, screen } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { PaginaCartera } from '../paginaCartera';
import { usarPortafolio } from '../../../hooks/usePortafolioQuery';

vi.mock('../../../hooks/usePortafolioQuery', () => ({ usarPortafolio: vi.fn() }));

const simularHook = (valores: Partial<ReturnType<typeof usarPortafolio>>) =>
  vi.mocked(usarPortafolio).mockReturnValue({
    datos: undefined,
    error: null,
    isLoading: false,
    refetch: vi.fn(),
    ...valores,
  });

describe('PaginaCartera', () => {
  beforeEach(() => vi.clearAllMocks());

  it('muestra el estado de carga', () => {
    simularHook({ isLoading: true });
    render(<PaginaCartera />);
    expect(screen.getByRole('status')).toBeTruthy();
  });

  it('muestra el error cuando el backend falla', () => {
    simularHook({ error: new Error('La cartera aún no está inicializada') });
    render(<PaginaCartera />);
    expect(screen.getByText('No se pudo obtener la cartera')).toBeTruthy();
  });

  it('muestra NAV, balances reales y no inventa lotes', () => {
    simularHook({
      datos: { btc_balance: 0.01, eth_balance: 0.5, usdt_balance: 1500, total_nav: 3300, initial_capital: 3000 },
    });
    render(<PaginaCartera />);
    expect(screen.getAllByText(/3[.,]?300[.,]00 USDT/).length).toBeGreaterThan(0);
    expect(screen.getByText(/^10[.,]00 %$/)).toBeTruthy();
    expect(screen.queryByText(/65000/)).toBeNull();
    expect(screen.queryByText(/balances de respaldo/i)).toBeNull();
  });
});
