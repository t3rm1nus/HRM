import { render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { Mercado } from '../Mercado';
import { usarMercado } from '../../../hooks/useMercado';

vi.mock('../../../hooks/useMercado', () => ({ usarMercado: vi.fn() }));
vi.mock('../../../componentes/graficas/graficaVelas', () => ({
  GraficaVelas: ({ simbolo }: { simbolo: string }) => <div data-testid="grafica">{simbolo}</div>,
}));

const vela = { instante: '2026-11-04T10:00:00+00:00', apertura: 1, maximo: 2, minimo: 0.5, cierre: 1.5, volumen: 3 };

const montarHook = (estado: object, velas: object, conexion = 'conectado') =>
  vi.mocked(usarMercado).mockReturnValue({ estado, velas, conexion } as never);

describe('Mercado', () => {
  beforeEach(() => vi.clearAllMocks());

  it('avisa de que el motor se está iniciando', () => {
    montarHook(
      { data: { disponible: false, frescura: 'sin_datos', simbolos: [] }, error: null },
      { data: undefined, error: null },
    );
    render(<Mercado />);
    expect(screen.getByText(/se está iniciando/i)).toBeInTheDocument();
    expect(screen.queryByTestId('grafica')).toBeNull();
  });

  it('avisa de datos obsoletos y no los presenta como directo', () => {
    montarHook(
      { data: { disponible: true, frescura: 'obsoleto', simbolos: ['BTCUSDT'] }, error: null },
      { data: { velas: [vela] }, error: null },
    );
    render(<Mercado />);
    expect(screen.getByRole('status')).toHaveTextContent(/obsoletos/i);
    expect(screen.getByText(/no son cotizaciones en directo/i)).toBeInTheDocument();
    expect(screen.getByText(/último cierre conocido/i)).toBeInTheDocument();
  });

  it('muestra la gráfica con datos actuales', () => {
    montarHook(
      { data: { disponible: true, frescura: 'actual', simbolos: ['BTCUSDT'] }, error: null },
      { data: { velas: [vela] }, error: null },
    );
    render(<Mercado />);
    expect(screen.getByTestId('grafica')).toBeInTheDocument();
    expect(screen.getByText(/datos actuales/i)).toBeInTheDocument();
  });

  it('indica cuando no hay WebSocket', () => {
    montarHook(
      { data: { disponible: true, frescura: 'actual', simbolos: [] }, error: null },
      { data: undefined, error: null },
      'desconectado',
    );
    render(<Mercado />);
    expect(screen.getByText(/sin conexión en tiempo real/i)).toBeInTheDocument();
  });

  it('muestra el error del backend', () => {
    montarHook({ data: undefined, error: new Error('fallo de red') }, { data: undefined, error: null });
    render(<Mercado />);
    expect(screen.getByRole('alert')).toHaveTextContent('fallo de red');
  });
});
