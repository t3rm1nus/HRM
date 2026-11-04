import { renderHook, waitFor, act } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { ReactNode } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { usarMercado } from '../useMercado';
import { fetchEstadoMercado, fetchVelas } from '../../api/mercado';
import type { EventoEnVivo } from '../useEventosEnVivo';

vi.mock('../../api/mercado', () => ({ fetchEstadoMercado: vi.fn(), fetchVelas: vi.fn() }));

let configuracionWs: { onEvent?: (e: EventoEnVivo) => void; onConnect?: () => void } = {};
vi.mock('../useEventosEnVivo', () => ({
  obtenerURLWS: () => 'ws://prueba',
  usarEventosEnVivo: (config: typeof configuracionWs) => {
    configuracionWs = config;
    return { estado: 'conectado' };
  },
}));

const vela = (minuto: number, cierre: number) => ({
  instante: new Date(Date.UTC(2026, 10, 4, 10, minuto)).toISOString(),
  apertura: 1, maximo: 2, minimo: 0.5, cierre, volumen: 3,
});

const estado = { disponible: true, frescura: 'actual', fuente: 'realtime', edad_segundos: 3,
  validez_segundos: 30, simbolos: ['BTCUSDT'], modo: 'paper' } as const;

const crearEnvolvedor = () => {
  const cliente = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={cliente}>{children}</QueryClientProvider>
  );
};

describe('usarMercado', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(fetchEstadoMercado).mockResolvedValue({ ...estado, simbolos: [...estado.simbolos] });
    vi.mocked(fetchVelas).mockResolvedValue({
      simbolo: 'BTCUSDT', cantidad: 1, frescura: 'actual', velas: [vela(0, 10)],
    });
  });

  it('carga estado y velas por REST', async () => {
    const { result } = renderHook(() => usarMercado('BTCUSDT'), { wrapper: crearEnvolvedor() });
    await waitFor(() => expect(result.current.velas.data?.velas).toHaveLength(1));
    expect(result.current.estado.data?.simbolos).toEqual(['BTCUSDT']);
  });

  it('no pide velas mientras no hay símbolo', () => {
    renderHook(() => usarMercado(''), { wrapper: crearEnvolvedor() });
    expect(fetchVelas).not.toHaveBeenCalled();
  });

  it('aplica una actualización del WebSocket sin recargar', async () => {
    const { result } = renderHook(() => usarMercado('BTCUSDT'), { wrapper: crearEnvolvedor() });
    await waitFor(() => expect(result.current.velas.data).toBeDefined());
    act(() => configuracionWs.onEvent?.({
      tipo: 'mercado', timestamp: '', datos: { frescura: 'obsoleto', ultimas_velas: { BTCUSDT: vela(1, 20) } },
    }));
    await waitFor(() => expect(result.current.velas.data?.velas).toHaveLength(2));
    expect(result.current.estado.data?.frescura).toBe('obsoleto');
  });

  it('rellena los símbolos si el estado inicial llegó vacío', async () => {
    vi.mocked(fetchEstadoMercado).mockResolvedValue({ ...estado, disponible: false, frescura: 'sin_datos', simbolos: [] });
    const { result } = renderHook(() => usarMercado('BTCUSDT'), { wrapper: crearEnvolvedor() });
    await waitFor(() => expect(result.current.estado.data).toBeDefined());
    act(() => configuracionWs.onEvent?.({
      tipo: 'mercado', timestamp: '', datos: { frescura: 'actual', ultimas_velas: { BTCUSDT: vela(1, 20) } },
    }));
    await waitFor(() => expect(result.current.estado.data?.simbolos).toEqual(['BTCUSDT']));
  });

  it('ignora eventos de otro tipo o malformados', async () => {
    const { result } = renderHook(() => usarMercado('BTCUSDT'), { wrapper: crearEnvolvedor() });
    await waitFor(() => expect(result.current.velas.data).toBeDefined());
    act(() => {
      configuracionWs.onEvent?.({ tipo: 'estado', timestamp: '', datos: {} });
      configuracionWs.onEvent?.({ tipo: 'mercado', timestamp: '', datos: { basura: true } });
    });
    expect(result.current.velas.data?.velas).toHaveLength(1);
  });

  it('recarga los datos al reconectar el WebSocket', async () => {
    const { result } = renderHook(() => usarMercado('BTCUSDT'), { wrapper: crearEnvolvedor() });
    await waitFor(() => expect(result.current.velas.data).toBeDefined());
    const llamadas = vi.mocked(fetchVelas).mock.calls.length;
    act(() => configuracionWs.onConnect?.());
    await waitFor(() => expect(vi.mocked(fetchVelas).mock.calls.length).toBeGreaterThan(llamadas));
  });

  it('expone el error cuando el backend responde 503', async () => {
    vi.mocked(fetchVelas).mockRejectedValue(new Error('El gestor de mercado aún no está inicializado'));
    const { result } = renderHook(() => usarMercado('BTCUSDT'), { wrapper: crearEnvolvedor() });
    await waitFor(() => expect(result.current.velas.error).toBeInstanceOf(Error));
  });
});
