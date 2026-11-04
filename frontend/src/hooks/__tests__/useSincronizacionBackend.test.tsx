import { renderHook, waitFor } from '@testing-library/react';
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { crearEnvolvedorConsultas } from '../../test/envolvedorConsultas';
import { traducirModoDelBackend, usarSincronizacionBackend } from '../useSincronizacionBackend';
import { useModoStore } from '../../estado/storeModo';
import { fetchStatus } from '../../api/usuarios';

vi.mock('../../api/usuarios', () => ({ fetchStatus: vi.fn() }));

const renderizarSincronizacion = () =>
  renderHook(() => usarSincronizacionBackend(), { wrapper: crearEnvolvedorConsultas() });

describe('usarSincronizacionBackend', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useModoStore.setState({ estadoConexion: 'pendiente', cicloCount: 0, modoConfirmado: false });
  });

  it('refleja conexión y ciclo del backend en el store', async () => {
    vi.mocked(fetchStatus).mockResolvedValue({
      status: 'running', trading_active: true, cycle_count: 42, uptime: '0h 01m 00s', mode: 'paper',
    });

    renderizarSincronizacion();

    await waitFor(() => expect(useModoStore.getState().estadoConexion).toBe('conectado'));
    expect(useModoStore.getState().cicloCount).toBe(42);
    expect(useModoStore.getState().modoConfirmado).toBe(true);
  });

  it('no confirma el modo cuando el backend lo declara desconocido', async () => {
    vi.mocked(fetchStatus).mockResolvedValue({
      status: 'paused', trading_active: false, cycle_count: 0, uptime: 'no disponible', mode: 'unknown',
    });

    renderizarSincronizacion();

    await waitFor(() => expect(useModoStore.getState().estadoConexion).toBe('conectado'));
    expect(useModoStore.getState().modoConfirmado).toBe(false);
  });

  it('marca desconectado cuando el backend no responde', async () => {
    vi.mocked(fetchStatus).mockRejectedValue(new Error('sin conexión'));

    renderizarSincronizacion();

    await waitFor(() => expect(useModoStore.getState().estadoConexion).toBe('desconectado'));
  });
});

describe('traducirModoDelBackend', () => {
  it('traduce paper y live, y rechaza el resto', () => {
    expect(traducirModoDelBackend('paper')).toBe('papel');
    expect(traducirModoDelBackend('live')).toBe('live');
    expect(traducirModoDelBackend('unknown')).toBeNull();
    expect(traducirModoDelBackend(undefined)).toBeNull();
  });
});
