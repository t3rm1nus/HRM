import { afterEach, describe, expect, it, vi } from 'vitest';
import { solicitarResetDatosPaper } from '../modoSesion';
import { simularFetch } from './test-helper';

describe('solicitarResetDatosPaper', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('envía la confirmación al endpoint de limpieza diferida', async () => {
    const funcion = simularFetch({
      aceptado: true,
      reset_pendiente: true,
      mensaje: 'Limpieza programada',
    }, 202);

    await solicitarResetDatosPaper();

    expect(String(funcion.mock.calls[0]?.[0])).toMatch(/\/sesion\/resetear-datos$/);
    expect(funcion.mock.calls[0]?.[1]).toEqual(
      expect.objectContaining({ body: JSON.stringify({ confirmacion: 'CONFIRMAR' }) }),
    );
  });

  it('devuelve la confirmación del servidor', async () => {
    simularFetch({
      aceptado: true,
      reset_pendiente: true,
      mensaje: 'Limpieza programada',
    }, 202);

    await expect(solicitarResetDatosPaper()).resolves.toEqual({
      aceptado: true,
      reset_pendiente: true,
      mensaje: 'Limpieza programada',
    });
  });

  it('propaga el rechazo del servidor', async () => {
    simularFetch({ detail: 'Solo PAPER' }, 403);
    await expect(solicitarResetDatosPaper()).rejects.toMatchObject({
      message: 'Solo PAPER',
      statusCode: 403,
    });
  });
});
