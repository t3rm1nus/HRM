import { afterEach, describe, expect, it, vi } from 'vitest';
import { fetchModoSesion } from '../modoSesion';
import { simularFetch } from './test-helper';

describe('fetchModoSesion', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('pide GET /sesion/modo', async () => {
    const funcion = simularFetch({ modo_activo: 'persistente', modo_configurado: 'persistente', reinicio_pendiente: false }, 200);
    await fetchModoSesion();
    expect(String(funcion.mock.calls[0]?.[0])).toMatch(/\/sesion\/modo$/);
  });

  it('devuelve el modo de sesión validado', async () => {
    simularFetch({ modo_activo: 'limpia', modo_configurado: 'persistente', reinicio_pendiente: true }, 200);
    const modo = await fetchModoSesion();
    expect(modo).toEqual({ modo_activo: 'limpia', modo_configurado: 'persistente', reinicio_pendiente: true });
  });

  it('propaga el error del backend', async () => {
    simularFetch({ detail: 'Modo no válido' }, 400);
    await expect(fetchModoSesion()).rejects.toMatchObject({
      code: 'RESPUESTA_INVALIDA',
    });
  });

  it('falla si la respuesta no cumple el contrato', async () => {
    simularFetch({ total: 'x' });
    await expect(fetchModoSesion()).rejects.toMatchObject({
      code: 'RESPUESTA_INVALIDA',
    });
  });
});