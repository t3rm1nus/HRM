import { afterEach, describe, expect, it, vi } from 'vitest';
import { cambiarModoSesion } from '../modoSesion';
import { simularFetch } from './test-helper';

describe('cambiarModoSesion', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('envía POST /sesion/modo con el modo especificado', async () => {
    const funcion = simularFetch({ modo_activo: 'persistente', modo_configurado: 'persistente', reinicio_pendiente: false, modo_operativo: 'paper', reinicio_datos_pendiente: false }, 200);
    await cambiarModoSesion('limpia');
    expect(String(funcion.mock.calls[0]?.[0])).toMatch(/\/sesion\/modo$/);
    expect(funcion.mock.calls[0]?.[1]).toEqual(expect.objectContaining({ body: JSON.stringify({ modo: 'limpia' }) }));
  });

  it('devuelve el modo de sesión cambiado', async () => {
    simularFetch({ modo_activo: 'persistente', modo_configurado: 'limpia', reinicio_pendiente: false, modo_operativo: 'paper', reinicio_datos_pendiente: false }, 200);
    const modo = await cambiarModoSesion('persistente');
    expect(modo).toEqual({ modo_activo: 'persistente', modo_configurado: 'limpia', reinicio_pendiente: false, modo_operativo: 'paper', reinicio_datos_pendiente: false });
  });

  it('propaga el error del backend', async () => {
    simularFetch({ detail: 'Modo no válido' }, 400);
    await expect(cambiarModoSesion('limpia')).rejects.toMatchObject({
      code: 'RESPUESTA_INVALIDA',
    });
  });

  it('falla si la respuesta no cumple el contrato', async () => {
    simularFetch({ total: 'x' });
    await expect(cambiarModoSesion('limpia')).rejects.toMatchObject({
      code: 'RESPUESTA_INVALIDA',
    });
  });
});
describe('reiniciarDatosPaper', () => {
  it('envía la confirmación al endpoint diferido', async () => {
    const { reiniciarDatosPaper } = await import('../modoSesion');
    const funcion = simularFetch({ estado: 'pendiente', mensaje: 'Se reiniciará al arrancar' }, 202);

    const respuesta = await reiniciarDatosPaper('CONFIRMAR');

    expect(String(funcion.mock.calls[0]?.[0])).toMatch(/\/sesion\/reiniciar-datos$/);
    expect(funcion.mock.calls[0]?.[1]).toEqual(
      expect.objectContaining({ body: JSON.stringify({ confirmacion: 'CONFIRMAR' }) }),
    );
    expect(respuesta.estado).toBe('pendiente');
  });
});
