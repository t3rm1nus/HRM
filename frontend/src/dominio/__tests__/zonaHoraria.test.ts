import { describe, expect, it } from 'vitest';
import {
  ZONA_LOCAL, ZONA_UTC, formatearFechaHora, formatearMarcaEje, listarZonas, resolverZona, zonaDelNavegador,
} from '../zonaHoraria';

const INSTANTE = Date.UTC(2026, 10, 4, 20, 6) / 1000; // 20:06 UTC

describe('zonaHoraria', () => {
  it('la preferencia automática usa la zona del navegador', () => {
    expect(resolverZona(ZONA_LOCAL)).toBe(zonaDelNavegador());
  });

  it('una zona inválida cae a la del navegador', () => {
    expect(resolverZona('Marte/Olimpo')).toBe(zonaDelNavegador());
  });

  it('UTC conserva la hora UTC', () => {
    expect(formatearFechaHora(INSTANTE, ZONA_UTC)).toContain('20:06');
  });

  it('Madrid suma una hora en noviembre', () => {
    expect(formatearFechaHora(INSTANTE, 'Europe/Madrid')).toContain('21:06');
    expect(formatearMarcaEje(INSTANTE, 'hora', 'Europe/Madrid')).toBe('21:06');
  });

  it('la lista de zonas incluye UTC y no tiene duplicados', () => {
    const zonas = listarZonas();
    expect(zonas).toContain(ZONA_UTC);
    expect(new Set(zonas).size).toBe(zonas.length);
  });
});
