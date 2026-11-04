import { render, screen } from '@testing-library/react';
import { describe, it, expect, beforeEach } from 'vitest';
import { PaginaInicial } from '../paginaInicio';
import { PaginaReportes } from '../paginaReportes';
import { ModoAplicacion, useModoStore } from '../../../estado/storeModo';
import { obtenerEtiquetaModo } from '../../../hooks/useEtiquetaModo';

describe('obtenerEtiquetaModo', () => {
  it('muestra el modo solo cuando el backend lo confirma', () => {
    expect(obtenerEtiquetaModo('papel', true)).toBe('PAPEL');
    expect(obtenerEtiquetaModo('papel', false)).toBe('SIN CONFIRMAR');
  });
});

describe('textos de modo en Inicio y Reportes', () => {
  beforeEach(() => {
    useModoStore.setState({ modo: ModoAplicacion.PAPERA, modoConfirmado: false });
  });

  it('Inicio no afirma un modo sin confirmación del backend', () => {
    render(<PaginaInicial />);
    expect(screen.getByText('SIN CONFIRMAR')).toBeTruthy();
    expect(screen.queryByText('PAPEL')).toBeNull();
  });

  it('Inicio muestra el modo confirmado', () => {
    useModoStore.setState({ modo: ModoAplicacion.LIVE, modoConfirmado: true });
    render(<PaginaInicial />);
    expect(screen.getByText('LIVE')).toBeTruthy();
  });

  it('Reportes no afirma un modo sin confirmación del backend', () => {
    render(<PaginaReportes />);
    expect(screen.getByText('SIN CONFIRMAR')).toBeTruthy();
    expect(screen.queryByText(/en modo/i)).toBeNull();
  });
});
