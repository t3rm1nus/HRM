import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { PaginaOperaciones, MENSAJE_OPERACIONES_SIN_DATOS } from '../paginaOperaciones';

describe('PaginaOperaciones', () => {
  it('indica que no hay datos conectados al backend', () => {
    render(<PaginaOperaciones />);
    expect(screen.getByText(MENSAJE_OPERACIONES_SIN_DATOS)).toBeTruthy();
  });

  it('no muestra filas de ejemplo ni tabla', () => {
    render(<PaginaOperaciones />);
    expect(screen.queryByRole('table')).toBeNull();
    expect(screen.queryByText(/BTCUSDT/)).toBeNull();
  });
});
