import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { EstadoCarga } from '../estadoCarga';

describe('EstadoCarga', () => {
  it('renderiza el mensaje por defecto', () => {
    render(<EstadoCarga animado={false} />);
    expect(screen.getByText('Cargando...')).toBeInTheDocument();
  });

  it('renderiza un mensaje personalizado', () => {
    render(<EstadoCarga animado={false} mensaje="Cargando datos..." />);
    expect(screen.getByText('Cargando datos...')).toBeInTheDocument();
  });

  it('tiene role="status"', () => {
    render(<EstadoCarga animado={false} />);
    expect(screen.getByRole('status')).toBeInTheDocument();
  });

  it('renderiza diferentes tamaños', () => {
    const { rerender } = render(<EstadoCarga animado={false} tamaño="small" />);
    expect(screen.getByRole('status')).toBeInTheDocument();
    rerender(<EstadoCarga animado={false} tamaño="large" />);
    expect(screen.getByRole('status')).toBeInTheDocument();
  });
});
