import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { CampoFormulario } from '../campoFormulario';

describe('CampoFormulario', () => {
  it('renderiza la etiqueta', () => {
    render(<CampoFormulario label="Nombre" name="nombre" value="" onChange={() => {}} />);
    expect(screen.getByText('Nombre')).toBeInTheDocument();
  });

  it('muestra el asterisco si es requerido', () => {
    render(<CampoFormulario label="Nombre" name="nombre" value="" onChange={() => {}} required />);
    expect(screen.getByText('*')).toBeInTheDocument();
  });

  it('muestra el placeholder', () => {
    render(<CampoFormulario label="Nombre" name="nombre" value="" onChange={() => {}} placeholder="Escribe tu nombre" />);
    expect(screen.getByPlaceholderText('Escribe tu nombre')).toBeInTheDocument();
  });

  it('muestra el error si se proporciona', () => {
    render(<CampoFormulario label="Nombre" name="nombre" value="" onChange={() => {}} error="Campo requerido" />);
    expect(screen.getByText('Campo requerido')).toBeInTheDocument();
  });

  it('llama a onChange cuando cambia el valor', () => {
    const onChange = vi.fn();
    render(<CampoFormulario label="Nombre" name="nombre" value="" onChange={onChange} />);
    const input = screen.getByRole('textbox');
    fireEvent.change(input, { target: { value: 'Juan' } });
    expect(onChange).toHaveBeenCalledWith('Juan');
  });
});

