import { render, screen, fireEvent } from '@testing-library/react';
import { SwitchModoSesion } from '../SwitchModoSesion';

describe('SwitchModoSesion', () => {
  const mockOnCambiarModo = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renderiza con modo persistente activo', () => {
    render(
      <SwitchModoSesion
        modoActual="persistente"
        reinicioPendiente={false}
        onCambiarModo={mockOnCambiarModo}
      />
    );

    // Verificar que el label tiene role="switch"
    const label = screen.getByRole('switch');
    expect(label).toBeInTheDocument();

    // Verificar aria-checked=true cuando modo es persistente
    expect(label).toHaveAttribute('aria-checked', 'true');
    expect(label).toHaveAttribute('title', expect.stringContaining('nunca toca Persistente'));
  });

  it('renderiza con modo limpia activo', () => {
    render(
      <SwitchModoSesion
        modoActual="limpia"
        reinicioPendiente={false}
        onCambiarModo={mockOnCambiarModo}
      />
    );

    const label = screen.getByRole('switch');
    expect(label).toHaveAttribute('aria-checked', 'false');
  });

  it('muestra el indicador de reinicio pendiente', () => {
    render(
      <SwitchModoSesion
        modoActual="persistente"
        reinicioPendiente={true}
        onCambiarModo={mockOnCambiarModo}
      />
    );

    // El indicador debe estar presente (bolita roja)
    const indicator = screen.getByTestId('reinicio-pendiente');
    expect(indicator).toBeInTheDocument();

    // Texto "Se aplicará al reiniciar"
    const textoReinicio = screen.getByText('Se aplicará al reiniciar');
    expect(textoReinicio).toBeInTheDocument();
  });

  it('cambia el modo al hacer clic en persistente', async () => {
    render(
      <SwitchModoSesion
        modoActual="limpia"
        reinicioPendiente={false}
        onCambiarModo={mockOnCambiarModo}
      />
    );

    const label = screen.getByRole('switch');
    fireEvent.click(label);

    expect(mockOnCambiarModo).toHaveBeenCalledWith('persistente');
  });

  it('cambia el modo al hacer clic en limpia', async () => {
    render(
      <SwitchModoSesion
        modoActual="persistente"
        reinicioPendiente={false}
        onCambiarModo={mockOnCambiarModo}
      />
    );

    const label = screen.getByRole('switch');
    fireEvent.click(label);

    expect(mockOnCambiarModo).toHaveBeenCalledWith('limpia');
  });
});