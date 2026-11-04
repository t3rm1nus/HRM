import { fireEvent, render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { PaginaOperaciones, MENSAJE_OPERACIONES_ERROR, MENSAJE_OPERACIONES_SIN_DATOS } from '../paginaOperaciones';
import { usarOperaciones } from '../../../hooks/useOperacionesQuery';
import { crearLoteOrdenesMock } from '../../../mocks/ordenesMock';

vi.mock('../../../hooks/useOperacionesQuery', () => ({ usarOperaciones: vi.fn() }));

const simularHook = (valores: Partial<ReturnType<typeof usarOperaciones>>) =>
  vi.mocked(usarOperaciones).mockReturnValue({
    datos: undefined,
    error: null,
    isLoading: false,
    isFetching: false,
    refetch: vi.fn(),
    ...valores,
  });

const loteConPaginas = () => crearLoteOrdenesMock({ pagina: 1, tamano: 3, simbolo: null, lado: null });

describe('PaginaOperaciones', () => {
  beforeEach(() => vi.clearAllMocks());

  it('muestra el estado de carga', () => {
    simularHook({ isLoading: true });
    render(<PaginaOperaciones />);
    expect(screen.getByRole('status')).toBeTruthy();
  });

  it('muestra el error del backend y permite reintentar', () => {
    const refetch = vi.fn();
    simularHook({ error: new Error('No se pudo leer el archivo'), refetch });
    render(<PaginaOperaciones />);
    expect(screen.getByText(MENSAJE_OPERACIONES_ERROR)).toBeTruthy();
    fireEvent.click(screen.getByText('Reintentar'));
    expect(refetch).toHaveBeenCalled();
  });

  it('indica que no hay operaciones cuando el lote está vacío', () => {
    simularHook({ datos: crearLoteOrdenesMock({ pagina: 1, tamano: 50, simbolo: 'XRPUSDT', lado: null }) });
    render(<PaginaOperaciones />);
    expect(screen.getByText(MENSAJE_OPERACIONES_SIN_DATOS)).toBeTruthy();
    expect(screen.queryByRole('table')).toBeNull();
  });

  it('muestra las órdenes recibidas y la paginación', () => {
    simularHook({ datos: loteConPaginas() });
    render(<PaginaOperaciones />);
    expect(screen.getAllByRole('row')).toHaveLength(4);
    expect(screen.getByText(/Página 1 de 4/)).toBeTruthy();
    expect(screen.getByText('Anterior')).toBeDisabled();
    expect(screen.getByText('Siguiente')).toBeEnabled();
  });

  it('pide la página siguiente al pulsar Siguiente', () => {
    simularHook({ datos: loteConPaginas() });
    render(<PaginaOperaciones />);
    fireEvent.click(screen.getByText('Siguiente'));
    expect(vi.mocked(usarOperaciones)).toHaveBeenLastCalledWith(expect.objectContaining({ pagina: 2 }));
  });

  it('filtra por lado volviendo a la primera página', () => {
    simularHook({ datos: loteConPaginas() });
    render(<PaginaOperaciones />);
    fireEvent.click(screen.getByText('Siguiente'));
    fireEvent.change(screen.getByLabelText('Lado'), { target: { value: 'sell' } });
    expect(vi.mocked(usarOperaciones)).toHaveBeenLastCalledWith(
      expect.objectContaining({ pagina: 1, lado: 'sell' }),
    );
  });

  it('filtra por símbolo al enviar el formulario', () => {
    simularHook({ datos: loteConPaginas() });
    render(<PaginaOperaciones />);
    fireEvent.change(screen.getByLabelText('Símbolo'), { target: { value: 'ETHUSDT' } });
    fireEvent.click(screen.getByText('Filtrar'));
    expect(vi.mocked(usarOperaciones)).toHaveBeenLastCalledWith(
      expect.objectContaining({ simbolo: 'ETHUSDT' }),
    );
  });
});
