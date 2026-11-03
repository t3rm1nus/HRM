import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { EstadoError } from '../estadoError';
import { EstadoVacio } from '../estadoVacio';
import { Semaforo } from '../semaforo';
import { InsigniaOrigen } from '../insigniaOrigen';
import { TarjetaKpi } from '../tarjetaKpi';

describe('EstadoError', () => {
  it('renderiza el mensaje de error', () => {
    render(<EstadoError mensaje="Error al cargar" />);
    expect(screen.getByText('Error al cargar')).toBeInTheDocument();
  });

  it('renderiza el detalle si se proporciona', () => {
    render(<EstadoError mensaje="Error" detalle="Detalle del error" />);
    expect(screen.getByText('Detalle del error')).toBeInTheDocument();
  });

  it('tiene role="alert"', () => {
    render(<EstadoError mensaje="Error" />);
    expect(screen.getByRole('alert')).toBeInTheDocument();
  });

  it('muestra botón de reintentar', () => {
    render(<EstadoError mensaje="Error" onReintentar={() => {}} />);
    expect(screen.getByText('Reintentar')).toBeInTheDocument();
  });
});

describe('EstadoVacio', () => {
  it('renderiza el mensaje', () => {
    render(<EstadoVacio mensaje="No hay datos" />);
    expect(screen.getByText('No hay datos')).toBeInTheDocument();
  });

  it('renderiza el detalle si se proporciona', () => {
    render(<EstadoVacio mensaje="Vacío" detalle="Detalle del vacío" />);
    expect(screen.getByText('Detalle del vacío')).toBeInTheDocument();
  });

  it('muestra botón de acción', () => {
    render(<EstadoVacio mensaje="Vacío" accion={{ texto: 'Crear', onClick: () => {} }} />);
    expect(screen.getByText('Crear')).toBeInTheDocument();
  });

  it('tiene role="status"', () => {
    render(<EstadoVacio mensaje="Vacío" />);
    expect(screen.getByRole('status')).toBeInTheDocument();
  });
});

describe('Semaforo', () => {
  it('renderiza con estado verde', () => {
    render(<Semaforo estado="verde" etiqueta="OK" />);
    expect(screen.getByText('OK')).toBeInTheDocument();
  });

  it('renderiza con estado rojo', () => {
    render(<Semaforo estado="rojo" etiqueta="Error" />);
    expect(screen.getByText('Error')).toBeInTheDocument();
  });

  it('tiene role="status"', () => {
    render(<Semaforo estado="verde" />);
    expect(screen.getByRole('status')).toBeInTheDocument();
  });
});

describe('InsigniaOrigen', () => {
  it('muestra texto SINTÉTICO', () => {
    render(<InsigniaOrigen tipo="sintetico" origen="API" antiguedad="1h" />);
    expect(screen.getByText('SINTÉTICO')).toBeInTheDocument();
  });

  it('muestra texto RESPALDO', () => {
    render(<InsigniaOrigen tipo="respaldo" origen="BD" antiguedad="1d" />);
    expect(screen.getByText('RESPALDO')).toBeInTheDocument();
  });

  it('muestra texto OBSOLETO', () => {
    render(<InsigniaOrigen tipo="obsoleto" origen="Cache" antiguedad="5d" />);
    expect(screen.getByText('OBSOLETO')).toBeInTheDocument();
  });
});

describe('TarjetaKpi', () => {
  it('renderiza el título', () => {
    render(<TarjetaKpi titulo="Ventas" valor={100} variacion={10} tendencia="alza" />);
    expect(screen.getByText('Ventas')).toBeInTheDocument();
  });

  it('renderiza el valor', () => {
    render(<TarjetaKpi titulo="Ventas" valor={100} variacion={10} tendencia="alza" />);
    expect(screen.getByText('100')).toBeInTheDocument();
  });

  it('tiene role="region"', () => {
    render(<TarjetaKpi titulo="Ventas" valor={100} variacion={10} tendencia="alza" />);
    expect(screen.getByRole('region')).toBeInTheDocument();
  });
});
