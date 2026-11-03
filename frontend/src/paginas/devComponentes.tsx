import React from 'react';
import {
  TarjetaKpi,
  TablaDatos,
  InsigniaOrigen,
  Semaforo,
  EstadoCarga,
  EstadoError,
  EstadoVacio,
  PanelLateralDetalle,
  CampoFormulario,
} from '../componentes/ui';
import { z } from 'zod';

const datosEjemplo = [
  { id: 1, nombre: 'Juan', edad: 30, estado: 'activo' },
  { id: 2, nombre: 'María', edad: 25, estado: 'inactivo' },
  { id: 3, nombre: 'Carlos', edad: 35, estado: 'activo' },
];

const esquemaEmail = z.string().email('Correo inválido');

export const PaginaDevComponentes: React.FC = () => {
  const [panelAbierto, setPanelAbierto] = React.useState(false);
  const [valorCampo, setValorCampo] = React.useState('');

  return (
    <div className="max-w-7xl mx-auto px-4 py-8 space-y-8">
      <h1 className="text-3xl font-bold text-gray-900">Componentes UI</h1>

      {/* TarjetaKpi */}
      <section className="space-y-4">
        <h2 className="text-xl font-semibold text-gray-800">TarjetaKpi</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          <TarjetaKpi titulo="Ventas" valor={1250} variacion={125} tendencia="alza" unidad="USD" />
          <TarjetaKpi titulo="Costos" valor={850} variacion={-50} tendencia="baja" unidad="USD" />
          <TarjetaKpi titulo="Balance" valor={400} variacion={0} tendencia="neutral" unidad="USD" />
        </div>
      </section>

      {/* InsigniaOrigen */}
      <section className="space-y-4">
        <h2 className="text-xl font-semibold text-gray-800">InsigniaOrigen</h2>
        <div className="flex gap-4 flex-wrap">
          <InsigniaOrigen tipo="sintetico" origen="API externa" antiguedad="2 horas" descripcion="Datos generados sintéticamente" />
          <InsigniaOrigen tipo="respaldo" origen="Base de datos local" antiguedad="1 día" descripcion="Datos de respaldo" />
          <InsigniaOrigen tipo="obsoleto" origen="Cache local" antiguedad="5 días" descripcion="Datos obsoletos" />
        </div>
      </section>

      {/* Semaforo */}
      <section className="space-y-4">
        <h2 className="text-xl font-semibold text-gray-800">Semaforo</h2>
        <div className="flex gap-4 flex-wrap items-center">
          <Semaforo estado="verde" etiqueta="Conectado" />
          <Semaforo estado="amarillo" etiqueta="Advertencia" />
          <Semaforo estado="rojo" etiqueta="Error" animado />
          <Semaforo estado="gris" etiqueta="Sin datos" />
        </div>
      </section>

      {/* EstadoCarga */}
      <section className="space-y-4">
        <h2 className="text-xl font-semibold text-gray-800">EstadoCarga</h2>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <EstadoCarga tamaño="small" mensaje="Cargando small" animado={false} />
          <EstadoCarga tamaño="medium" mensaje="Cargando medium" animado={false} />
          <EstadoCarga tamaño="large" mensaje="Cargando large" animado={false} />
        </div>
      </section>
      {/* CampoFormulario */}
      <section className="space-y-4">
        <h2 className="text-xl font-semibold text-gray-800">CampoFormulario</h2>
        <div className="max-w-md space-y-4">
          <CampoFormulario
            label="Correo electrónico"
            name="email"
            value={valorCampo}
            onChange={setValorCampo}
            schema={esquemaEmail}
            type="email"
            placeholder="ejemplo@correo.com"
            required
            ayuda="Ingresa un correo válido"
          />
          <CampoFormulario
            label="Comentarios"
            name="comentarios"
            value=""
            onChange={() => {}}
            type="textarea"
            placeholder="Escribe tus comentarios..."
          />
        </div>
      </section>

      {/* PanelLateralDetalle */}
      <section className="space-y-4">
        <h2 className="text-xl font-semibold text-gray-800">PanelLateralDetalle</h2>
        <button
          type="button"
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          onClick={() => setPanelAbierto(true)}
        >
          Abrir panel lateral
        </button>
        <PanelLateralDetalle
          abierto={panelAbierto}
          onCerrar={() => setPanelAbierto(false)}
          titulo="Detalle del registro"
        >
          <p className="text-gray-600">Contenido del panel lateral.</p>
          <div className="mt-4">
            <InsigniaOrigen tipo="sintetico" origen="API" antiguedad="1 hora" />
          </div>
        </PanelLateralDetalle>
      </section>
      {/* EstadoError */}
      <section className="space-y-4">
        <h2 className="text-xl font-semibold text-gray-800">EstadoError</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <EstadoError mensaje="Error al cargar datos" detalle="No se pudo conectar con el servidor" onReintentar={() => alert('Reintentando...')} />
          <EstadoError mensaje="Error de validacion" onCerrar={() => alert('Cerrado')} />
        </div>
      </section>

      {/* EstadoVacio */}
      <section className="space-y-4">
        <h2 className="text-xl font-semibold text-gray-800">EstadoVacio</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <EstadoVacio mensaje="No hay datos" detalle="No se encontraron registros" accion={{ texto: 'Crear nuevo', onClick: () => alert('Crear') }} />
          <EstadoVacio mensaje="Sin resultados" icono="🔍" />
        </div>
      </section>

      {/* TablaDatos */}
      <section className="space-y-4">
        <h2 className="text-xl font-semibold text-gray-800">TablaDatos</h2>
        <TablaDatos
          datos={datosEjemplo}
          columnas={[
            { clave: 'id', titulo: 'ID', sortable: true },
            { clave: 'nombre', titulo: 'Nombre', sortable: true },
            { clave: 'edad', titulo: 'Edad', sortable: true },
            { clave: 'estado', titulo: 'Estado' },
          ]}
          porPagina={5}
          exportable
          ordenable
          filtrable
        />
      </section>
    </div>
  );
};

export default PaginaDevComponentes;
