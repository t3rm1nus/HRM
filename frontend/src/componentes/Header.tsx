import React from 'react';
import { Link } from 'react-router-dom';
import { RUTAS } from '../constantes';

interface HeaderProps {
  titulo: string;
  subtitulo: string;
}

export const Header: React.FC<HeaderProps> = ({ titulo, subtitulo }) => {
  const { modo, modoConfirmado } = usarModoStore();
  const {
    modoActual,
    cargando,
    error,
    cambiarModo,
    refetch,
  } = useModoSesion();

  const [mostrandoConfirmacion, setMostrandoConfirmacion] = React.useState(false);
  const [modoSeleccionadoAnterior, setModoSeleccionadoAnterior] =
    React.useState<'persistente' | 'limpia' | null>(null);

  const manejarCambiarModo = async (modo: 'persistente' | 'limpia') => {
    if (modo === 'limpia') {
      // Pedir confirmación antes de cambiar a sesión limpia
      setModoSeleccionadoAnterior(modo);
      setMostrandoConfirmacion(true);
    } else {
      // Cambiar a persistente sin confirmación
      try {
        await cambiarModo(modo);
        await refetch();
      } catch (err: any) {
        // En caso de error, mostrar y dejar en estado anterior
        error && console.error('Error al cambiar modo:', err);
      }
    }
  };

  const confirmarCambio = async () => {
    setMostrandoConfirmacion(false);
    try {
      await cambiarModo('limpia');
      await refetch();
    } catch (err: any) {
      error && console.error('Error al confirmar cambio de modo:', err);
      // Volver al estado anterior en caso de error
      if (modoSeleccionadoAnterior) {
        await cambiarModo(modoSeleccionadoAnterior);
        await refetch();
      }
    }
    setModoSeleccionadoAnterior(null);
  };

  const cancelarCambio = () => {
    setMostrandoConfirmacion(false);
    setModoSeleccionadoAnterior(null);
  };

  return (
    <header className="bg-white shadow-sm border-b border-gray-100">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center space-x-4">
            <Link to={RUTAS.INICIO} className="flex items-center space-x-2">
              <span className="text-2xl font-bold text-gray-800">HRM</span>
              <span className="text-sm text-gray-500">Trading</span>
            </Link>
            <div className="h-6 w-px bg-gray-200" />
            <div className="flex items-center space-x-2">
              <span className="text-sm font-medium text-gray-600">{titulo}</span>
              <span className="text-xs text-gray-400">{subtitulo}</span>
            </div>
          </div>
          <div className="flex items-center space-x-4">
            <button
              type="button"
              className="inline-flex items-center px-3 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 bg-white hover:bg-gray-50"
            >
              <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h7" />
              </svg>
              <span className="ml-2">Activar</span>
            </button>

            <SwitchModoSesion
              modoActual={modoActual}
              reinicioPendiente={false}
              onCambiarModo={manejarCambiarModo}
            />

            {/* Modal de confirmación para sesión limpia */}
            <ModalConfirmacionFuerte
              activo={mostrandoConfirmacion}
              titulo="Cambiar modo de sesión"
              textoPrimario="En el próximo arranque se borrarán la cartera y el historial de operaciones."
              palabraEsperada="CONFIRMAR"
              onConfirmar={confirmarCambio}
              onCancelar={cancelarCambio}
            />
          </div>
        </div>
      </div>
    </header>
  );
};

export const BotónPrimario: React.FC<{
  children: React.ReactNode;
  tipo?: 'primary' | 'secundario' | 'success' | 'warning' | 'error';
  onClick?: () => void;
}> = ({ children, tipo = 'primary', onClick }) => {
  const estilos: Record<'primary' | 'secundario' | 'success' | 'warning' | 'error', string> = {
    primary: `bg-blue-600 text-white hover:bg-blue-700`,
    secundario: `bg-gray-200 text-gray-800 hover:bg-gray-300`,
    success: `bg-green-600 text-white hover:bg-green-700`,
    warning: `bg-yellow-500 text-white hover:bg-yellow-600`,
    error: `bg-red-600 text-white hover:bg-red-700`,
  };
  return (
    <button
      type="button"
      className={`inline-flex items-center px-4 py-2 border border-transparent rounded-md text-sm font-medium shadow-sm ${estilos[tipo]} ${onClick ? 'cursor-pointer' : ''}`}
      onClick={onClick}
    >
      {children}
    </button>
  );
};

export const BotonSecundario: React.FC<{
  children: React.ReactNode;
  onClick?: () => void;
}> = ({ children, onClick }) => {
  return (
    <button
      type="button"
      className="inline-flex items-center px-3 py-1.5 border border-gray-300 rounded-md text-sm font-medium text-gray-700 bg-white hover:bg-gray-50"
      onClick={onClick}
    >
      {children}
    </button>
  );
};