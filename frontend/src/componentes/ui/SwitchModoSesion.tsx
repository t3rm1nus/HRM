import * as React from 'react';

export interface SwitchModoSesionProps {
  modoActual: 'persistente' | 'limpia' | null;
  reinicioPendiente: boolean;
  onCambiarModo: (modo: 'persistente' | 'limpia') => void;
}

export const SwitchModoSesion: React.FC<SwitchModoSesionProps> = ({
  modoActual,
  reinicioPendiente,
  onCambiarModo,
}) => {
  const [mostrandoReinicio, setMostrandoReinicio] = React.useState(false);

  React.useEffect(() => {
    setMostrandoReinicio(!!reinicioPendiente);
  }, [reinicioPendiente]);

  const modoSeleccionado = modoActual === 'persistente' ? 'persistente' : 'limpia';

  return (
    <div className="relative inline-block">
      {/* Label para accesibilidad */}
      <label
        className="flex w-24 items-center justify-center rounded-full bg-gray-200 p-1"
        role="switch"
        aria-checked={modoActual === 'persistente'}
      >
        {/* Botón Persistente */}
        <button
          key="persistente"
          type="button"
          className="w-1/2 h-5 rounded-full bg-white shadow-md transition-colors duration-200"
          onClick={() => onCambiarModo('persistente')}
          style={{ backgroundColor: modoSeleccionado === 'persistente' ? '#3b82f6' : 'transparent' }}
          aria-label="Modo persistente"
        >
          {/* No contenido necesario, el color indica el estado */}
        </button>

        {/* Botón Sesión Limpia */}
        <button
          key="limpia"
          type="button"
          className="w-1/2 h-5 rounded-full bg-white shadow-md transition-colors duration-200"
          onClick={() => onCambiarModo('limpia')}
          style={{ backgroundColor: modoSeleccionado === 'limpia' ? '#3b82f6' : 'transparent' }}
          aria-label="Sesión limpia"
        >
          {/* No contenido necesario, el color indica el estado */}
        </button>
      </label>

      {/* Indicador de reinicio pendiente */}
      {mostrandoReinicio && (
        <span
          className="absolute -top-1 -right-1 rounded-full bg-red-500 text-xs text-white w-3 h-3"
          aria-label="Reinicio pendiente"
          data-testid="reinicio-pendiente"
        >
          !
        </span>
      )}

      {/* Texto "Se aplicará al reiniciar" */}
      {mostrandoReinicio && (
        <span
          className="absolute right-6 top-1/2 -translate-y-1/2 text-xs text-gray-500"
        >
          Se aplicará al reiniciar
        </span>
      )}
    </div>
  );
};