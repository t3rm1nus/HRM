import * as React from 'react';
import type { ModoSesion } from '../../dominio';

export interface SwitchModoSesionProps {
  modoActual: ModoSesion | null;
  reinicioPendiente: boolean;
  onCambiarModo: (modo: ModoSesion) => void;
  deshabilitado?: boolean;
}

const MODOS: readonly ModoSesion[] = ['persistente', 'limpia'];

const ETIQUETAS: Record<ModoSesion, string> = {
  persistente: 'Persistente',
  limpia: 'Limpia (aislada)',
};

const DESCRIPCION =
  'Persistente conserva su cartera e historial. Limpia usa otro almacenamiento y solo vacía ese perfil al arrancar; nunca toca Persistente.';

const alternarModo = (modo: ModoSesion | null): ModoSesion =>
  modo === 'persistente' ? 'limpia' : 'persistente';

const claseSegmento = (activo: boolean): string =>
  activo ? 'bg-blue-600 text-white shadow' : 'text-gray-600';

export const SwitchModoSesion: React.FC<SwitchModoSesionProps> = ({
  modoActual,
  reinicioPendiente,
  onCambiarModo,
  deshabilitado = false,
}) => (
  <div className="flex items-center gap-2">
    <button
      type="button"
      role="switch"
      aria-checked={modoActual === 'persistente'}
      aria-label="Modo de sesión"
      title={DESCRIPCION}
      disabled={deshabilitado || modoActual === null}
      onClick={() => onCambiarModo(alternarModo(modoActual))}
      className="flex rounded-full bg-gray-200 p-0.5 text-xs font-semibold disabled:cursor-not-allowed disabled:opacity-50"
    >
      {MODOS.map((modo) => (
        <span
          key={modo}
          className={`rounded-full px-3 py-1 transition-colors ${claseSegmento(modo === modoActual)}`}
        >
          {ETIQUETAS[modo]}
        </span>
      ))}
    </button>

    {reinicioPendiente && (
      <>
        <span
          data-testid="reinicio-pendiente"
          aria-label="Reinicio pendiente"
          className="flex h-4 w-4 items-center justify-center rounded-full bg-amber-500 text-xs font-bold text-white"
        >
          !
        </span>
        <span className="text-xs text-gray-500">Se aplicará al reiniciar</span>
      </>
    )}
  </div>
);
