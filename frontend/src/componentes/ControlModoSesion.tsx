import React from 'react';
import { PALABRAS_VERIFICACION } from '../dominio/modo';
import { useCambioModoSesion } from '../hooks/useCambioModoSesion';
import { useReinicioDatosPaper } from '../hooks/useReinicioDatosPaper';
import { ModalConfirmacionFuerte } from './modaConfirmacionFuerte';
import { SwitchModoSesion } from './ui/SwitchModoSesion';

const TEXTO_CONFIRMACION_LIMPIA =
  'En el próximo arranque se activará el perfil LIMPIA independiente. Solo se vaciarán los datos de ese perfil; la cartera y el historial PERSISTENTES permanecerán intactos. Escribe CONFIRMAR para continuar.';

const TEXTO_CONFIRMACION_REINICIO =
  'No se borrará nada ahora. En el próximo arranque con el perfil PAPER persistente se eliminarán la cartera, las operaciones y las estadísticas del perfil PERSISTENTE. El perfil LIMPIA, la configuración, las credenciales, los modelos y los logs técnicos no se modificarán. Escribe CONFIRMAR para programarlo.';

export const ControlModoSesion: React.FC = () => {
  const {
    modoConfigurado,
    modoActivo,
    modoOperativo,
    reinicioPendiente,
    reinicioDatosPendiente,
    cargando,
    confirmacionPendiente,
    solicitarCambio,
    confirmarCambio,
    cancelarCambio,
  } = useCambioModoSesion();
  const reinicio = useReinicioDatosPaper();

  const puedeReiniciarDatos =
    modoConfigurado === 'persistente' && modoActivo === 'persistente' && modoOperativo === 'paper';

  return (
    <>
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs font-medium text-gray-500">Sesión</span>
        <SwitchModoSesion
          modoActual={modoConfigurado}
          reinicioPendiente={reinicioPendiente}
          onCambiarModo={solicitarCambio}
          deshabilitado={cargando}
        />
        {puedeReiniciarDatos && (
          <button
            type="button"
            onClick={reinicio.solicitarReinicio}
            disabled={reinicio.reinicioEnCurso || reinicioDatosPendiente}
            className="rounded-lg border border-red-300 px-3 py-1.5 text-xs font-semibold text-red-700 hover:bg-red-50 disabled:cursor-not-allowed disabled:opacity-50"
          >
            Reiniciar datos PAPER
          </button>
        )}
        {reinicioDatosPendiente && (
          <span role="status" className="text-xs font-medium text-amber-700">
            Reinicio PAPER pendiente para el próximo arranque persistente
          </span>
        )}
      </div>

      {reinicio.mensajeError && (
        <p role="alert" className="mt-1 text-xs text-red-600">{reinicio.mensajeError}</p>
      )}

      <ModalConfirmacionFuerte
        activo={confirmacionPendiente}
        verificando={cargando}
        titulo="Cambiar a sesión limpia"
        textoPrimario={TEXTO_CONFIRMACION_LIMPIA}
        palabraEsperada={PALABRAS_VERIFICACION.CONFIRMAR}
        onConfirmar={confirmarCambio}
        onCancelar={cancelarCambio}
      />

      <ModalConfirmacionFuerte
        activo={reinicio.confirmacionPendiente}
        verificando={reinicio.reinicioEnCurso}
        titulo="Reiniciar datos PAPER persistentes"
        textoPrimario={TEXTO_CONFIRMACION_REINICIO}
        palabraEsperada={PALABRAS_VERIFICACION.CONFIRMAR}
        onConfirmar={reinicio.confirmarReinicio}
        onCancelar={reinicio.cancelarReinicio}
      />
    </>
  );
};
