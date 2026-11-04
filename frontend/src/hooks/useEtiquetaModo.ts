import { useModoStore } from '../estado/storeModo';

export const ETIQUETA_MODO_SIN_CONFIRMAR = 'SIN CONFIRMAR';

export const obtenerEtiquetaModo = (modo: string, modoConfirmado: boolean): string =>
  modoConfirmado ? modo.toUpperCase() : ETIQUETA_MODO_SIN_CONFIRMAR;

export const usarEtiquetaModo = (): string => {
  const modo = useModoStore((estado) => estado.modo);
  const modoConfirmado = useModoStore((estado) => estado.modoConfirmado);
  return obtenerEtiquetaModo(modo, modoConfirmado);
};
