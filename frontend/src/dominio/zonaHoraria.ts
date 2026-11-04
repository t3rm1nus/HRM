export const ZONA_LOCAL = 'local';
export const ZONA_UTC = 'UTC';

export type PreferenciaZona = string;

const ZONAS_SUGERIDAS = [
  'Europe/Madrid', 'Europe/London', 'America/New_York', 'America/Mexico_City',
  'America/Argentina/Buenos_Aires', 'Asia/Tokyo',
];

export const zonaDelNavegador = (): string => Intl.DateTimeFormat().resolvedOptions().timeZone || ZONA_UTC;

const esZonaValida = (zona: string): boolean => {
  try {
    new Intl.DateTimeFormat('es-ES', { timeZone: zona });
    return true;
  } catch {
    return false;
  }
};

export const resolverZona = (preferencia: PreferenciaZona): string => {
  if (preferencia === ZONA_LOCAL) return zonaDelNavegador();
  return esZonaValida(preferencia) ? preferencia : zonaDelNavegador();
};

export const listarZonas = (): string[] => {
  const soportadas = (Intl as unknown as { supportedValuesOf?: (clave: string) => string[] }).supportedValuesOf;
  const todas = soportadas ? soportadas('timeZone') : ZONAS_SUGERIDAS;
  return [...new Set([ZONA_UTC, ...todas])];
};

const formatear = (segundos: number, zona: string, opciones: Intl.DateTimeFormatOptions): string =>
  new Intl.DateTimeFormat('es-ES', { timeZone: zona, hour12: false, ...opciones }).format(new Date(segundos * 1000));

export const formatearFechaHora = (segundos: number, preferencia: PreferenciaZona): string =>
  formatear(segundos, resolverZona(preferencia), {
    day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit',
  });

export type TipoMarca = 'anio' | 'mes' | 'dia' | 'hora';

export const formatearMarcaEje = (segundos: number, tipo: TipoMarca, preferencia: PreferenciaZona): string => {
  const zona = resolverZona(preferencia);
  const formatos: Record<TipoMarca, Intl.DateTimeFormatOptions> = {
    anio: { year: 'numeric' },
    mes: { month: 'short' },
    dia: { day: '2-digit', month: '2-digit' },
    hora: { hour: '2-digit', minute: '2-digit' },
  };
  return formatear(segundos, zona, formatos[tipo]);
};
