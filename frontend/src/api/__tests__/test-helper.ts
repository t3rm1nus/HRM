export const simularFetch = (cuerpo: unknown, estado = 200) => {
  const funcion = vi.fn().mockResolvedValue(new Response(JSON.stringify(cuerpo), { status: estado }));
  vi.stubGlobal('fetch', funcion);
  return funcion;
};