import { z } from 'zod';
import {
  TipoOperacion,
  EstadoOperacion,
  Rol,
  TipoMoneda,
  TipoFecha,
} from './tipos';

export const escalarString = z
  .string()
  .min(1, 'El valor no puede estar vacío')
  .max(255, 'El valor no puede exceder 255 caracteres');

export const escalarNumero = z
  .number()
  .positive('El número debe ser positivo')
  .finite('El valor no puede ser infinito');

export const escalarBooleano = z.boolean();

export const escalarArray = <T>(escalar: z.ZodType<T>) =>
  z.array(escalar).max(1000, 'El array no puede exceder 1000 elementos');

export const escalarOpcion = <T extends string>(opciones: T[]) =>
  z.enum(opciones as [string, ...string[]], `Debe ser uno de: ${opciones.join(', ')}`);

export const validarCorreo = z.string().email('Debe ser un correo válido');

export const esquemaOperacion = z.object({
  id: z.string().uuid('ID inválido'),
  usuario: z.object({
    id: z.string().uuid(),
    nombre: escalarString,
    email: validarCorreo,
    rol: z.nativeEnum(Rol),
    activo: escalarBooleano,
  }),
  tipo: z.nativeEnum(TipoOperacion),
  monto: escalarNumero,
  fecha: z.string().datetime(),
  estado: z.nativeEnum(EstadoOperacion),
  descripcion: escalarString.optional(),
});

export type EsquemaOperacion = z.infer<typeof esquemaOperacion>;

export const esquemaCrearOperacion = z.object({
  usuarioId: z.string().uuid(),
  tipo: z.nativeEnum(TipoOperacion),
  monto: escalarNumero,
  descripcion: escalarString.optional(),
});

export type EsquemaCrearOperacion = z.infer<typeof esquemaCrearOperacion>;

export const esquemaUsuario = z.object({
  id: z.string().uuid(),
  nombre: escalarString,
  email: validarCorreo,
  rol: z.nativeEnum(Rol),
  activo: escalarBooleano,
});

export type EsquemaUsuario = z.infer<typeof esquemaUsuario>;

export const esquemaReportes = z.object({
  totalOperaciones: escalarNumero,
  operacionesExitosas: escalarNumero,
  montoTotal: escalarNumero,
  operacionesPorDia: z.array(
    z.object({
      fecha: z.string().datetime(),
      monto: escalarNumero,
      count: escalarNumero,
    })
  ),
});

export type EsquemaReportes = z.infer<typeof esquemaReportes>;

export const validarEstados = (datos: unknown): datos is EsquemaOperacion => {
  return esquemaOperacion.safeParse(datos).success;
};

export const validarCrearOperacion = (datos: unknown): datos is EsquemaCrearOperacion => {
  return esquemaCrearOperacion.safeParse(datos).success;
};