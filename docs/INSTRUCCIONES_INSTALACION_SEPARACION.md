# HRM — Separación de almacenamiento PAPER

## Instalación

1. Detén HRM por completo.
2. Haz una copia de seguridad de toda la carpeta del proyecto, especialmente `data/`, `persistent_state/` y `.env`.
3. Descomprime el contenido de este ZIP directamente sobre la raíz de tu HRM original y acepta sustituir los archivos.
4. No sustituyas `.env`, credenciales, configuraciones personales ni claves API.
5. Arranca HRM en PAPER con el perfil PERSISTENTE y comprueba cartera e historial.

## Comportamiento

- `PERSISTENTE` utiliza `data/paper/persistente/`.
- `LIMPIA` utiliza `data/paper/limpia/` y solo vacía ese perfil en cada arranque.
- Volver a PERSISTENTE recupera su cartera e historial sin mezclar los datos de LIMPIA.
- La migración de los archivos antiguos se ejecuta una sola vez. Copia los datos legacy al perfil persistente, conserva los originales y crea `persistent_state/migracion_paper_perfiles_v1.json` para evitar que un reinicio posterior vuelva a importar copias antiguas.
- El botón `Reiniciar datos PAPER` es distinto de cambiar a LIMPIA. Solo aparece en PAPER persistente, exige `CONFIRMAR` y programa el borrado del perfil persistente para el siguiente arranque PAPER persistente. No borra al pulsarlo.
- LIVE no consume el marcador ni borra perfiles PAPER.

## Pruebas focalizadas

Desde la raíz del repositorio:

```bash
PYTHONPATH=. pytest -q tests/test_preparar_sesion.py tests/test_perfil_datos.py tests/api/test_rutas_operaciones.py tests/api/test_rutas_modo_sesion.py
```

