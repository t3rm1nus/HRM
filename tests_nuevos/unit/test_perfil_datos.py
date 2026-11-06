import json

import pytest

import storage.perfil_datos as perfiles


@pytest.fixture(autouse=True)
def restaurar_perfil_global(monkeypatch):
    monkeypatch.setattr(perfiles, "_PERFIL_ACTIVO", perfiles.PERFIL_PERSISTENTE)


@pytest.mark.unit
def test_configurar_y_obtener_perfil_paper():
    perfiles.configurar_perfil_datos(perfiles.PERFIL_LIMPIO)
    assert perfiles.obtener_perfil_datos() == perfiles.PERFIL_LIMPIO


@pytest.mark.unit
def test_rechaza_perfil_desconocido():
    with pytest.raises(ValueError, match="Perfil de datos no válido"):
        perfiles.configurar_perfil_datos("desconocido")


@pytest.mark.unit
@pytest.mark.parametrize("perfil", ["persistente", "limpia"])
def test_construye_rutas_paper_separadas(perfil, tmp_path):
    assert perfiles.obtener_directorio_paper(perfil, tmp_path) == (
        tmp_path / "data" / "paper" / perfil
    )


@pytest.mark.unit
def test_ruta_live_no_se_mezcla_con_perfiles_paper(tmp_path):
    assert perfiles.obtener_ruta_cartera(raiz=tmp_path, modo="live") == (
        tmp_path / "persistent_state" / "portfolio_state_live.json"
    )


@pytest.mark.unit
def test_limpia_solo_el_perfil_solicitado(tmp_path):
    persistente = perfiles.obtener_directorio_paper("persistente", tmp_path)
    limpia = perfiles.obtener_directorio_paper("limpia", tmp_path)
    persistente.mkdir(parents=True)
    limpia.mkdir(parents=True)
    (persistente / "cartera.json").write_text("persistir", encoding="utf-8")
    (limpia / "cartera.json").write_text("limpiar", encoding="utf-8")

    eliminados = perfiles.limpiar_perfil_paper("limpia", tmp_path)

    assert eliminados == 1
    assert (persistente / "cartera.json").read_text(encoding="utf-8") == "persistir"
    assert limpia.exists()
    assert list(limpia.iterdir()) == []


@pytest.mark.unit
def test_migracion_paper_correcta_en_directorio_temporal(tmp_path):
    estado = tmp_path / "persistent_state" / "portfolio_state_live.json"
    estado.parent.mkdir(parents=True)
    estado.write_text(json.dumps({"mode": "paper", "balance": 250.0}), encoding="utf-8")

    resultado = perfiles.migrar_datos_legacy_a_persistente(tmp_path)
    destino = tmp_path / "data" / "paper" / "persistente" / "portfolio_state.json"
    marcador = tmp_path / "persistent_state" / "migracion_paper_perfiles_v1.json"

    assert resultado["cartera"] == 1
    assert json.loads(destino.read_text(encoding="utf-8"))["balance"] == 250.0
    assert marcador.is_file()


@pytest.mark.unit
@pytest.mark.regression
@pytest.mark.xfail(strict=True, reason="Defecto conocido: puede crearse el marcador aunque falle una copia.")
def test_migracion_no_debe_marcarse_completada_si_falla_una_copia(tmp_path, monkeypatch):
    estado = tmp_path / "persistent_state" / "portfolio_state_live.json"
    estado.parent.mkdir(parents=True)
    estado.write_text(json.dumps({"mode": "paper", "balance": 250.0}), encoding="utf-8")

    def fallar_copia(*args, **kwargs):
        raise OSError("fallo de copia simulado")

    monkeypatch.setattr(perfiles.shutil, "copy2", fallar_copia)
    perfiles.migrar_datos_legacy_a_persistente(tmp_path)

    marcador = tmp_path / "persistent_state" / "migracion_paper_perfiles_v1.json"
    assert not marcador.exists()
