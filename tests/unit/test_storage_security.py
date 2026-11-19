"""Segurança de upload e armazenamento (RF-10): tipo, tamanho, extensão, conteúdo e path traversal."""

from __future__ import annotations

import pytest

from fachadaguard.config import Settings
from fachadaguard.storage.images import LocalBlobStore, UploadError, capture_key, validate_image_upload
from fachadaguard.vision.synthetic import FacadeStyle, render
from tests.conftest import png_bytes

MB = 2**20


def test_valid_png_accepted():
    data = png_bytes(render(FacadeStyle(seed=1)).image)
    info = validate_image_upload(data, "image/png", "foto.PNG", 15 * MB)
    assert (info.content_type, info.width, info.height, info.extension) == ("image/png", 640, 480, ".png")


@pytest.mark.parametrize(
    "data,ctype,name,msg",
    [
        (b"", "image/png", "a.png", "vazio"),
        (b"GIF89a....", "image/gif", "a.png", "não é PNG"),
        (b"\x89PNG\r\n\x1a\n" + b"lixo" * 10, "image/png", "a.png", "corrompida"),
        (b"\x89PNG\r\n\x1a\n", "image/png", "a.exe", "extensão"),
    ],
)
def test_invalid_uploads_rejected(data, ctype, name, msg):
    with pytest.raises(UploadError, match=msg):
        validate_image_upload(data, ctype, name, 15 * MB)


def test_declared_type_mismatch_and_size_limit():
    data = png_bytes(render(FacadeStyle(seed=1)).image)
    with pytest.raises(UploadError, match="difere"):
        validate_image_upload(data, "image/jpeg", "a.jpg", 15 * MB)
    with pytest.raises(UploadError, match="limite"):
        validate_image_upload(data, "image/png", "a.png", 1000)


@pytest.mark.parametrize(
    "key", ["../etc/passwd", "captures/../../x.png", "/abs/x.png", "captures/a b.png", "captures/A.png", "captures/x"]
)
def test_path_traversal_and_bad_keys_blocked(tmp_path, key):
    store = LocalBlobStore(tmp_path)
    with pytest.raises(ValueError):
        store.path_for(key)


def test_internal_names_are_generated(tmp_path):
    store = LocalBlobStore(tmp_path)
    key = capture_key("cap_ABCdef123", ".png")
    store.put(key, b"x")
    assert key == "captures/cap_abcdef123.png" and store.get(key) == b"x"


def test_safe_dict_hides_secrets():
    s = Settings(
        mqtt_password="segredo",
        anthropic_api_key="sk-x",
        api_token="tok",
        database_url="postgresql+psycopg://u:senha@db:5432/x",
    )
    d = s.safe_dict()
    assert "segredo" not in str(d) and "sk-x" not in str(d) and "senha" not in str(d) and "tok" not in str(d)


def test_topic_prefix_validation():
    with pytest.raises(ValueError):
        Settings(mqtt_topic_prefix="fachadaguard/#")
