"""Armazenamento de binários (imagens, mapas de calor, relatórios) atrás de uma interface simples.

A implementação local grava em disco (volume Docker). Uma implementação S3/MinIO pode
satisfazer o mesmo protocolo ``BlobStore`` sem alterar API ou worker.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

ALLOWED_TYPES = {"image/png": (b"\x89PNG\r\n\x1a\n", ".png"), "image/jpeg": (b"\xff\xd8\xff", ".jpg")}
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
MAX_PIXELS = 40_000_000
KEY_RE = re.compile(r"^[a-z0-9_\-]+(/[a-z0-9_\-]+)*\.[a-z0-9]{2,5}$")


class UploadError(ValueError):
    """Upload rejeitado por tipo, tamanho, conteúdo ou integridade."""


@dataclass(frozen=True)
class ValidatedImage:
    content_type: str
    extension: str
    sha256: str
    size_bytes: int
    width: int
    height: int


class BlobStore(Protocol):
    def put(self, key: str, data: bytes) -> None: ...
    def get(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...
    def path_for(self, key: str) -> Path: ...


def validate_image_upload(
    data: bytes, declared_type: str | None, filename: str | None, max_bytes: int
) -> ValidatedImage:
    if not data:
        raise UploadError("arquivo vazio")
    if len(data) > max_bytes:
        raise UploadError(f"arquivo excede o limite de {max_bytes // (1024 * 1024)} MB")
    if filename:
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise UploadError(f"extensão não permitida: {ext or '(nenhuma)'}")
    detected = next((ct for ct, (magic, _) in ALLOWED_TYPES.items() if data.startswith(magic)), None)
    if detected is None:
        raise UploadError("conteúdo não é PNG nem JPEG")
    if declared_type and declared_type.split(";")[0].strip() not in (detected, "application/octet-stream"):
        raise UploadError(f"tipo declarado ({declared_type}) difere do conteúdo ({detected})")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise UploadError("imagem corrompida ou ilegível")
    h, w = img.shape[:2]
    if h * w > MAX_PIXELS or min(h, w) < 64:
        raise UploadError(f"dimensões fora do intervalo aceito: {w}x{h}")
    return ValidatedImage(
        content_type=detected,
        extension=ALLOWED_TYPES[detected][1],
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        width=w,
        height=h,
    )


class LocalBlobStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: str) -> Path:
        if not KEY_RE.match(key):
            raise ValueError("chave de armazenamento inválida")
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("chave fora do diretório de armazenamento")
        return path

    def put(self, key: str, data: bytes) -> None:
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)

    def get(self, key: str) -> bytes:
        return self.path_for(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self.path_for(key).exists()

    def put_image(self, key: str, image: np.ndarray, ext: str = ".png") -> None:
        ok, buf = cv2.imencode(ext, image)
        if not ok:
            raise ValueError("falha ao codificar imagem")
        self.put(key, buf.tobytes())

    def read_image(self, key: str) -> np.ndarray:
        img = cv2.imdecode(np.frombuffer(self.get(key), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"imagem ilegível: {key}")
        return img


def capture_key(capture_id: str, extension: str) -> str:
    return f"captures/{capture_id.lower()}{extension}"
