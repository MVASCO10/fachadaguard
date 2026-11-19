"""Contratos HTTP (entrada) da API v1. Saídas são serializadas por ``fachadaguard.api_serializers``."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from fachadaguard.schemas.mqtt import ID_PATTERN

Identifier = Annotated[str, Field(pattern=ID_PATTERN)]


class _In(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BuildingIn(_In):
    id: Identifier | None = None
    name: Annotated[str, Field(min_length=2, max_length=200)]
    address: Annotated[str, Field(max_length=300)] | None = None
    latitude: Annotated[float, Field(ge=-90, le=90)] | None = None
    longitude: Annotated[float, Field(ge=-180, le=180)] | None = None
    is_synthetic: bool = False


class FacadeIn(_In):
    id: Identifier | None = None
    name: Annotated[str, Field(min_length=2, max_length=120)]
    orientation: Annotated[str, Field(max_length=20)] | None = None
    cladding: Annotated[str, Field(max_length=120)] | None = None


class RegionIn(_In):
    code: Annotated[str, Field(pattern=r"^[A-Za-z0-9_\-]{1,40}$")]
    description: Annotated[str, Field(max_length=200)] | None = None
    floor_from: int | None = None
    floor_to: int | None = None
    center_latitude: Annotated[float, Field(ge=-90, le=90)] | None = None
    center_longitude: Annotated[float, Field(ge=-180, le=180)] | None = None
    center_altitude_m: Annotated[float, Field(ge=-10, le=500)] | None = None


class MissionIn(_In):
    id: Identifier | None = None
    building_id: Identifier
    drone_id: Identifier
    name: Annotated[str, Field(min_length=2, max_length=200)]
    planned_for: datetime | None = None
    is_simulated: bool = False
    notes: Annotated[str, Field(max_length=2000)] | None = None


class ReviewIn(_In):
    status: Literal["pendente", "confirmado", "descartado", "encaminhado_inspecao_tecnica"]
    note: Annotated[str, Field(max_length=2000)] | None = None
    reviewer: Annotated[str, Field(min_length=2, max_length=120)] = "revisor-demo"


class ComparisonIn(_In):
    before_inspection_id: Identifier
    after_inspection_id: Identifier
    force: bool = Field(default=False, description="Compara mesmo regiões diferentes (gera aviso explícito)")


class AnalysisIn(_In):
    force: bool = Field(default=False, description="Reprocessa mesmo com análise concluída")
