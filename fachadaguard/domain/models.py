"""Modelo de dados operacional (PostgreSQL; SQLite apenas em testes unitários).

Convenções: IDs textuais estáveis com prefixo, timestamps sempre em UTC (timezone-aware),
coordenadas WGS84 em graus decimais e altitude em metros acima do solo.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator

BigIntPK = BigInteger().with_variant(Integer, "sqlite")


class UTCDateTime(TypeDecorator[datetime]):
    """Garante datetimes timezone-aware em UTC também no SQLite."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Any) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("datetime sem timezone; use UTC explícito")
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect: Any) -> datetime | None:
        if value is None:
            return None
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def utcnow() -> datetime:
    return datetime.now(UTC)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


class Base(DeclarativeBase):
    type_annotation_map = {datetime: UTCDateTime(), dict[str, Any]: JSON, list[Any]: JSON}


class MissionStatus(StrEnum):
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ABORTED = "aborted"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ReviewStatus(StrEnum):
    PENDENTE = "pendente"
    CONFIRMADO = "confirmado"
    DESCARTADO = "descartado"
    ENCAMINHADO = "encaminhado_inspecao_tecnica"


class Category(StrEnum):
    FISSURA = "fissura"
    UMIDADE = "umidade_infiltracao"
    DESPLACAMENTO = "desplacamento_aparente"
    INDETERMINADO = "indeterminado"


class Building(Base):
    __tablename__ = "buildings"
    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("bld"))
    name: Mapped[str] = mapped_column(String(200))
    address: Mapped[str | None] = mapped_column(String(300))
    latitude: Mapped[float | None] = mapped_column(Float)
    longitude: Mapped[float | None] = mapped_column(Float)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    facades: Mapped[list[Facade]] = relationship(back_populates="building", cascade="all, delete-orphan")


class Facade(Base):
    __tablename__ = "facades"
    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("fac"))
    building_id: Mapped[str] = mapped_column(ForeignKey("buildings.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(120))
    orientation: Mapped[str | None] = mapped_column(String(20))
    cladding: Mapped[str | None] = mapped_column(String(120))
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    building: Mapped[Building] = relationship(back_populates="facades")
    regions: Mapped[list[FacadeRegion]] = relationship(back_populates="facade", cascade="all, delete-orphan")


class FacadeRegion(Base):
    __tablename__ = "facade_regions"
    __table_args__ = (UniqueConstraint("facade_id", "code"),)
    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("reg"))
    facade_id: Mapped[str] = mapped_column(ForeignKey("facades.id", ondelete="CASCADE"), index=True)
    code: Mapped[str] = mapped_column(String(40))
    description: Mapped[str | None] = mapped_column(String(200))
    floor_from: Mapped[int | None] = mapped_column(Integer)
    floor_to: Mapped[int | None] = mapped_column(Integer)
    center_latitude: Mapped[float | None] = mapped_column(Float)
    center_longitude: Mapped[float | None] = mapped_column(Float)
    center_altitude_m: Mapped[float | None] = mapped_column(Float)
    facade: Mapped[Facade] = relationship(back_populates="regions")


class Mission(Base):
    __tablename__ = "missions"
    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("msn"))
    building_id: Mapped[str] = mapped_column(ForeignKey("buildings.id"), index=True)
    drone_id: Mapped[str] = mapped_column(String(64), index=True)
    name: Mapped[str] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(20), default=MissionStatus.PLANNED)
    is_simulated: Mapped[bool] = mapped_column(Boolean, default=False)
    planned_for: Mapped[datetime | None] = mapped_column()
    started_at: Mapped[datetime | None] = mapped_column()
    ended_at: Mapped[datetime | None] = mapped_column()
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    building: Mapped[Building] = relationship()
    inspections: Mapped[list[Inspection]] = relationship(back_populates="mission")


class Inspection(Base):
    """Inspeção de uma região de fachada dentro de uma missão (unidade de comparação temporal)."""

    __tablename__ = "inspections"
    __table_args__ = (UniqueConstraint("mission_id", "region_id"),)
    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("ins"))
    mission_id: Mapped[str] = mapped_column(ForeignKey("missions.id", ondelete="CASCADE"), index=True)
    region_id: Mapped[str] = mapped_column(ForeignKey("facade_regions.id"), index=True)
    inspected_at: Mapped[datetime] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    mission: Mapped[Mission] = relationship(back_populates="inspections")
    region: Mapped[FacadeRegion] = relationship()
    captures: Mapped[list[Capture]] = relationship(back_populates="inspection")


class TelemetryEvent(Base):
    __tablename__ = "telemetry_events"
    __table_args__ = (Index("ix_telemetry_mission_time", "mission_id", "captured_at"),)
    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(64), unique=True)
    schema_version: Mapped[str] = mapped_column(String(10))
    drone_id: Mapped[str] = mapped_column(String(64), index=True)
    mission_id: Mapped[str] = mapped_column(String(64))
    captured_at: Mapped[datetime] = mapped_column()
    received_at: Mapped[datetime] = mapped_column(default=utcnow, index=True)
    latitude: Mapped[float] = mapped_column(Float)
    longitude: Mapped[float] = mapped_column(Float)
    altitude_m: Mapped[float] = mapped_column(Float)
    battery_pct: Mapped[float] = mapped_column(Float)
    speed_mps: Mapped[float] = mapped_column(Float)
    heading_deg: Mapped[float] = mapped_column(Float)
    signal_quality: Mapped[float] = mapped_column(Float)
    mission_state: Mapped[str] = mapped_column(String(20))
    is_simulated: Mapped[bool] = mapped_column(Boolean, default=False)
    payload_sha256: Mapped[str] = mapped_column(String(64))


class Capture(Base):
    """Imagem capturada. Metadados chegam por MQTT e o binário por HTTP; ambos usam o mesmo event_id."""

    __tablename__ = "captures"
    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("cap"))
    event_id: Mapped[str] = mapped_column(String(64), unique=True)
    mission_id: Mapped[str] = mapped_column(ForeignKey("missions.id"), index=True)
    inspection_id: Mapped[str | None] = mapped_column(ForeignKey("inspections.id"), index=True)
    region_id: Mapped[str | None] = mapped_column(ForeignKey("facade_regions.id"), index=True)
    drone_id: Mapped[str] = mapped_column(String(64))
    captured_at: Mapped[datetime] = mapped_column()
    latitude: Mapped[float | None] = mapped_column(Float)
    longitude: Mapped[float | None] = mapped_column(Float)
    altitude_m: Mapped[float | None] = mapped_column(Float)
    heading_deg: Mapped[float | None] = mapped_column(Float)
    gimbal_pitch_deg: Mapped[float | None] = mapped_column(Float)
    storage_key: Mapped[str | None] = mapped_column(String(200))
    sha256: Mapped[str | None] = mapped_column(String(64))
    expected_sha256: Mapped[str | None] = mapped_column(String(64))
    content_type: Mapped[str | None] = mapped_column(String(40))
    size_bytes: Mapped[int | None] = mapped_column(Integer)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    metadata_received: Mapped[bool] = mapped_column(Boolean, default=False)
    blob_received: Mapped[bool] = mapped_column(Boolean, default=False)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False)
    source: Mapped[str] = mapped_column(String(20), default="mqtt")
    raw_metadata: Mapped[dict[str, Any] | None] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    inspection: Mapped[Inspection | None] = relationship(back_populates="captures")
    runs: Mapped[list[ModelRun]] = relationship(back_populates="capture", order_by="ModelRun.queued_at")


class ModelRun(Base):
    """Execução de modelo (job da fila durável). Estados: queued → running → succeeded|failed."""

    __tablename__ = "model_runs"
    __table_args__ = (Index("ix_model_runs_status_queued", "status", "queued_at"),)
    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("run"))
    capture_id: Mapped[str] = mapped_column(ForeignKey("captures.id", ondelete="CASCADE"), index=True)
    status: Mapped[str] = mapped_column(String(20), default=JobStatus.QUEUED)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    model_name: Mapped[str | None] = mapped_column(String(80))
    model_version: Mapped[str | None] = mapped_column(String(80))
    params: Mapped[dict[str, Any] | None] = mapped_column()
    reference_set_hash: Mapped[str | None] = mapped_column(String(64))
    image_score: Mapped[float | None] = mapped_column(Float)
    threshold: Mapped[float | None] = mapped_column(Float)
    anomalous_area_ratio: Mapped[float | None] = mapped_column(Float)
    quality: Mapped[dict[str, Any] | None] = mapped_column()
    heatmap_key: Mapped[str | None] = mapped_column(String(200))
    overlay_key: Mapped[str | None] = mapped_column(String(200))
    duration_ms: Mapped[float | None] = mapped_column(Float)
    error: Mapped[str | None] = mapped_column(String(500))
    queued_at: Mapped[datetime] = mapped_column(default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column()
    finished_at: Mapped[datetime | None] = mapped_column()
    capture: Mapped[Capture] = relationship(back_populates="runs")
    findings: Mapped[list[Finding]] = relationship(back_populates="model_run", cascade="all, delete-orphan")


class Finding(Base):
    __tablename__ = "findings"
    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("fnd"))
    model_run_id: Mapped[str] = mapped_column(ForeignKey("model_runs.id", ondelete="CASCADE"), index=True)
    capture_id: Mapped[str] = mapped_column(ForeignKey("captures.id", ondelete="CASCADE"), index=True)
    inspection_id: Mapped[str | None] = mapped_column(ForeignKey("inspections.id"), index=True)
    region_id: Mapped[str | None] = mapped_column(ForeignKey("facade_regions.id"), index=True)
    bbox_x: Mapped[int] = mapped_column(Integer)
    bbox_y: Mapped[int] = mapped_column(Integer)
    bbox_w: Mapped[int] = mapped_column(Integer)
    bbox_h: Mapped[int] = mapped_column(Integer)
    area_px: Mapped[int] = mapped_column(Integer)
    area_ratio: Mapped[float] = mapped_column(Float)
    peak_score: Mapped[float] = mapped_column(Float)
    score_ratio: Mapped[float] = mapped_column(Float)
    candidate_category: Mapped[str] = mapped_column(String(40))
    category_scores: Mapped[dict[str, Any]] = mapped_column()
    category_confidence: Mapped[str] = mapped_column(String(10))
    category_explanation: Mapped[str] = mapped_column(Text)
    growth_ratio: Mapped[float | None] = mapped_column(Float)
    severity_points: Mapped[float] = mapped_column(Float)
    severity_level: Mapped[str] = mapped_column(String(10), index=True)
    severity_explanation: Mapped[dict[str, Any]] = mapped_column()
    review_status: Mapped[str] = mapped_column(String(40), default=ReviewStatus.PENDENTE, index=True)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    model_run: Mapped[ModelRun] = relationship(back_populates="findings")
    reviews: Mapped[list[FindingReview]] = relationship(
        back_populates="finding", cascade="all, delete-orphan", order_by="FindingReview.created_at"
    )


class FindingReview(Base):
    __tablename__ = "finding_reviews"
    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("rev"))
    finding_id: Mapped[str] = mapped_column(ForeignKey("findings.id", ondelete="CASCADE"), index=True)
    previous_status: Mapped[str] = mapped_column(String(40))
    status: Mapped[str] = mapped_column(String(40))
    note: Mapped[str | None] = mapped_column(Text)
    reviewer: Mapped[str] = mapped_column(String(120))
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    finding: Mapped[Finding] = relationship(back_populates="reviews")


class Alert(Base):
    __tablename__ = "alerts"
    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("alr"))
    finding_id: Mapped[str] = mapped_column(ForeignKey("findings.id", ondelete="CASCADE"), unique=True)
    mission_id: Mapped[str] = mapped_column(ForeignKey("missions.id"), index=True)
    level: Mapped[str] = mapped_column(String(10))
    reason: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="aberto")
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class Comparison(Base):
    __tablename__ = "comparisons"
    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("cmp"))
    inspection_before_id: Mapped[str] = mapped_column(ForeignKey("inspections.id"), index=True)
    inspection_after_id: Mapped[str] = mapped_column(ForeignKey("inspections.id"), index=True)
    status: Mapped[str] = mapped_column(String(30))
    compatible: Mapped[bool] = mapped_column(Boolean)
    interval_days: Mapped[float | None] = mapped_column(Float)
    alignment: Mapped[dict[str, Any]] = mapped_column()
    ssim: Mapped[float | None] = mapped_column(Float)
    changed_area_ratio: Mapped[float | None] = mapped_column(Float)
    score_before: Mapped[float | None] = mapped_column(Float)
    score_after: Mapped[float | None] = mapped_column(Float)
    anomalous_area_before: Mapped[float | None] = mapped_column(Float)
    anomalous_area_after: Mapped[float | None] = mapped_column(Float)
    warnings: Mapped[list[Any]] = mapped_column()
    overlay_key: Mapped[str | None] = mapped_column(String(200))
    aligned_key: Mapped[str | None] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class Report(Base):
    __tablename__ = "reports"
    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("rpt"))
    mission_id: Mapped[str] = mapped_column(ForeignKey("missions.id"), index=True)
    generator: Mapped[str] = mapped_column(String(30))
    storage_key: Mapped[str] = mapped_column(String(200))
    sha256: Mapped[str] = mapped_column(String(64))
    summary: Mapped[dict[str, Any]] = mapped_column()
    generated_at: Mapped[datetime] = mapped_column(default=utcnow)


class DeadLetter(Base):
    """Mensagens rejeitadas (payload inválido); guarda apenas um trecho truncado."""

    __tablename__ = "dead_letters"
    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    topic: Mapped[str] = mapped_column(String(300))
    error: Mapped[str] = mapped_column(String(500))
    payload_excerpt: Mapped[str] = mapped_column(String(500))
    received_at: Mapped[datetime] = mapped_column(default=utcnow)


class ServiceHeartbeat(Base):
    __tablename__ = "service_heartbeats"
    service: Mapped[str] = mapped_column(String(60), primary_key=True)
    last_seen: Mapped[datetime] = mapped_column(default=utcnow)
    stats: Mapped[dict[str, Any]] = mapped_column(default=dict)
