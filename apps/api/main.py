"""API HTTP do FachadaGuard (FastAPI). Documentação OpenAPI em /docs e /openapi.json."""

import logging
import secrets
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Path, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from fachadaguard import DISCLAIMER, __version__
from fachadaguard import serializers as S
from fachadaguard.analytics import lake
from fachadaguard.config import Settings, get_settings
from fachadaguard.db import create_schema, get_engine, init_engine, session_scope
from fachadaguard.demo_seed import seed_demo
from fachadaguard.domain.models import (
    Alert,
    Building,
    Capture,
    Comparison,
    DeadLetter,
    Facade,
    FacadeRegion,
    Finding,
    FindingReview,
    Inspection,
    JobStatus,
    Mission,
    ModelRun,
    Report,
    ServiceHeartbeat,
    TelemetryEvent,
)
from fachadaguard.ingestion import IngestionError, attach_blob, get_or_create_inspection
from fachadaguard.pipeline import enqueue_analysis, save_comparison
from fachadaguard.reporting.mission_report import generate_report
from fachadaguard.schemas.api import (
    AnalysisIn,
    BuildingIn,
    ComparisonIn,
    FacadeIn,
    MissionIn,
    RegionIn,
    ReviewIn,
)
from fachadaguard.schemas.mqtt import ID_PATTERN
from fachadaguard.storage.images import LocalBlobStore, UploadError, validate_image_upload

log = logging.getLogger("fachadaguard.api")
FILE_PREFIXES = ("captures/", "runs/", "comparisons/", "reports/")
EVENT_ID = r"^[A-Za-z0-9_\-]{8,64}$"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    init_engine(settings.database_url)
    create_schema()
    store = LocalBlobStore(settings.data_dir / "store")

    app = FastAPI(
        title="FachadaGuard API",
        version=__version__,
        description="Protótipo acadêmico de triagem de anomalias em fachadas (IoT + Big Data + IA). " + DISCLAIMER,
    )

    def db() -> Iterator[Session]:
        with session_scope() as s:
            yield s

    def auth(x_api_token: Annotated[str | None, Header()] = None) -> None:
        if settings.api_token and not secrets.compare_digest(x_api_token or "", settings.api_token):
            raise HTTPException(401, "token de API ausente ou inválido (cabeçalho X-API-Token)")

    DB = Annotated[Session, Depends(db)]
    Auth = Depends(auth)

    def get_or_404(session: Session, model: Any, id_: str) -> Any:
        obj = session.get(model, id_)
        if obj is None:
            raise HTTPException(404, f"{model.__name__} não encontrado")
        return obj

    # ------------------------------------------------------------------ saúde
    @app.get("/health", tags=["saúde"])
    def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__, "disclaimer": DISCLAIMER}

    @app.get("/ready", tags=["saúde"])
    def ready(session: DB) -> dict[str, Any]:
        checks: dict[str, Any] = {}
        try:
            session.execute(text("SELECT 1"))
            checks["database"] = "ok"
        except Exception as exc:  # pragma: no cover - depende do ambiente
            checks["database"] = f"erro: {type(exc).__name__}"
        checks["storage"] = "ok" if store.root.exists() else "ausente"
        checks["model"] = "ok" if (settings.models_dir / "current.json").exists() else "não treinado (worker treina)"
        ok = checks["database"] == "ok" and checks["storage"] == "ok"
        if not ok:
            raise HTTPException(503, checks)
        return {"status": "ready", "checks": checks, "config": settings.safe_dict()}

    # ------------------------------------------------------------------ cadastro
    @app.post("/api/v1/demo/seed", tags=["cadastro"], dependencies=[Auth])
    def demo_seed(session: DB) -> dict[str, Any]:
        return {"created": seed_demo(session)}

    @app.post("/api/v1/buildings", tags=["cadastro"], status_code=201, dependencies=[Auth])
    def create_building(body: BuildingIn, session: DB) -> dict[str, Any]:
        if body.id and session.get(Building, body.id):
            raise HTTPException(409, "edifício já existe")
        b = Building(**body.model_dump(exclude_none=True))
        session.add(b)
        session.flush()
        return S.building(b)

    @app.get("/api/v1/buildings", tags=["cadastro"])
    def list_buildings(session: DB) -> list[dict[str, Any]]:
        return [S.building(b, facades=True) for b in session.scalars(select(Building).order_by(Building.name))]

    @app.get("/api/v1/buildings/{building_id}", tags=["cadastro"])
    def get_building(building_id: str, session: DB) -> dict[str, Any]:
        return S.building(get_or_404(session, Building, building_id), facades=True)

    @app.post("/api/v1/buildings/{building_id}/facades", tags=["cadastro"], status_code=201, dependencies=[Auth])
    def create_facade(building_id: str, body: FacadeIn, session: DB) -> dict[str, Any]:
        get_or_404(session, Building, building_id)
        f = Facade(building_id=building_id, **body.model_dump(exclude_none=True))
        session.add(f)
        session.flush()
        return S.facade(f)

    @app.post("/api/v1/facades/{facade_id}/regions", tags=["cadastro"], status_code=201, dependencies=[Auth])
    def create_region(facade_id: str, body: RegionIn, session: DB) -> dict[str, Any]:
        get_or_404(session, Facade, facade_id)
        if session.scalar(
            select(FacadeRegion).where(FacadeRegion.facade_id == facade_id, FacadeRegion.code == body.code)
        ):
            raise HTTPException(409, "código de região já existe nesta fachada")
        r = FacadeRegion(facade_id=facade_id, **body.model_dump(exclude_none=True))
        session.add(r)
        session.flush()
        return S.region(r)

    # ------------------------------------------------------------------ missões
    @app.post("/api/v1/missions", tags=["missões"], status_code=201, dependencies=[Auth])
    def create_mission(body: MissionIn, session: DB) -> dict[str, Any]:
        get_or_404(session, Building, body.building_id)
        if body.id and (existing := session.get(Mission, body.id)):
            if existing.building_id != body.building_id:
                raise HTTPException(409, "missão existe para outro edifício")
            return S.mission(existing)  # idempotente para o simulador
        m = Mission(**body.model_dump(exclude_none=True))
        session.add(m)
        session.flush()
        return S.mission(m)

    @app.get("/api/v1/missions", tags=["missões"])
    def list_missions(session: DB, building_id: str | None = None) -> list[dict[str, Any]]:
        stmt = select(Mission).order_by(Mission.planned_for.desc().nulls_last(), Mission.created_at.desc())
        if building_id:
            stmt = stmt.where(Mission.building_id == building_id)
        out = []
        for m in session.scalars(stmt):
            d = S.mission(m)
            d["telemetry_events"] = session.scalar(
                select(func.count()).select_from(TelemetryEvent).where(TelemetryEvent.mission_id == m.id)
            )
            d["captures"] = session.scalar(select(func.count()).select_from(Capture).where(Capture.mission_id == m.id))
            d["open_alerts"] = session.scalar(
                select(func.count()).select_from(Alert).where(Alert.mission_id == m.id, Alert.status == "aberto")
            )
            out.append(d)
        return out

    @app.get("/api/v1/missions/{mission_id}", tags=["missões"])
    def get_mission(mission_id: str, session: DB) -> dict[str, Any]:
        """Missão com todo o histórico: inspeções, capturas, análises, achados e alertas."""
        m = get_or_404(session, Mission, mission_id)
        inspections = []
        for insp in session.scalars(
            select(Inspection).where(Inspection.mission_id == mission_id).order_by(Inspection.inspected_at)
        ):
            region = session.get(FacadeRegion, insp.region_id)
            caps = []
            for c in insp.captures:
                caps.append({**S.capture(c), "runs": [S.run(r) for r in c.runs]})
            inspections.append({**S.inspection(insp), "region": S.region(region) if region else None, "captures": caps})
        orphan = session.scalars(
            select(Capture).where(Capture.mission_id == mission_id, Capture.inspection_id.is_(None))
        ).all()
        findings = session.scalars(
            select(Finding).join(Capture, Capture.id == Finding.capture_id).where(Capture.mission_id == mission_id)
        ).all()
        return {
            **S.mission(m),
            "inspections": inspections,
            "captures_awaiting_metadata": [S.capture(c) for c in orphan],
            "findings": [S.finding(f) for f in findings],
            "alerts": [S.alert(a) for a in session.scalars(select(Alert).where(Alert.mission_id == mission_id))],
            "reports": [
                S.report(r)
                for r in session.scalars(
                    select(Report).where(Report.mission_id == mission_id).order_by(Report.generated_at.desc())
                )
            ],
        }

    @app.get("/api/v1/missions/{mission_id}/telemetry", tags=["missões"])
    def mission_telemetry(
        mission_id: str,
        session: DB,
        limit: Annotated[int, Query(ge=1, le=20000)] = 2000,
        every: Annotated[int, Query(ge=1, le=1000)] = 1,
    ) -> list[dict[str, Any]]:
        get_or_404(session, Mission, mission_id)
        T = TelemetryEvent
        rows = session.execute(
            select(
                T.captured_at,
                T.latitude,
                T.longitude,
                T.altitude_m,
                T.battery_pct,
                T.speed_mps,
                T.heading_deg,
                T.signal_quality,
                T.mission_state,
                T.is_simulated,
            )
            .where(T.mission_id == mission_id, (T.id % every) == 0)
            .order_by(T.captured_at)
            .limit(limit)
        ).all()
        return [dict(r._mapping) for r in rows]

    # ------------------------------------------------------------------ capturas
    async def _read_upload(file: UploadFile) -> tuple[bytes, Any]:
        max_bytes = int(settings.max_upload_mb * 1024 * 1024)
        data = await file.read(max_bytes + 1)
        try:
            return data, validate_image_upload(data, file.content_type, file.filename, max_bytes)
        except UploadError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/v1/captures/{event_id}/image", tags=["capturas"], dependencies=[Auth])
    async def upload_capture_image(
        event_id: Annotated[str, Path(pattern=EVENT_ID)],
        session: DB,
        file: UploadFile = File(...),
        mission_id: str = Form(..., pattern=ID_PATTERN),
        captured_at: datetime = Form(...),
        drone_id: str = Form("unknown", pattern=ID_PATTERN),
    ) -> dict[str, Any]:
        """Upload binário do drone. Associa ao evento MQTT de mesmo ``event_id`` (idempotente)."""
        if captured_at.tzinfo is None:
            raise HTTPException(422, "captured_at deve conter timezone (UTC)")
        data, info = await _read_upload(file)
        try:
            cap, new, enq = attach_blob(session, store, event_id, mission_id, captured_at, data, info, drone_id)
        except IngestionError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"capture": S.capture(cap), "created": new, "analysis_enqueued": enq}

    @app.post("/api/v1/missions/{mission_id}/images", tags=["capturas"], status_code=201, dependencies=[Auth])
    async def manual_upload(
        mission_id: str,
        session: DB,
        file: UploadFile = File(...),
        region_id: str = Form(..., pattern=ID_PATTERN),
        captured_at: datetime | None = Form(None),
    ) -> dict[str, Any]:
        """Upload manual (ex.: pelo dashboard) de uma imagem para uma região de fachada da missão."""
        m = get_or_404(session, Mission, mission_id)
        region = get_or_404(session, FacadeRegion, region_id)
        facade = session.get(Facade, region.facade_id)
        if facade is None or facade.building_id != m.building_id:
            raise HTTPException(422, "região não pertence ao edifício da missão")
        at = (captured_at or datetime.now(UTC)).astimezone(UTC)
        data, info = await _read_upload(file)
        event_id = f"manual-{secrets.token_hex(8)}"
        cap, _, _ = attach_blob(session, store, event_id, mission_id, at, data, info, m.drone_id, source="manual")
        insp = get_or_create_inspection(session, mission_id, region.id, at)
        cap.inspection_id, cap.region_id, cap.metadata_received = insp.id, region.id, True
        cap.expected_sha256 = info.sha256
        session.flush()
        run = enqueue_analysis(session, cap)
        return {"capture": S.capture(cap), "analysis": S.run(run) if run else None}

    @app.get("/api/v1/captures/{capture_id}", tags=["capturas"])
    def get_capture(capture_id: str, session: DB) -> dict[str, Any]:
        c = get_or_404(session, Capture, capture_id)
        return {**S.capture(c), "runs": [S.run(r) for r in c.runs]}

    @app.get("/api/v1/files/{key:path}", tags=["capturas"], include_in_schema=True)
    def get_file(key: str) -> FileResponse:
        if not key.startswith(FILE_PREFIXES):
            raise HTTPException(404, "arquivo não encontrado")
        try:
            path = store.path_for(key)
        except ValueError as exc:
            raise HTTPException(400, "chave inválida") from exc
        if not path.exists():
            raise HTTPException(404, "arquivo não encontrado")
        return FileResponse(path)

    # ------------------------------------------------------------------ inspeções e análises
    @app.get("/api/v1/inspections", tags=["inspeções"])
    def list_inspections(
        session: DB,
        building_id: str | None = None,
        facade_id: str | None = None,
        region_id: str | None = None,
        mission_id: str | None = None,
    ) -> list[dict[str, Any]]:
        stmt = (
            select(Inspection, FacadeRegion, Facade, Mission)
            .join(FacadeRegion, FacadeRegion.id == Inspection.region_id)
            .join(Facade, Facade.id == FacadeRegion.facade_id)
            .join(Mission, Mission.id == Inspection.mission_id)
            .order_by(Inspection.inspected_at)
        )
        for col, val in (
            (Facade.building_id, building_id),
            (Facade.id, facade_id),
            (Inspection.region_id, region_id),
            (Inspection.mission_id, mission_id),
        ):
            if val:
                stmt = stmt.where(col == val)
        out = []
        for insp, reg, fac, mis in session.execute(stmt):
            cap = next(iter(sorted(insp.captures, key=lambda c: c.captured_at)), None)
            last = next((r for r in reversed(cap.runs) if r.status == JobStatus.SUCCEEDED), None) if cap else None
            out.append(
                {
                    **S.inspection(insp),
                    "region_code": reg.code,
                    "facade_id": fac.id,
                    "facade_name": fac.name,
                    "mission_name": mis.name,
                    "is_simulated": mis.is_simulated,
                    "capture": S.capture(cap) if cap else None,
                    "analysis": S.run(last) if last else None,
                }
            )
        return out

    @app.post("/api/v1/captures/{capture_id}/analyses", tags=["análises"], status_code=202, dependencies=[Auth])
    def request_analysis(capture_id: str, body: AnalysisIn, session: DB) -> dict[str, Any]:
        cap = get_or_404(session, Capture, capture_id)
        run = enqueue_analysis(session, cap, force=body.force)
        if run is None:
            raise HTTPException(409, "captura incompleta ou já possui análise ativa/concluída (use force=true)")
        return S.run(run)

    @app.get("/api/v1/analyses", tags=["análises"])
    def list_analyses(
        session: DB, status: str | None = None, limit: Annotated[int, Query(ge=1, le=500)] = 100
    ) -> list[dict[str, Any]]:
        stmt = select(ModelRun).order_by(ModelRun.queued_at.desc()).limit(limit)
        if status:
            stmt = stmt.where(ModelRun.status == status)
        return [S.run(r) for r in session.scalars(stmt)]

    @app.get("/api/v1/analyses/{run_id}", tags=["análises"])
    def get_analysis(run_id: str, session: DB) -> dict[str, Any]:
        r = get_or_404(session, ModelRun, run_id)
        return {**S.run(r), "findings": [S.finding(f) for f in r.findings]}

    # ------------------------------------------------------------------ achados, revisão, alertas
    @app.get("/api/v1/findings", tags=["achados"])
    def list_findings(
        session: DB,
        building_id: str | None = None,
        facade_id: str | None = None,
        mission_id: str | None = None,
        severity: Annotated[list[str] | None, Query()] = None,
        review_status: Annotated[list[str] | None, Query()] = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        limit: Annotated[int, Query(ge=1, le=2000)] = 500,
    ) -> list[dict[str, Any]]:
        stmt = (
            select(Finding, Inspection, FacadeRegion, Facade)
            .join(Inspection, Inspection.id == Finding.inspection_id)
            .join(FacadeRegion, FacadeRegion.id == Finding.region_id)
            .join(Facade, Facade.id == FacadeRegion.facade_id)
            .order_by(Finding.severity_points.desc())
            .limit(limit)
        )
        if building_id:
            stmt = stmt.where(Facade.building_id == building_id)
        if facade_id:
            stmt = stmt.where(Facade.id == facade_id)
        if mission_id:
            stmt = stmt.where(Inspection.mission_id == mission_id)
        if severity:
            stmt = stmt.where(Finding.severity_level.in_(severity))
        if review_status:
            stmt = stmt.where(Finding.review_status.in_(review_status))
        if date_from:
            stmt = stmt.where(Inspection.inspected_at >= date_from)
        if date_to:
            stmt = stmt.where(Inspection.inspected_at <= date_to)
        return [
            {
                **S.finding(f),
                "inspected_at": i.inspected_at,
                "mission_id": i.mission_id,
                "region_code": r.code,
                "facade_id": fa.id,
                "facade_name": fa.name,
            }
            for f, i, r, fa in session.execute(stmt)
        ]

    @app.get("/api/v1/findings/{finding_id}", tags=["achados"])
    def get_finding(finding_id: str, session: DB) -> dict[str, Any]:
        f = get_or_404(session, Finding, finding_id)
        cap = session.get(Capture, f.capture_id)
        run = session.get(ModelRun, f.model_run_id)
        alert = session.scalar(select(Alert).where(Alert.finding_id == f.id))
        return {
            **S.finding(f, detail=True),
            "capture": S.capture(cap) if cap else None,
            "analysis": S.run(run) if run else None,
            "alert": S.alert(alert) if alert else None,
            "disclaimer": DISCLAIMER,
        }

    @app.post("/api/v1/findings/{finding_id}/review", tags=["achados"], dependencies=[Auth])
    def review_finding(finding_id: str, body: ReviewIn, session: DB) -> dict[str, Any]:
        """Registra decisão humana. Não dispara nenhuma ação física."""
        f = get_or_404(session, Finding, finding_id)
        session.add(
            FindingReview(
                finding_id=f.id,
                previous_status=f.review_status,
                status=body.status,
                note=body.note,
                reviewer=body.reviewer,
            )
        )
        f.review_status = body.status
        alert = session.scalar(select(Alert).where(Alert.finding_id == f.id))
        if alert is not None:
            alert.status = {"descartado": "fechado", "pendente": "aberto"}.get(body.status, "reconhecido")
        session.flush()
        session.refresh(f)
        return S.finding(f, detail=True)

    @app.get("/api/v1/alerts", tags=["achados"])
    def list_alerts(session: DB, status: str | None = None) -> list[dict[str, Any]]:
        stmt = select(Alert).order_by(Alert.created_at.desc())
        if status:
            stmt = stmt.where(Alert.status == status)
        return [S.alert(a) for a in session.scalars(stmt)]

    # ------------------------------------------------------------------ comparação temporal
    @app.post("/api/v1/comparisons", tags=["comparação"], status_code=201, dependencies=[Auth])
    def create_comparison(body: ComparisonIn, session: DB) -> dict[str, Any]:
        a = get_or_404(session, Inspection, body.before_inspection_id)
        b = get_or_404(session, Inspection, body.after_inspection_id)
        if a.id == b.id:
            raise HTTPException(422, "selecione duas inspeções diferentes")
        if a.inspected_at > b.inspected_at:
            a, b = b, a
        warnings = []
        if a.region_id != b.region_id:
            if not body.force:
                raise HTTPException(
                    422,
                    "inspeções de regiões diferentes não são comparáveis automaticamente; "
                    "use force=true apenas para fins exploratórios",
                )
            warnings.append(
                "ATENÇÃO: regiões diferentes comparadas por solicitação explícita; resultado não "
                "representa evolução de uma mesma área."
            )
        caps = []
        for insp in (a, b):
            cap = next((c for c in insp.captures if c.blob_received), None)
            if cap is None or not cap.storage_key:
                raise HTTPException(409, f"inspeção {insp.id} sem imagem")
            caps.append(cap)
        runs = [next((r for r in reversed(c.runs) if r.status == JobStatus.SUCCEEDED), None) for c in caps]
        cmp, _ = save_comparison(
            session,
            store,
            a,
            b,
            store.read_image(caps[0].storage_key),
            store.read_image(caps[1].storage_key),
            runs[0],
            runs[1],
            warnings,
        )
        return {
            **S.comparison(cmp),
            "image_before_url": S.file_url(caps[0].storage_key),
            "image_after_url": S.file_url(caps[1].storage_key),
        }

    @app.get("/api/v1/comparisons", tags=["comparação"])
    def list_comparisons(session: DB, region_id: str | None = None) -> list[dict[str, Any]]:
        stmt = select(Comparison).order_by(Comparison.created_at.desc())
        if region_id:
            stmt = stmt.join(Inspection, Inspection.id == Comparison.inspection_after_id).where(
                Inspection.region_id == region_id
            )
        return [S.comparison(c) for c in session.scalars(stmt)]

    @app.get("/api/v1/comparisons/{comparison_id}", tags=["comparação"])
    def get_comparison(comparison_id: str, session: DB) -> dict[str, Any]:
        c = get_or_404(session, Comparison, comparison_id)
        out = S.comparison(c)
        for side, iid in (("before", c.inspection_before_id), ("after", c.inspection_after_id)):
            insp = session.get(Inspection, iid)
            cap = next((x for x in insp.captures if x.storage_key), None) if insp else None
            out[f"image_{side}_url"] = S.file_url(cap.storage_key) if cap else None
            out[f"inspected_at_{side}"] = insp.inspected_at if insp else None
        return out

    # ------------------------------------------------------------------ relatórios
    @app.post("/api/v1/missions/{mission_id}/reports", tags=["relatórios"], status_code=201, dependencies=[Auth])
    def create_report(mission_id: str, session: DB) -> dict[str, Any]:
        get_or_404(session, Mission, mission_id)
        return S.report(generate_report(session, store, settings, mission_id))

    @app.get("/api/v1/reports/{report_id}", tags=["relatórios"])
    def get_report(report_id: str, session: DB) -> dict[str, Any]:
        return S.report(get_or_404(session, Report, report_id))

    @app.get("/api/v1/reports/{report_id}/html", tags=["relatórios"], response_class=HTMLResponse)
    def get_report_html(report_id: str, session: DB) -> HTMLResponse:
        r = get_or_404(session, Report, report_id)
        return HTMLResponse(
            store.get(r.storage_key).decode("utf-8"), headers={"Content-Disposition": f'inline; filename="{r.id}.html"'}
        )

    # ------------------------------------------------------------------ métricas, ingestão, analítica
    @app.get("/api/v1/metrics/summary", tags=["métricas"])
    def metrics_summary(session: DB) -> dict[str, Any]:
        def count(model: Any, *where: Any) -> int:
            return int(session.scalar(select(func.count()).select_from(model).where(*where)) or 0)

        since = datetime.now(UTC) - timedelta(minutes=5)
        sev = dict(session.execute(select(Finding.severity_level, func.count()).group_by(Finding.severity_level)).all())
        rev = dict(session.execute(select(Finding.review_status, func.count()).group_by(Finding.review_status)).all())
        jobs = dict(session.execute(select(ModelRun.status, func.count()).group_by(ModelRun.status)).all())
        return {
            "buildings": count(Building),
            "missions": count(Mission),
            "inspections": count(Inspection),
            "telemetry_events": count(TelemetryEvent),
            "telemetry_last_5min": count(TelemetryEvent, TelemetryEvent.received_at >= since),
            "captures": count(Capture),
            "captures_analyzed": count(ModelRun, ModelRun.status == JobStatus.SUCCEEDED),
            "jobs": jobs,
            "findings": count(Finding),
            "findings_by_severity": sev,
            "findings_by_review": rev,
            "alerts_open": count(Alert, Alert.status == "aberto"),
            "dead_letters": count(DeadLetter),
            "comparisons": count(Comparison),
            "reports": count(Report),
        }

    @app.get("/api/v1/ingestion/status", tags=["métricas"])
    def ingestion_status(session: DB) -> dict[str, Any]:
        beats = session.scalars(select(ServiceHeartbeat)).all()
        recent = session.execute(
            select(
                TelemetryEvent.drone_id,
                TelemetryEvent.mission_id,
                TelemetryEvent.captured_at,
                TelemetryEvent.received_at,
                TelemetryEvent.battery_pct,
                TelemetryEvent.signal_quality,
                TelemetryEvent.mission_state,
            )
            .order_by(TelemetryEvent.id.desc())
            .limit(15)
        ).all()
        dl = session.scalars(select(DeadLetter).order_by(DeadLetter.id.desc()).limit(10)).all()
        now = datetime.now(UTC)
        return {
            "services": [
                {
                    "service": b.service,
                    "last_seen": b.last_seen,
                    "alive": (now - b.last_seen).total_seconds() < 30,
                    "stats": b.stats,
                }
                for b in beats
            ],
            "recent_events": [dict(r._mapping) for r in recent],
            "dead_letters": [{"topic": d.topic, "error": d.error, "received_at": d.received_at} for d in dl],
        }

    @app.post("/api/v1/analytics/export", tags=["analítica"], dependencies=[Auth])
    def analytics_export() -> dict[str, Any]:
        return lake.export_all(get_engine(), settings.lake_dir)

    @app.get("/api/v1/analytics/{name}", tags=["analítica"])
    def analytics(name: str) -> Any:
        queries = {
            "telemetry-by-mission": lake.telemetry_by_mission,
            "telemetry-distribution": lake.telemetry_distribution,
            "events-per-day": lake.events_per_day,
            "anomalies-by-facade": lake.anomalies_by_facade,
            "findings-by-category": lake.findings_by_category,
            "lake-stats": lake.lake_stats,
        }
        if name not in queries:
            raise HTTPException(404, f"consulta desconhecida; disponíveis: {sorted(queries)}")
        return queries[name](settings.lake_dir)

    return app


# Execução: uvicorn apps.api.main:create_app --factory --host 0.0.0.0 --port 8000
