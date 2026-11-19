"""Worker de IA: consome a fila durável (tabela model_runs) e exporta periodicamente para o lake.

Vários workers podem rodar em paralelo: a reserva de job usa ``FOR UPDATE SKIP LOCKED``.
"""

from __future__ import annotations

import logging
import signal
import threading
import time
from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.exc import OperationalError

from fachadaguard.analytics.lake import export_all
from fachadaguard.config import get_settings
from fachadaguard.db import create_schema, get_engine, init_engine, session_scope
from fachadaguard.domain.models import JobStatus, ModelRun, ServiceHeartbeat, utcnow
from fachadaguard.pipeline import analyze_run, claim_next_job, mark_failed
from fachadaguard.storage.images import LocalBlobStore
from fachadaguard.vision.registry import get_or_train

log = logging.getLogger("fachadaguard.worker")
SERVICE = "ai_worker"
MAX_ATTEMPTS = 3
STALE_AFTER = timedelta(minutes=10)


def recover_stale_jobs() -> int:
    """Jobs 'running' esquecidos (worker morto) voltam à fila ou falham após MAX_ATTEMPTS."""
    cutoff = utcnow() - STALE_AFTER
    with session_scope() as s:
        stale = s.scalars(
            select(ModelRun).where(ModelRun.status == JobStatus.RUNNING, ModelRun.started_at < cutoff)
        ).all()
        for run in stale:
            run.status = JobStatus.QUEUED if run.attempts < MAX_ATTEMPTS else JobStatus.FAILED
            run.error = "recuperado após worker interrompido"
        return len(stale)


def process_one(model: object, store: LocalBlobStore) -> bool:
    with session_scope() as s:
        run = claim_next_job(s)
        if run is None:
            return False
        run_id = run.id
    try:
        with session_scope() as s:
            run = s.get(ModelRun, run_id)
            assert run is not None
            analyze_run(s, run, model, store)  # type: ignore[arg-type]
        log.info("análise %s concluída", run_id)
    except Exception as exc:  # erro sanitizado e persistido; job pode ser reprocessado
        log.exception("falha na análise %s", run_id)
        with session_scope() as s:
            mark_failed(s, run_id, exc)
    return True


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    init_engine(settings.database_url)
    create_schema()  # aguarda o banco e serializa a criação entre serviços
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    t0 = time.perf_counter()
    model = get_or_train(settings.models_dir)
    log.info("modelo %s pronto em %.1fs", model.version, time.perf_counter() - t0)
    store = LocalBlobStore(settings.data_dir / "store")
    log.info("jobs recuperados: %d", recover_stale_jobs())
    stats = {"processed": 0, "model_version": model.version, "last_export": None}
    last_export = last_beat = 0.0
    while not stop.is_set():
        try:
            if process_one(model, store):
                stats["processed"] = int(stats["processed"] or 0) + 1
                continue
            if time.monotonic() - last_export > settings.export_interval_seconds:
                stats["last_export"] = export_all(get_engine(), settings.lake_dir)
                last_export = time.monotonic()
            if time.monotonic() - last_beat > 5:
                with session_scope() as s:
                    s.merge(ServiceHeartbeat(service=SERVICE, last_seen=utcnow(), stats=dict(stats)))
                last_beat = time.monotonic()
        except OperationalError as exc:
            log.error("banco indisponível: %s", type(exc).__name__)
            time.sleep(3)
        stop.wait(settings.worker_poll_seconds)


if __name__ == "__main__":
    main()
