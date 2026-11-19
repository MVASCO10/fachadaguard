"""Cadastro idempotente do edifício fictício de demonstração (IDs estáveis)."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from fachadaguard.domain.models import Building, Facade, FacadeRegion
from fachadaguard.vision.dataset import DEMO_BUILDING, DEMO_FACADES, DEMO_REGIONS

# Coordenadas fictícias: fachada Norte alinhada a leste-oeste; Leste alinhada a norte-sul.
_ORIGIN = (float(DEMO_BUILDING["latitude"]), float(DEMO_BUILDING["longitude"]))  # type: ignore[arg-type]


def region_center(code: str) -> tuple[float, float, float]:
    idx = int(code.split("-")[1]) - 1
    lat0, lon0 = _ORIGIN
    if code.startswith("N"):
        return lat0 + 0.00030, lon0 - 0.00025 + idx * 0.00025, 8.0 + 6.0 * (idx // 2)
    return lat0 + 0.00015 - idx * 0.00015, lon0 + 0.00045, 8.0 + 5.0 * idx


def seed_demo(session: Session) -> dict[str, int]:
    created = {"buildings": 0, "facades": 0, "regions": 0}
    if session.get(Building, DEMO_BUILDING["id"]) is None:
        session.add(Building(**DEMO_BUILDING))
        created["buildings"] += 1
    session.flush()
    for fac in DEMO_FACADES:
        if session.get(Facade, fac["id"]) is None:
            session.add(Facade(building_id=DEMO_BUILDING["id"], **fac))
            created["facades"] += 1
    session.flush()
    for reg in DEMO_REGIONS:
        exists = session.scalar(
            select(FacadeRegion).where(FacadeRegion.facade_id == reg["facade_id"], FacadeRegion.code == reg["code"])
        )
        if exists is None:
            lat, lon, alt = region_center(reg["code"])
            session.add(
                FacadeRegion(
                    id=f"reg-demo-{reg['code'].lower()}",
                    facade_id=reg["facade_id"],
                    code=reg["code"],
                    description=reg["description"],
                    center_latitude=lat,
                    center_longitude=lon,
                    center_altitude_m=alt,
                )
            )
            created["regions"] += 1
    session.flush()
    return created
