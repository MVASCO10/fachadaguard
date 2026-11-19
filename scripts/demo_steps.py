"""Passos auxiliares da demonstração executados via API (aguardar fila, exportar lake, gerar relatórios).

Uso: python scripts/demo_steps.py wait [--expected N] | export | reports | summary
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import httpx

from fachadaguard.config import get_settings


def api() -> httpx.Client:
    s = get_settings()
    return httpx.Client(
        base_url=s.api_base_url, timeout=120, headers={"X-API-Token": s.api_token} if s.api_token else {}
    )


def wait(expected: int, timeout: float) -> None:
    c = api()
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        m = c.get("/api/v1/metrics/summary").json()
        jobs = m["jobs"]
        busy = jobs.get("queued", 0) + jobs.get("running", 0)
        print(
            f"  capturas={m['captures']} analisadas={m['captures_analyzed']} fila={busy} "
            f"falhas={jobs.get('failed', 0)}",
            flush=True,
        )
        if busy == 0 and m["captures_analyzed"] >= expected:
            return
        time.sleep(3)
    sys.exit(f"tempo esgotado aguardando análises (esperado {expected})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["seed", "wait", "export", "reports", "summary"])
    ap.add_argument("--expected", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=600)
    a = ap.parse_args()
    c = api()
    if a.step == "seed":
        print(c.post("/api/v1/demo/seed").raise_for_status().json())
    elif a.step == "wait":
        wait(a.expected, a.timeout)
    elif a.step == "export":
        print(json.dumps(c.post("/api/v1/analytics/export").raise_for_status().json(), indent=2))
    elif a.step == "reports":
        for m in c.get("/api/v1/missions").json():
            if m["captures"]:
                r = c.post(f"/api/v1/missions/{m['id']}/reports").raise_for_status().json()
                print(f"  relatório {r['id']} → http://localhost:8000{r['url']}")
    else:
        print(json.dumps(c.get("/api/v1/metrics/summary").json(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
