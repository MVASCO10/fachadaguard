"""Gera o dataset SINTÉTICO (referência, dev, teste e cenário de demo) com manifesto e checksums.

Uso: python scripts/generate_demo_dataset.py [--out data/synthetic] [--verify]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fachadaguard.vision.dataset import verify_dataset, write_dataset  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "synthetic")
    ap.add_argument("--verify", action="store_true", help="apenas confere checksums do manifesto existente")
    a = ap.parse_args()
    if a.verify:
        problems = verify_dataset(a.out)
        print("íntegro" if not problems else "\n".join(problems))
        sys.exit(1 if problems else 0)
    manifest = write_dataset(a.out)
    counts = {k: len(v) for k, v in manifest["splits"].items()}
    print(
        json.dumps(
            {
                "out": str(a.out),
                "files": len(manifest["files"]),
                "splits": counts,
                "reference_set_hash": manifest["reference_set_hash"],
            },
            indent=2,
        )
    )
    assert not verify_dataset(a.out)


if __name__ == "__main__":
    main()
