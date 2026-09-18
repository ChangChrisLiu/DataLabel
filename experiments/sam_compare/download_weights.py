"""Task 1 -- pull the gated SAM 3 / DINOv3 weights into the D: HuggingFace cache.

Reports, per repo: gating status, downloaded size, wall-clock download time.
A repo that is still pending approval (401/403) is reported and skipped rather
than aborting the run, so the rest of the experiment can proceed on whatever is
available.

Run::

    D:\\Anaconda\\envs\\tda\\python.exe -m experiments.sam_compare.download_weights
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from experiments.sam_compare import _env  # noqa: F401  (env side effects first)


def _dir_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def fetch(repo_id: str) -> dict:
    """Snapshot one repo into the D: cache; never raises on a gating error."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    record: dict = {"repo": repo_id}
    t0 = time.perf_counter()
    try:
        local = snapshot_download(repo_id=repo_id, cache_dir=_env._VARS["HF_HUB_CACHE"])
    except GatedRepoError as exc:
        record.update(status="gated", error=f"{type(exc).__name__}: access not granted")
        return record
    except RepositoryNotFoundError as exc:
        record.update(status="not_found", error=f"{type(exc).__name__}")
        return record
    except Exception as exc:  # 401/403 surface as HfHubHTTPError subclasses too
        text = str(exc).splitlines()[0][:200]
        status = "gated" if ("401" in text or "403" in text or "gated" in text.lower()) else "error"
        record.update(status=status, error=f"{type(exc).__name__}: {text}")
        return record
    seconds = time.perf_counter() - t0
    size = _dir_size_bytes(Path(local))
    record.update(
        status="ok",
        path=str(local),
        bytes=size,
        gib=round(size / 1024**3, 3),
        download_s=round(seconds, 1),
    )
    return record


def main() -> int:
    records = [fetch(_env.SAM3_REPO), fetch(_env.DINOV3_REPO)]
    out = _env.OUT_ROOT / "weights.json"
    out.write_text(json.dumps(records, indent=2), encoding="utf-8")
    for rec in records:
        print(json.dumps(rec, ensure_ascii=False), flush=True)
    print(f"[weights] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
