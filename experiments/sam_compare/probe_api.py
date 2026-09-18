"""Throwaway probe: which SAM 3 / DINOv3 entry points actually work on this box.

Kept in the repo because the answers (gating status, which predictor class the
interactive task needs, whether transformers ships a SAM 3 processor) are the
reason the rest of the experiment is wired the way it is.
"""
from __future__ import annotations

from experiments.sam_compare import _env  # noqa: F401  (env side effects first)


def probe_gating() -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    for repo in [
        _env.DINOV3_REPO,
        "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "facebook/dinov3-vits16-pretrain-lvd1689m",
        _env.SAM3_REPO,
    ]:
        try:
            info = api.model_info(repo)
            print(f"[gating] {repo}: OK gated={getattr(info, 'gated', None)}")
        except Exception as exc:
            print(f"[gating] {repo}: {type(exc).__name__}: {str(exc).splitlines()[0][:160]}")


def probe_transformers_sam3() -> None:
    import transformers

    print(f"[tf] transformers {transformers.__version__}")
    for name in ("Sam3Model", "Sam3Processor", "Sam3TrackerModel", "Sam3VideoModel"):
        print(f"[tf] has {name}: {hasattr(transformers, name)}")


def probe_dinov3_transformers() -> None:
    import transformers

    for name in ("DINOv3ViTModel", "Dinov3ViTModel", "AutoModel"):
        print(f"[tf] has {name}: {hasattr(transformers, name)}")


if __name__ == "__main__":
    probe_gating()
    probe_transformers_sam3()
    probe_dinov3_transformers()
