"""Model services (Qt-free): SAM 2.1 interactive segmentation, later detectors.

Nothing in this package may import Qt. UI-facing wrappers live in ``tda.ui``.
"""
from tda.models.sam_service import (
    SamQueue,
    SamRequest,
    SamResult,
    SamService,
    default_checkpoint,
)

__all__ = [
    "SamQueue",
    "SamRequest",
    "SamResult",
    "SamService",
    "default_checkpoint",
]
