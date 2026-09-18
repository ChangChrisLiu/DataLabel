"""Pin every cache/temp directory to D: before torch / HF / sam are imported.

Import this module *first* in every script of this experiment::

    from experiments.sam_compare import _env  # noqa: F401  (side effect)

The project rule is that nothing may be written to C:. Environment variables set
by ``envs/activate_tda.sh`` only live in the shell that sourced it, and the
agent harness starts a fresh shell per command, so the variables are set here in
process instead. ``huggingface_hub`` reads ``HF_HUB_CACHE`` at *import* time, so
this has to run before it is imported -- hence a module with import side effects
rather than a function somebody might forget to call.
"""
from __future__ import annotations

import os
from pathlib import Path

D_ROOT = Path("D:/DataSet")

#: Written to by the scripts; big artefacts stay out of git.
OUT_ROOT = D_ROOT / "experiments_out" / "sam_compare"

_VARS = {
    "HF_HOME": D_ROOT / "models" / "hf",
    "HF_HUB_CACHE": D_ROOT / "models" / "hf" / "hub",
    "HUGGINGFACE_HUB_CACHE": D_ROOT / "models" / "hf" / "hub",
    "TRANSFORMERS_CACHE": D_ROOT / "models" / "hf" / "transformers",
    "TORCH_HOME": D_ROOT / "models" / "torch_home",
    "XDG_CACHE_HOME": D_ROOT / ".cache",
    "TMP": D_ROOT / ".cache" / "tmp",
    "TEMP": D_ROOT / ".cache" / "tmp",
    "PIP_CACHE_DIR": D_ROOT / ".cache" / "pip",
    "TRITON_CACHE_DIR": D_ROOT / ".cache" / "triton",
}

for _key, _path in _VARS.items():
    os.environ[_key] = str(_path).replace("/", os.sep)
    _path.mkdir(parents=True, exist_ok=True)

# SAM 2 ships an optional CUDA extension; the env doc says not to build it.
os.environ.setdefault("SAM2_BUILD_CUDA", "0")
os.environ.setdefault("SAM2_BUILD_ALLOW_ERRORS", "1")

OUT_ROOT.mkdir(parents=True, exist_ok=True)

#: Paths the experiment reads.
LS_EXPORT = D_ROOT / "raw_logs" / "labelstudio" / "humansignal_annotated_projects_export.json"
SCAN_CACHE = D_ROOT / "cache" / "scan"
SAM2_CKPT = D_ROOT / "models" / "weights" / "sam2.1_hiera_large.pt"

#: HuggingFace repo ids for the two gated Meta models.
SAM3_REPO = "facebook/sam3"
DINOV3_REPO = "facebook/dinov3-vitl16-pretrain-lvd1689m"
