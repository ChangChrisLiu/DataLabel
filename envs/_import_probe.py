"""逐个 import，定位被 Windows Application Control 策略拦截的二进制扩展。"""
import importlib
import traceback

MODS = ["numpy", "scipy", "scipy.ndimage", "cv2", "PIL", "skimage",
        "skimage.measure", "pandas", "openpyxl", "yaml", "pytest", "tqdm",
        "pycocotools", "pycocotools.mask", "torch", "torchvision", "triton",
        "PySide6.QtWidgets", "matplotlib", "hydra", "sam2", "sam3", "dinov3",
        "ultralytics", "sklearn", "sklearn.metrics", "transformers",
        "tokenizers", "safetensors", "timm", "einops", "polars", "av",
        "supervision", "rfdetr"]

blocked, failed, ok = [], [], []
for m in MODS:
    try:
        importlib.import_module(m)
        ok.append(m)
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        if "Application Control" in msg or "blocked" in msg.lower():
            blocked.append((m, msg.splitlines()[-1][:160]))
        else:
            failed.append((m, msg.splitlines()[-1][:160]))

print(f"OK ({len(ok)}): {ok}")
print()
print(f"BLOCKED BY APP-CONTROL ({len(blocked)}):")
for m, e in blocked:
    print(f"  {m}: {e}")
print()
print(f"OTHER FAILURES ({len(failed)}):")
for m, e in failed:
    print(f"  {m}: {e}")
