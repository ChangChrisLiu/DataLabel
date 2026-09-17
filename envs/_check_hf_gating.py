"""检查 HuggingFace 上 SAM 3 / DINOv3 权重仓库的可访问性（是否受许可门控）。
不尝试绕过任何许可，只做只读探测。"""
import os
import json

os.environ.setdefault("HF_HOME", r"D:\DataSet\models\hf")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", r"D:\DataSet\models\hf\hub")

from huggingface_hub import HfApi, whoami  # noqa: E402

api = HfApi()

try:
    me = whoami()
    print(f"HF 登录状态: 已登录 as {me.get('name')}")
except Exception as e:
    print(f"HF 登录状态: 未登录 ({type(e).__name__})")

REPOS = [
    "facebook/sam3",
    "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "facebook/sam2.1-hiera-large",
]

for r in REPOS:
    try:
        info = api.model_info(r, files_metadata=False)
        files = [s.rfilename for s in info.siblings][:12]
        print(f"\n[OK] {r}")
        print(f"     gated={getattr(info, 'gated', None)}  "
              f"private={info.private}  downloads={info.downloads}")
        print(f"     files: {files}")
    except Exception as e:
        msg = str(e).replace("\n", " ")[:300]
        print(f"\n[BLOCKED] {r}")
        print(f"     {type(e).__name__}: {msg}")
