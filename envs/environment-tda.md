# `tda` conda 环境 —— 创建、版本与已知问题

面向 **PySide6 标注工具 + SAM 2.1 / SAM 3 + Ultralytics YOLO26 + RF-DETR-Seg + DINOv3** 的
桌面主机拆解数据集标注环境。

| 项 | 值 |
|---|---|
| 环境路径 | `D:\Anaconda\envs\tda` |
| 解释器 | `D:\Anaconda\envs\tda\python.exe` — Python **3.11.16** |
| GPU | NVIDIA GeForce RTX 5090 (Blackwell, **sm_120**), 31.8 GiB, 驱动 616.92 |
| 验证日期 | 2026-09-17 |
| 验证报告 | [`verify_report.md`](verify_report.md) |
| 版本锁 | [`requirements-lock.txt`](requirements-lock.txt) (104 个包) |

---

## 0. 每次使用前先激活

**所有缓存 / 权重 / 数据集 / 临时文件都必须落在 D 盘**，靠激活脚本里的环境变量实现。
不要直接 `conda activate tda` 就开跑，否则 pip 缓存、RF-DETR 权重、HF 权重会回到 C 盘。

```powershell
# PowerShell
. D:\DataSet\envs\activate_tda.ps1
```

```bash
# Git Bash
source /d/DataSet/envs/activate_tda.sh
```

脚本设置的变量：

| 变量 | 值 | 作用 |
|---|---|---|
| `PIP_CACHE_DIR` | `D:\DataSet\.cache\pip` | pip 轮子缓存 |
| `TMP` / `TEMP` | `D:\DataSet\.cache\tmp` | 临时文件（pip 构建、torch 编译） |
| `TORCH_HOME` | `D:\DataSet\models\torch_home` | `torch.hub` 权重 |
| `HF_HOME` / `HUGGINGFACE_HUB_CACHE` | `D:\DataSet\models\hf` | HuggingFace 权重（SAM 3 / DINOv3） |
| `YOLO_CONFIG_DIR` | `D:\DataSet\.cache\ultralytics_cfg` | Ultralytics `settings.json` 本体 |
| `RF_HOME` / `ROBOFLOW_HOME` | `D:\DataSet\models\rfdetr` | RF-DETR 预训练权重 |
| `SAM2_BUILD_CUDA=0` | — | 不编译 SAM 2 的自定义 CUDA 扩展 |
| `XDG_CACHE_HOME` | `D:\DataSet\.cache` | 杂项缓存 |

Ultralytics 的三个目录写在 `settings.json` 里（不是环境变量），已经设好：

```
datasets_dir = D:\DataSet\models\ultralytics\datasets
weights_dir  = D:\DataSet\models\ultralytics\weights
runs_dir     = D:\DataSet\models\ultralytics\runs
sync         = False        # 关闭匿名遥测上报
```

改动方式（任选其一）：

```powershell
yolo settings datasets_dir=... weights_dir=... runs_dir=...
# 或
D:\Anaconda\envs\tda\python.exe D:\DataSet\envs\_set_ultralytics_dirs.py
```

---

## 1. 从零重建步骤

```powershell
# 1) 建环境（务必用 PowerShell；Git Bash 会吃掉反斜杠，见「已知问题 1」）
D:\Anaconda\Scripts\conda.exe create -p D:\Anaconda\envs\tda python=3.11 -y

# 2) 固定所有缓存到 D 盘
. D:\DataSet\envs\activate_tda.ps1

$PY = "D:\Anaconda\envs\tda\python.exe"

# 3) PyTorch —— 必须 cu128 构建，5090 是 sm_120，需 torch >= 2.7
& $PY -m pip install --upgrade pip setuptools wheel
& $PY -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 4) 标注工具基础栈
& $PY -m pip install PySide6 numpy opencv-python pillow scipy scikit-image `
                     openpyxl pyyaml pytest tqdm
#    ↓ 注意版本钉死，原因见「已知问题 3」
& $PY -m pip install "pandas==2.3.3" "scikit-learn==1.7.2"

# 5) COCO 标注格式（Windows 有官方轮子，无需 VS Build Tools）
& $PY -m pip install pycocotools

# 6) SAM 2.1
& $PY -m pip install "git+https://github.com/facebookresearch/sam2.git"

# 7) Ultralytics（YOLO26）+ RF-DETR
& $PY -m pip install ultralytics rfdetr

# 8) SAM 3 —— 必须 --no-deps，原因见「已知问题 4」
& $PY -m pip install "timm>=1.0.17" "ftfy==6.1.1" regex "iopath>=0.1.10" huggingface_hub einops
& $PY -m pip install triton-windows                       # sam3 需要 triton，见「已知问题 5」
& $PY -m pip install --no-deps "git+https://github.com/facebookresearch/sam3.git"

# 9) DINOv3
& $PY -m pip install "git+https://github.com/facebookresearch/dinov3.git"

# 10) Ultralytics 目录指向 D 盘
& $PY D:\DataSet\envs\_set_ultralytics_dirs.py

# 11) 验证
& $PY D:\DataSet\envs\verify_env.py
```

### 权重下载

```powershell
# SAM 2.1 large（无门控，Meta 公开直链）
curl.exe -L --retry 3 -o D:\DataSet\models\weights\sam2.1_hiera_large.pt `
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

`yolo26n-seg.pt`（6.7 MB）与 `rf-detr-seg-preview.pt`（130 MB）在首次使用时自动下载，
分别落到 `D:\DataSet\models\weights\` 与 `D:\DataSet\models\rfdetr\`。

---

## 2. 精确版本（关键包）

完整清单见 `requirements-lock.txt`。

| 包 | 版本 |
|---|---|
| python | 3.11.16 |
| **torch** | **2.11.0+cu128** |
| **torchvision** | **0.26.0+cu128** |
| triton-windows | 3.8.0.post28 (提供 `triton` 3.8.0) |
| numpy | 2.4.6 |
| **PySide6** | **6.11.2** (Qt 6.11.2) |
| opencv-python | 5.0.0.93 |
| pillow | 12.3.0 |
| scipy | 1.17.1 |
| scikit-image | 0.26.0 |
| **pandas** | **2.3.3** ← 钉死 |
| **scikit-learn** | **1.7.2** ← 钉死 |
| openpyxl | 3.1.5 |
| pyyaml | 6.0.3 |
| pytest | 9.1.1 |
| tqdm | 4.70.1 |
| pycocotools | 2.0.11 |
| **SAM-2** | `git+…/sam2.git@2b90b9f` (1.0) |
| **sam3** | `git+…/sam3.git@660a5e9` (0.1.0) |
| **dinov3** | `git+…/dinov3.git@6876159` (0.0.1) |
| **ultralytics** | **8.4.155** |
| **rfdetr** | **1.10.1** |
| transformers | 5.17.0 |
| timm | 1.0.29 |
| hydra-core | 1.3.7 / omegaconf 2.3.1 |

`torch.cuda.get_arch_list()` = `['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']` —— 含 **sm_120**，
4096×4096 矩阵乘无任何架构告警，fp32 2.03 ms / fp16 0.71 ms。

---

## 3. 已知问题与解决方法

### 1) Git Bash 会吃掉 conda 命令里的反斜杠

```bash
# 错误示范：真的在 D:\DataSet\ 下建出了一个叫 "Anacondaenvstda" 的目录
/d/Anaconda/Scripts/conda.exe create -p D:\\Anaconda\\envs\\tda python=3.11 -y
```

**解决**：conda 建环境一律用 PowerShell；Git Bash 里只用 `/d/Anaconda/envs/tda/python.exe -m pip …`
这种正斜杠形式。（本次误建的目录已用 `conda remove -p … --all` 清掉。）

### 2) Windows Application Control 策略拦截部分新发布的 `.pyd`

症状：

```
ImportError: DLL load failed while importing tslib:
  An Application Control policy has blocked this file.
ImportError: DLL load failed while importing _argkmin_classmode:
  An Application Control policy has blocked this file.
```

命中的是 **pandas 3.0.5** 与 **scikit-learn 1.9.1**——两个刚发布不久、还没有信誉记录的轮子。
`sklearn.metrics` 被拦会连锁拖垮 `transformers.generation` → **`rfdetr` 完全无法 import**。

**解决**：降到成熟版本 `pandas==2.3.3` + `scikit-learn==1.7.2`，全部恢复正常。
这不是绕过安全策略，只是换一个已有信誉的合法发行版本。
**不要**为此去改系统的 WDAC / 智能应用控制设置。

> 复现检查脚本：`D:\DataSet\envs\_import_probe.py`（逐个 import，把被策略拦截的模块单列出来）。
> 以后升级 pandas / sklearn / 任何带 C 扩展的包之后，建议先跑一遍它。

### 3) `pycocotools` 在 Windows 上不需要 VS Build Tools

PyPI 已有官方 `cp311-win_amd64` 轮子（2.0.11），直接 `pip install pycocotools` 即可，
**不需要** `pycocotools-windows`，也不需要源码编译。RLE 编解码自检通过。

### 4) SAM 3 的 `numpy>=1.26,<2` 硬上界会砸掉整个栈

`sam3` 的 `pyproject.toml` 写死 `numpy>=1.26,<2`，直接 `pip install` 会把 numpy 从 2.4.6
降到 1.26.4，进而与 torch 2.11 / opencv 5 / pandas 2.3 全线冲突。

**解决**：用 `--no-deps` 安装 sam3，手动补齐它真正需要的依赖：

```powershell
& $PY -m pip install "timm>=1.0.17" "ftfy==6.1.1" regex "iopath>=0.1.10" huggingface_hub einops triton-windows
& $PY -m pip install --no-deps "git+https://github.com/facebookresearch/sam3.git"
```

实测 **sam3 在 numpy 2.4.6 下 import 与建模均正常**，那个 `<2` 上界是保守写法。

### 5) SAM 3 未声明的依赖：`einops` 和 `triton`

`sam3/sam/rope.py` 要 `einops`，`sam3/model/edt.py` 要 `triton`，但 `pyproject.toml` 里都没写。
Windows 上没有官方 `triton` 轮子，用社区维护的 **`triton-windows`**（装上后 `import triton` 正常，版本 3.8.0）。

### 6) SAM 2 的自定义 CUDA 扩展

设 `SAM2_BUILD_CUDA=0` 跳过编译，mask 连通域后处理退回 CPU 实现。
对图像级标注（`SAM2ImagePredictor`）没有可感知影响，推理全程仍在 GPU。

### 7) Ultralytics 训练结束后 `model.device` 显示 `cpu`

这是正常行为——训练完成后 ultralytics 会把模型搬回 CPU。
要确认真的用了 GPU，看 **`model.trainer.device`**（本次为 `cuda:0`），或看训练期间的显存占用。

### 8) SAM 3 / DINOv3 权重受 HuggingFace 许可门控

两者的 HF 仓库都是 `gated=manual`（需人工审批），未登录时报 401 `GatedRepoError`。
**代码包已装好、可 import，只差权重。** 需要用户本人操作：

1. 浏览器登录 HuggingFace，打开
   <https://huggingface.co/facebook/sam3> 与
   <https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m>，
   点 *Agree and access repository* 接受 Meta 许可并等待审批。
2. 在 <https://huggingface.co/settings/tokens> 生成 read token。
3. 本机写入 token（token 由用户自己输入，不要落到脚本或仓库里）：
   ```powershell
   . D:\DataSet\envs\activate_tda.ps1
   D:\Anaconda\envs\tda\Scripts\hf.exe auth login
   ```
4. 之后权重会自动下到 `D:\DataSet\models\hf\hub`（`HF_HOME` 已指向 D 盘）。

离线备选：把拿到的 `sam3.pt` 放进 `D:\DataSet\models\weights\`，
`verify_env.py` 会自动走 `build_sam3_image_model(checkpoint_path=…)` 加载。

### 9) RF-DETR 默认把权重下到 `C:\Users\<user>\.roboflow\models`

**解决**：设 `RF_HOME`（或别名 `ROBOFLOW_HOME`）直接指向目标目录——
注意它就是 models 目录本身，不是它的父目录：

```powershell
$Env:RF_HOME = "D:\DataSet\models\rfdetr"
```

### 10) RF-DETR 加载时的 DINOv2 backbone 告警

```
Using a different number of positional encodings than DINOv2 …
Using patch size 12 instead of 14 …
```

预期行为——RF-DETR-Seg-Preview 用的是改过 patch size 的 backbone，
权重从它自己的 checkpoint 加载，不从 DINOv2 拉。可忽略。

---

## 4. 性能基线（RTX 5090，2026-09-17）

测试图：`scanner_crop_native.png`（900×900 主机内部局部）。

| 项目 | 耗时 | 峰值显存 |
|---|---|---|
| torch 4096² matmul fp32 | 2.03 ms (~68 TFLOP/s) | — |
| torch 4096² matmul fp16 | 0.71 ms (~194 TFLOP/s) | — |
| SAM 2.1-L `set_image` 900² | **33.2 ms** | 1292 MiB |
| SAM 2.1-L `set_image` 1600² | **30.8 ms** | 同上 |
| SAM 2.1-L `predict` 点提示 | **7.6 ms** | 同上 |
| SAM 2.1-L `predict` 框提示 | **17.3 ms** | 同上 |
| YOLO26n-seg `predict` (imgsz 640) | **26.0 ms** 端到端 / 7.5 ms 纯推理 | 70 MiB |
| YOLO26n-seg 训练 1 epoch (coco8-seg, batch 4) | 11.6 s | 940 MiB |
| RF-DETR-Seg-Preview `predict` | **22.2 ms** | 275 MiB |

> **为什么 1600² 不比 900² 慢？** SAM 2 内部一律把输入 resize 到 1024×1024，
> GPU 图像编码耗时与原图尺寸无关，差异只来自 CPU 端 resize/归一化，量级在噪声范围内。
> 也就是说：**标注工具可以放心喂原始分辨率图，`set_image` 成本基本恒定 ~33 ms。**

交互标注的实际体感：一次 `set_image`（~33 ms）之后，每加一个点提示只要 ~8 ms，
完全够做实时交互式分割。

### 生成的可视化

| 文件 | 内容 |
|---|---|
| `sam2_point_overlay.png` | 点提示 (785, 215)，命中风扇轮毂，预测 IoU 0.969 |
| `sam2_box_overlay.png` | 框提示 [660, 95, 899, 335]，分出整个 CPU 散热器，预测 IoU 0.977 |
| `rfdetr_overlay.png` | RF-DETR-Seg-Preview 在 COCO 类别上的输出（该图只检出 1 个实例——COCO 没有 PC 部件类别，属预期） |
| `_bench_1600.png` | 由 900² LANCZOS 上采样得到的 1600² 计时用图 |

---

## 5. 目录约定

```
D:\DataSet\
├── envs\                               # 本目录
│   ├── activate_tda.ps1 / .sh          # 激活脚本（每次必用）
│   ├── verify_env.py                   # 完整验证 + 计时脚本
│   ├── verify_report.md                # 最近一次验证输出
│   ├── environment-tda.md              # 本文件
│   ├── requirements-lock.txt           # pip freeze
│   ├── _set_ultralytics_dirs.py        # 把 ultralytics 目录指到 D 盘
│   ├── _check_hf_gating.py             # 查 HF 权重门控状态
│   ├── _import_probe.py                # 查被 App Control 拦截的模块
│   └── *.png                           # 验证产生的叠加图
├── models\
│   ├── weights\                        # sam2.1_hiera_large.pt, yolo26n-seg.pt
│   ├── rfdetr\                         # rf-detr-seg-preview.pt        (RF_HOME)
│   ├── hf\                             # HuggingFace 缓存              (HF_HOME)
│   ├── torch_home\                     # torch.hub 缓存                (TORCH_HOME)
│   └── ultralytics\
│       ├── datasets\                   # coco8-seg 等
│       ├── weights\
│       └── runs\                       # 训练产物
└── .cache\
    ├── pip\                            # pip 缓存（从 C 盘迁来，3.27 GB）
    ├── tmp\                            # TMP / TEMP
    └── ultralytics_cfg\                # Ultralytics settings.json
```

---

## 6. 仍在 C 盘的内容

见 `verify_report.md` 第 8 节的扫描结果。截至验证时：

| 位置 | 状态 | 说明 |
|---|---|---|
| `C:\Users\61908\AppData\Local\pip\cache` | 已清空并迁走 | 1196 文件 / 3.27 GB → `D:\DataSet\.cache\pip` |
| `C:\Users\61908\.roboflow\models` | 已迁走，剩空目录 | `rf-detr-seg-preview.pt` → `D:\DataSet\models\rfdetr\`。空目录按「不删除」约束保留 |
| `C:\Users\61908\AppData\Roaming\Ultralytics` | 从未生成 | `YOLO_CONFIG_DIR` 生效 |
| `C:\Users\61908\.cache\huggingface`、`.cache\torch`、`.triton` | 从未生成 | `HF_HOME` / `TORCH_HOME` / `XDG_CACHE_HOME` 生效 |
| **`D:\Anaconda\envs\tda`（6.13 GB）** | 在 D 盘 | 环境本体 |

**无法搬到 D 盘的两类**：

1. **测试图片本身** —— `C:\Users\61908\AppData\Local\Temp\claude\…\scratchpad\views\scanner_crop_native.png`
   是会话临时目录，由外部工具管理。正式使用时换成 D 盘上的数据路径即可。
2. **未 source 激活脚本时的默认缓存** —— 环境变量只在当前 shell 生效。
   如果直接 `conda activate tda` 就跑 pip，缓存会重新在 C 盘生成。
   要彻底根治（会影响 base / CV 环境，请自行决定是否执行）：

   ```powershell
   D:\Anaconda\envs\tda\python.exe -m pip config set global.cache-dir D:\DataSet\.cache\pip
   ```

   这会写 `C:\Users\61908\AppData\Roaming\pip\pip.ini`，让**所有** pip 调用都用 D 盘缓存。
   撤销：`pip config unset global.cache-dir`。本次**没有**擅自执行。
