"""
tda 环境验证脚本
------------------------------------------------------------------
验证内容:
  1. PyTorch / CUDA (RTX 5090, sm_120) + 4096x4096 矩阵乘
  2. SAM 2.1 large: 点 / 框提示推理, set_image 与 predict 计时, 峰值显存
  3. Ultralytics YOLO: 推理计时 + 1 epoch 训练冒烟, 峰值显存
  4. RF-DETR-Seg: 权重加载 + 推理计时, 峰值显存
  5. SAM 3 / DINOv3: 代码包可用性 + HuggingFace 权重门控状态
  6. 其它依赖导入自检
  7. C 盘残留扫描

输出:
  D:\\DataSet\\envs\\verify_report.md
  D:\\DataSet\\envs\\sam2_point_overlay.png / sam2_box_overlay.png
  D:\\DataSet\\envs\\rfdetr_overlay.png
  D:\\DataSet\\envs\\sam3_screw_overlay.png   (仅当 SAM 3 权重可用)

用法 (先 source activate_tda.ps1 / .sh):
  D:\\Anaconda\\envs\\tda\\python.exe D:\\DataSet\\envs\\verify_env.py
"""

import os
import sys
import time
import json
import warnings
import traceback
from pathlib import Path

# ---- 所有缓存/临时文件强制在 D 盘 (即使没 source 激活脚本也生效) ----
os.environ.setdefault("SAM2_BUILD_CUDA", "0")
os.environ.setdefault("TORCH_HOME", r"D:\DataSet\models\torch_home")
os.environ.setdefault("HF_HOME", r"D:\DataSet\models\hf")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", r"D:\DataSet\models\hf\hub")
os.environ.setdefault("YOLO_CONFIG_DIR", r"D:\DataSet\.cache\ultralytics_cfg")
os.environ.setdefault("RF_HOME", r"D:\DataSet\models\rfdetr")
os.environ.setdefault("ROBOFLOW_HOME", r"D:\DataSet\models\rfdetr")
os.environ.setdefault("TMP", r"D:\DataSet\.cache\tmp")
os.environ.setdefault("TEMP", r"D:\DataSet\.cache\tmp")

OUT_DIR = Path(r"D:\DataSet\envs")
WEIGHTS = Path(r"D:\DataSet\models\weights")
OUT_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS.mkdir(parents=True, exist_ok=True)
REPORT = OUT_DIR / "verify_report.md"

IMG_900 = Path(
    r"C:\Users\61908\AppData\Local\Temp\claude\D--DataSet"
    r"\3f6dd74d-e1dc-46a6-bbc2-2fe9e89c0835\scratchpad\views\scanner_crop_native.png"
)
IMG_1600 = OUT_DIR / "_bench_1600.png"

SAM2_CKPT = WEIGHTS / "sam2.1_hiera_large.pt"
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"

# 主机内 CPU 风扇 (右上角) 的点提示与框提示
POINT = [785, 215]
BOX = [660, 95, 899, 335]

lines: list[str] = []
results: dict = {}


def log(s: str = "") -> None:
    print(s, flush=True)
    lines.append(s)


def sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timeit(fn, n=1):
    """返回 (结果, 平均毫秒)。调用前后做 cuda 同步。"""
    sync()
    t0 = time.perf_counter()
    out = None
    for _ in range(n):
        out = fn()
    sync()
    return out, (time.perf_counter() - t0) * 1000.0 / n


def vram(reset=False):
    import torch
    if not torch.cuda.is_available():
        return 0.0, 0.0
    peak = torch.cuda.max_memory_allocated() / 1024**2
    res = torch.cuda.max_memory_reserved() / 1024**2
    if reset:
        torch.cuda.reset_peak_memory_stats()
    return peak, res


def flush_report():
    REPORT.write_text("\n".join(lines), encoding="utf-8")


# ===================================================================
# 1. PyTorch / CUDA
# ===================================================================
def section_torch():
    log("## 1. PyTorch / CUDA")
    log()
    import torch
    import torchvision

    results["torch_version"] = torch.__version__
    results["torchvision_version"] = torchvision.__version__
    avail = torch.cuda.is_available()
    results["cuda_available"] = avail

    log(f"- torch: `{torch.__version__}`")
    log(f"- torchvision: `{torchvision.__version__}`")
    log(f"- torch 编译所用 CUDA: `{torch.version.cuda}`")
    log(f"- `torch.cuda.is_available()`: **{avail}**")
    log(f"- 编译支持的架构: `{torch.cuda.get_arch_list()}`")

    if not avail:
        log("- **GPU 不可用, 后续 GPU 测试跳过**")
        log()
        return False

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    results["device_name"] = name
    results["capability"] = f"sm_{cap[0]}{cap[1]}"
    log(f"- 设备名: **{name}**")
    log(f"- `get_device_capability()`: **{cap}** → **sm_{cap[0]}{cap[1]}**")
    log(f"- 显存总量: {total:.2f} GiB")
    log()

    log("### 4096x4096 矩阵乘")
    log()
    torch.cuda.reset_peak_memory_stats()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        a = torch.randn(4096, 4096, device="cuda")
        b = torch.randn(4096, 4096, device="cuda")
        timeit(lambda: a @ b)                      # 预热
        c, ms32 = timeit(lambda: a @ b, n=20)
        ah, bh = a.half(), b.half()
        timeit(lambda: ah @ bh)
        ch, ms16 = timeit(lambda: ah @ bh, n=20)
        caught = [str(x.message) for x in w]

    ref = a[:64].cpu() @ b.cpu()
    err = (c[:64].cpu() - ref).abs().max().item()
    log(f"- fp32: **{ms32:.2f} ms** (~{2*4096**3/(ms32/1000)/1e12:.1f} TFLOP/s)")
    log(f"- fp16: **{ms16:.2f} ms** (~{2*4096**3/(ms16/1000)/1e12:.1f} TFLOP/s)")
    log(f"- 正确性: 与 CPU 参考最大绝对误差 `{err:.3e}`, "
        f"有限值={bool(torch.isfinite(c).all())}")
    arch_warn = [x for x in caught if "sm_" in x or "capability" in x.lower()
                 or "not compatible" in x.lower() or "no kernel image" in x.lower()]
    if arch_warn:
        log(f"- ⚠️ **架构相关告警**: {arch_warn}")
    else:
        log("- ✅ **无 sm_120 / 架构不兼容告警或错误**")
    if caught:
        log(f"- 其它 warning: `{caught}`")
    p, r = vram(reset=True)
    log(f"- matmul 峰值显存: allocated {p:.0f} MiB / reserved {r:.0f} MiB")
    results["matmul_fp32_ms"] = round(ms32, 2)
    results["matmul_fp16_ms"] = round(ms16, 2)
    results["matmul_arch_warnings"] = arch_warn
    del a, b, c, ah, bh, ch
    torch.cuda.empty_cache()
    log()
    return True


# ===================================================================
# 工具: mask 叠加
# ===================================================================
def overlay(img_np, masks, path, title="", pt=None, box=None, scores=None,
            boxes=None, labels=None):
    import numpy as np
    from PIL import Image, ImageDraw

    base = img_np.astype(np.float32).copy()
    palette = [(30, 144, 255), (255, 99, 71), (60, 220, 120), (255, 200, 40),
               (180, 100, 255), (0, 220, 220)]
    if masks is not None:
        for i, m in enumerate(masks):
            m = np.asarray(m).astype(bool)
            if m.shape != base.shape[:2]:
                continue
            col = np.array(palette[i % len(palette)], dtype=np.float32)
            base[m] = base[m] * 0.55 + col * 0.45
    out = Image.fromarray(base.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(out)
    if boxes is not None:
        for i, bb in enumerate(boxes):
            d.rectangle([float(v) for v in bb],
                        outline=palette[i % len(palette)], width=3)
            if labels is not None and i < len(labels):
                d.text((float(bb[0]) + 3, float(bb[1]) + 3), str(labels[i]),
                       fill=(255, 255, 0))
    if box is not None:
        d.rectangle([float(v) for v in box], outline=(255, 215, 0), width=4)
    if pt is not None:
        x, y = pt
        d.ellipse([x - 9, y - 9, x + 9, y + 9], fill=(255, 60, 60),
                  outline=(255, 255, 255), width=3)
    sub = f"  score={scores}" if scores is not None else ""
    d.text((10, 10), f"{title}{sub}", fill=(255, 255, 0))
    out.save(path)


# ===================================================================
# 2. SAM 2.1
# ===================================================================
def make_1600():
    from PIL import Image
    if not IMG_1600.exists():
        Image.open(IMG_900).convert("RGB").resize(
            (1600, 1600), Image.LANCZOS).save(IMG_1600)


def section_sam2(gpu_ok):
    log("## 2. SAM 2.1 (hiera large)")
    log()
    import numpy as np
    import torch
    from PIL import Image

    if not SAM2_CKPT.exists():
        log(f"- ❌ 权重不存在: `{SAM2_CKPT}`")
        log()
        return

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    log(f"- 权重: `{SAM2_CKPT}` ({SAM2_CKPT.stat().st_size / 1024**2:.0f} MiB)")
    log(f"- config: `{SAM2_CFG}`")
    log(f"- `SAM2_BUILD_CUDA={os.environ.get('SAM2_BUILD_CUDA')}` "
        "→ 跳过自定义 CUDA 扩展, 连通域后处理走 CPU")

    device = "cuda" if gpu_ok else "cpu"
    if gpu_ok:
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    model = build_sam2(SAM2_CFG, str(SAM2_CKPT), device=device)
    predictor = SAM2ImagePredictor(model)
    load_ms = (time.perf_counter() - t0) * 1000
    log(f"- 模型加载: **{load_ms:.0f} ms**")
    results["sam2_load_ms"] = round(load_ms)
    log()

    img = np.array(Image.open(IMG_900).convert("RGB"))
    log(f"### 900x900 图 `{IMG_900.name}` (shape={img.shape})")
    log()

    ac = torch.autocast("cuda", dtype=torch.bfloat16) if gpu_ok \
        else torch.autocast("cpu", dtype=torch.bfloat16)

    with torch.inference_mode(), ac:
        predictor.set_image(img)                                     # 预热
        predictor.predict(point_coords=np.array([POINT], dtype=np.float32),
                          point_labels=np.array([1]), multimask_output=True)
        sync()

        _, set_ms = timeit(lambda: predictor.set_image(img), n=3)

        def _pt():
            return predictor.predict(
                point_coords=np.array([POINT], dtype=np.float32),
                point_labels=np.array([1]), multimask_output=True)
        (m_pt, s_pt, _), pt_ms = timeit(_pt, n=5)

        def _bx():
            return predictor.predict(
                box=np.array(BOX, dtype=np.float32)[None, :],
                multimask_output=False)
        (m_bx, s_bx, _), bx_ms = timeit(_bx, n=5)

    best = int(np.argmax(s_pt))
    log(f"- `set_image` (900x900, 预热后 3 次均值): **{set_ms:.1f} ms**")
    log(f"- `predict` 点提示 `{POINT}` (5 次均值): **{pt_ms:.1f} ms** — "
        f"3 候选 mask, 预测 IoU = {np.round(s_pt, 3).tolist()}, "
        f"最佳 mask 面积 {int(m_pt[best].sum())} px")
    log(f"- `predict` 框提示 `{BOX}` (5 次均值): **{bx_ms:.1f} ms** — "
        f"预测 IoU = {float(s_bx[0]):.3f}, 面积 {int(m_bx[0].sum())} px")
    results["sam2_set_image_900_ms"] = round(set_ms, 1)
    results["sam2_predict_point_ms"] = round(pt_ms, 1)
    results["sam2_predict_box_ms"] = round(bx_ms, 1)

    p1, p2 = OUT_DIR / "sam2_point_overlay.png", OUT_DIR / "sam2_box_overlay.png"
    overlay(img, [m_pt[best]], p1, "SAM2.1 point prompt",
            pt=POINT, scores=f"{float(s_pt[best]):.3f}")
    overlay(img, [m_bx[0]], p2, "SAM2.1 box prompt",
            box=BOX, scores=f"{float(s_bx[0]):.3f}")
    log(f"- 叠加图: `{p1.name}`, `{p2.name}` (均在 `{OUT_DIR}`)")

    make_1600()
    img16 = np.array(Image.open(IMG_1600).convert("RGB"))
    with torch.inference_mode(), ac:
        predictor.set_image(img16)
        sync()
        _, set16_ms = timeit(lambda: predictor.set_image(img16), n=3)
    log(f"- `set_image` (1600x1600, 由 900x900 LANCZOS 上采样得到 "
        f"`{IMG_1600.name}`): **{set16_ms:.1f} ms**")
    log("  - 说明: SAM2 内部一律 resize 到 1024x1024, 因此 900 与 1600 输入的 "
        "GPU 图像编码耗时几乎相同, 差值主要来自 CPU 端 resize/归一化。")
    results["sam2_set_image_1600_ms"] = round(set16_ms, 1)

    if gpu_ok:
        p, r = vram(reset=True)
        log(f"- **SAM 2.1 large 峰值显存: allocated {p:.0f} MiB "
            f"({p/1024:.2f} GiB) / reserved {r:.0f} MiB**")
        results["sam2_peak_vram_mib"] = round(p)

    del predictor, model
    if gpu_ok:
        torch.cuda.empty_cache()
    log()


# ===================================================================
# 3. Ultralytics
# ===================================================================
def section_yolo(gpu_ok):
    log("## 3. Ultralytics YOLO")
    log()
    import torch
    import ultralytics
    from ultralytics import YOLO, settings

    log(f"- ultralytics: `{ultralytics.__version__}`")
    log(f"- settings 文件: `{settings.file}`")
    log(f"- datasets_dir=`{settings['datasets_dir']}`  "
        f"weights_dir=`{settings['weights_dir']}`  runs_dir=`{settings['runs_dir']}`")
    os.chdir(WEIGHTS)   # 自动下载的 .pt 落到 D:\DataSet\models\weights

    model, model_name = None, None
    for cand in ["yolo26n-seg.pt", "yolo11n-seg.pt"]:
        try:
            model = YOLO(cand)
            model_name = cand
            log(f"- ✅ 加载模型: **{cand}**")
            break
        except Exception as e:
            log(f"- ⚠️ `{cand}` 不可用 → `{type(e).__name__}: {str(e)[:220]}`")
    if model is None:
        log("- ❌ 无可用 YOLO 分割模型")
        log()
        return
    results["yolo_model"] = model_name

    if gpu_ok:
        torch.cuda.reset_peak_memory_stats()
    dev = 0 if gpu_ok else "cpu"
    model.predict(str(IMG_900), device=dev, imgsz=640, verbose=False)   # 预热
    r, pred_ms = timeit(
        lambda: model.predict(str(IMG_900), device=dev, imgsz=640, verbose=False), n=5)
    nbox = 0 if r[0].boxes is None else len(r[0].boxes)
    log(f"- `predict` (`{IMG_900.name}`, imgsz=640, 5 次均值): **{pred_ms:.1f} ms** "
        f"→ 检出 {nbox} 个实例")
    log(f"  - 内部分解 (ms): `{ {k: round(v, 2) for k, v in r[0].speed.items()} }`")
    results["yolo_predict_ms"] = round(pred_ms, 1)
    results["yolo_predict_speed"] = {k: round(v, 2) for k, v in r[0].speed.items()}
    if gpu_ok:
        p, _ = vram(reset=True)
        log(f"- YOLO 推理峰值显存: allocated **{p:.0f} MiB**")
        results["yolo_infer_peak_vram_mib"] = round(p)

    log()
    log("### 训练冒烟测试 (coco8-seg, 1 epoch, imgsz=640)")
    log()
    try:
        if gpu_ok:
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        tr = YOLO(model_name)
        res = tr.train(data="coco8-seg.yaml", epochs=1, imgsz=640, device=dev,
                       batch=4, workers=0, plots=False, val=True,
                       name="tda_smoke", exist_ok=True, verbose=False)
        dt = time.perf_counter() - t0
        # 注意: 训练结束后 ultralytics 会把模型搬回 CPU, 所以要看 trainer.device
        train_dev = str(getattr(getattr(tr, "trainer", None), "device", tr.device))
        on_gpu = "cuda" in train_dev
        log(f"- ✅ 训练完成, 总耗时 **{dt:.1f} s**")
        log(f"- 训练时 `trainer.device` = `{train_dev}` → "
            f"**GPU 训练: {'是' if on_gpu else '否'}**  "
            f"(训练结束后 `model.device` 会被搬回 `{tr.device}`, 属正常行为)")
        try:
            log(f"- 1 epoch 后 mask mAP50-95 = `{res.seg.map:.4f}`, "
                f"mAP50 = `{res.seg.map50:.4f}` (仅冒烟, 数值无意义)")
        except Exception:
            pass
        log(f"- 产物目录: `{res.save_dir}`")
        results["yolo_train_ok"] = True
        results["yolo_train_seconds"] = round(dt, 1)
        results["yolo_train_device"] = train_dev
        if gpu_ok:
            p, r2 = vram(reset=True)
            log(f"- **YOLO 训练峰值显存: allocated {p:.0f} MiB "
                f"({p/1024:.2f} GiB) / reserved {r2:.0f} MiB**")
            results["yolo_train_peak_vram_mib"] = round(p)
    except Exception as e:
        log(f"- ❌ 训练失败: `{type(e).__name__}: {str(e)[:300]}`")
        log("```"); log(traceback.format_exc()[-1800:]); log("```")
        results["yolo_train_ok"] = False
    log()


# ===================================================================
# 4. RF-DETR-Seg
# ===================================================================
def section_rfdetr(gpu_ok):
    log("## 4. RF-DETR (分割)")
    log()
    import numpy as np
    import torch
    from PIL import Image

    try:
        import rfdetr
        from importlib.metadata import version
        log(f"- rfdetr: `{version('rfdetr')}`")
    except Exception as e:
        log(f"- ❌ rfdetr 导入失败: `{type(e).__name__}: {str(e)[:250]}`")
        log()
        return

    os.chdir(WEIGHTS)   # 预训练权重下载到 D:\DataSet\models\weights
    if gpu_ok:
        torch.cuda.reset_peak_memory_stats()

    model, used = None, None
    for cls_name in ["RFDETRSegPreview", "RFDETRSegNano", "RFDETRSegSmall"]:
        try:
            cls = getattr(rfdetr, cls_name)
            t0 = time.perf_counter()
            model = cls()
            load_ms = (time.perf_counter() - t0) * 1000
            used = cls_name
            log(f"- ✅ 加载 **{cls_name}** 预训练权重, 耗时 **{load_ms:.0f} ms**")
            results["rfdetr_model"] = cls_name
            break
        except Exception as e:
            log(f"- ⚠️ `{cls_name}` 加载失败 → `{type(e).__name__}: {str(e)[:250]}`")
    if model is None:
        log("- ❌ 所有 RF-DETR-Seg 变体均无法加载")
        log()
        return

    try:
        pil = Image.open(IMG_900).convert("RGB")
        model.predict(pil, threshold=0.5)                    # 预热
        det, ms = timeit(lambda: model.predict(pil, threshold=0.5), n=5)
        n = len(det) if hasattr(det, "__len__") else 0
        log(f"- `predict` (`{IMG_900.name}`, 5 次均值): **{ms:.1f} ms** → {n} 个实例")
        results["rfdetr_predict_ms"] = round(ms, 1)
        results["rfdetr_detections"] = int(n)

        masks = getattr(det, "mask", None)
        boxes = getattr(det, "xyxy", None)
        conf = getattr(det, "confidence", None)
        labs = None
        if conf is not None and boxes is not None:
            labs = [f"{c:.2f}" for c in conf]
        pth = OUT_DIR / "rfdetr_overlay.png"
        overlay(np.array(pil), masks, pth, f"RF-DETR {used}",
                boxes=boxes, labels=labs)
        log(f"- 叠加图: `{pth.name}`  (mask 输出: "
            f"{'有' if masks is not None else '无'})")
    except Exception as e:
        log(f"- ❌ 推理失败: `{type(e).__name__}: {str(e)[:300]}`")
        log("```"); log(traceback.format_exc()[-1500:]); log("```")

    if gpu_ok:
        p, _ = vram(reset=True)
        log(f"- **RF-DETR 峰值显存: allocated {p:.0f} MiB ({p/1024:.2f} GiB)**")
        results["rfdetr_peak_vram_mib"] = round(p)
    del model
    if gpu_ok:
        torch.cuda.empty_cache()
    log()


# ===================================================================
# 5. SAM 3 / DINOv3
# ===================================================================
def hf_status(repo):
    from huggingface_hub import HfApi
    try:
        info = HfApi().model_info(repo)
        return f"gated=**{getattr(info, 'gated', None)}**, private={info.private}"
    except Exception as e:
        return f"不可读取: {type(e).__name__}: {str(e)[:120]}"


def section_sam3(gpu_ok):
    log("## 5. SAM 3")
    log()
    import numpy as np
    import torch
    from PIL import Image

    try:
        import sam3   # noqa
        log("- ✅ `sam3` 代码包导入成功 (从 GitHub 源码安装)")
    except Exception as e:
        log(f"- ❌ `sam3` 导入失败: `{type(e).__name__}: {str(e)[:250]}`")
        log()
        return

    log(f"- HuggingFace `facebook/sam3` 状态: {hf_status('facebook/sam3')}")

    local = [p for p in WEIGHTS.glob("sam3*.pt")] + \
            [p for p in WEIGHTS.glob("sam3*.safetensors")]
    if local:
        log(f"- 本地权重: {[p.name for p in local]}")

    # 图像级概念分割: build_sam3_image_model + Sam3Processor
    # (无 HF token 且未接受许可时, 这里会报 401 / GatedRepoError)
    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        dev = "cuda" if gpu_ok else "cpu"
        if gpu_ok:
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        ckpt = local[0] if local else None
        model = build_sam3_image_model(device=dev, load_from_HF=ckpt is None,
                                       checkpoint_path=str(ckpt) if ckpt else None)
        log(f"- ✅ SAM 3 image model 构建成功, 耗时 "
            f"**{(time.perf_counter()-t0)*1000:.0f} ms**")

        proc = Sam3Processor(model, device=dev, confidence_threshold=0.5)
        img = np.array(Image.open(IMG_900).convert("RGB"))
        state = proc.set_image(img)
        proc.set_text_prompt("screw", state)          # 预热
        sync()
        state, ms = timeit(lambda: proc.set_text_prompt("screw", state), n=3)

        masks = state.get("masks")
        scores = state.get("scores")
        boxes = state.get("boxes")
        n = 0 if masks is None else len(masks)
        log(f"- 文字提示 `\"screw\"` 概念分割 (3 次均值): **{ms:.1f} ms** "
            f"→ {n} 个实例")
        if n:
            sc = [round(float(s), 3) for s in scores[:10]]
            log(f"  - 置信度 (前 10): `{sc}`")
        pth = OUT_DIR / "sam3_screw_overlay.png"
        overlay(img,
                None if masks is None else [np.asarray(m).squeeze() for m in masks],
                pth, 'SAM3 text prompt "screw"',
                boxes=None if boxes is None else
                [list(map(float, b)) for b in boxes])
        log(f"- 叠加图: `{pth.name}`")
        results["sam3_text_prompt_ms"] = round(ms, 1)
        results["sam3_instances"] = int(n)
        results["sam3_weights"] = "可用"
        if gpu_ok:
            p, _ = vram(reset=True)
            log(f"- **SAM 3 峰值显存: allocated {p:.0f} MiB ({p/1024:.2f} GiB)**")
            results["sam3_peak_vram_mib"] = round(p)
        del proc, model
        if gpu_ok:
            torch.cuda.empty_cache()
    except Exception as e:
        msg = str(e).replace("\n", " ")[:400]
        log(f"- ⚠️ **SAM 3 权重不可用** → `{type(e).__name__}: {msg}`")
        log()
        log("  **需要用户手动操作 (不能绕过许可):**")
        log("  1. 用浏览器登录 HuggingFace, 打开 <https://huggingface.co/facebook/sam3>,")
        log("     点击页面上的 *Agree and access repository* 接受 Meta 的许可 "
            "(该仓库 `gated=manual`, 需人工审批)。")
        log("  2. 在 <https://huggingface.co/settings/tokens> 生成一个 read token。")
        log("  3. 在本机执行 (token 由用户自行输入, 不要写进脚本):")
        log("     ```")
        log("     $Env:HF_HOME=\"D:\\DataSet\\models\\hf\"")
        log("     D:\\Anaconda\\envs\\tda\\Scripts\\hf.exe auth login")
        log("     ```")
        log("  4. 之后 `build_sam3_image_model(load_from_HF=True)` 会把权重下载到 "
            "`D:\\DataSet\\models\\hf\\hub` (HF_HOME 已指向 D 盘)。")
        log("  备选: 若已从别处拿到 `sam3.pt`, 放到 "
            "`D:\\DataSet\\models\\weights\\` 下, 本脚本会自动用 "
            "`checkpoint_path=` 离线加载。")
        results["sam3_weights"] = "gated / 未获授权"
    log()


def section_dinov3(gpu_ok):
    log("## 6. DINOv3")
    log()
    try:
        import dinov3  # noqa
        import dinov3.models  # noqa
        log("- ✅ `dinov3` 代码包导入成功 (从 GitHub 源码安装)")
    except Exception as e:
        log(f"- ❌ `dinov3` 导入失败: `{type(e).__name__}: {str(e)[:250]}`")
        log()
        return

    for repo in ["facebook/dinov3-vitl16-pretrain-lvd1689m",
                 "facebook/dinov3-vitb16-pretrain-lvd1689m",
                 "facebook/dinov3-vits16-pretrain-lvd1689m"]:
        log(f"- HuggingFace `{repo}`: {hf_status(repo)}")

    try:
        import torch
        from transformers import AutoModel
        t0 = time.perf_counter()
        m = AutoModel.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
        log(f"- ✅ 权重加载成功 ({(time.perf_counter()-t0)*1000:.0f} ms), "
            f"参数量 {sum(p.numel() for p in m.parameters())/1e6:.1f} M")
        results["dinov3_weights"] = "可用"
        del m
        torch.cuda.empty_cache() if gpu_ok else None
    except Exception as e:
        msg = str(e).replace("\n", " ")[:350]
        log(f"- ⚠️ **DINOv3 权重不可用** → `{type(e).__name__}: {msg}`")
        log()
        log("  **需要用户手动操作 (不能绕过许可):** 与 SAM 3 相同 —— "
            "登录 HuggingFace 后在 "
            "<https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m> "
            "接受 DINOv3 License (`gated=manual`), 再 `hf auth login` 写入 token。")
        log("  备选: 若已有 Meta 官方直链的 `.pth`, 可用 "
            "`torch.hub.load('facebookresearch/dinov3','dinov3_vitl16', "
            "weights='D:/DataSet/models/weights/xxx.pth')` 离线加载。")
        results["dinov3_weights"] = "gated / 未获授权"
    log()


# ===================================================================
# 7. 其它依赖导入自检
# ===================================================================
def section_imports():
    log("## 7. 依赖导入自检")
    log()
    mods = [
        ("PySide6", "PySide6.QtCore"), ("numpy", "numpy"),
        ("opencv-python", "cv2"), ("pillow", "PIL"), ("scipy", "scipy"),
        ("scikit-image", "skimage"), ("pandas", "pandas"),
        ("openpyxl", "openpyxl"), ("pyyaml", "yaml"), ("pytest", "pytest"),
        ("tqdm", "tqdm"), ("pycocotools", "pycocotools"),
        ("scikit-learn", "sklearn.metrics"), ("transformers", "transformers"),
        ("timm", "timm"), ("triton", "triton"), ("hydra-core", "hydra"),
        ("matplotlib", "matplotlib"), ("supervision", "supervision"),
    ]
    log("| 包 | 状态 | 版本 / 自检 |")
    log("|---|---|---|")
    for disp, mod in mods:
        try:
            m = __import__(mod, fromlist=["x"])
            if mod == "PySide6.QtCore":
                import PySide6
                from PySide6.QtCore import qVersion
                v = f"{PySide6.__version__} (Qt {qVersion()})"
            elif mod == "pycocotools":
                import numpy as np
                from pycocotools import mask as cm
                from pycocotools.coco import COCO  # noqa
                rle = cm.encode(np.asfortranarray((np.eye(20) > 0).astype("uint8")))
                v = f"RLE 编解码 OK (area={int(cm.area(rle))})"
            elif mod == "sklearn.metrics":
                import sklearn
                v = sklearn.__version__
            else:
                v = getattr(m, "__version__", "ok")
            log(f"| {disp} | ✅ | `{v}` |")
        except Exception as e:
            log(f"| {disp} | ❌ | `{type(e).__name__}: {str(e)[:110]}` |")
    log()

    try:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        from PySide6.QtWidgets import QApplication, QLabel
        app = QApplication.instance() or QApplication([])
        w = QLabel("tda")
        w.resize(120, 40)
        log(f"- PySide6 离屏 `QApplication` + `QWidget` 创建: ✅ "
            f"({w.size().width()}x{w.size().height()})")
        del w, app
    except Exception as e:
        log(f"- PySide6 `QApplication` 自检: ❌ `{type(e).__name__}: {e}`")
    log()


# ===================================================================
# 8. C 盘残留扫描
# ===================================================================
def section_cdrive():
    log("## 8. C 盘残留扫描")
    log()
    targets = [
        (r"C:\Users\61908\AppData\Local\pip\cache", "pip 缓存"),
        (r"C:\Users\61908\AppData\Roaming\Ultralytics", "Ultralytics settings"),
        (r"C:\Users\61908\AppData\Local\torch", "torch hub 缓存"),
        (r"C:\Users\61908\.cache\torch", "torch hub 缓存"),
        (r"C:\Users\61908\.cache\huggingface", "HuggingFace 缓存"),
        (r"C:\Users\61908\.triton", "triton 编译缓存"),
        (r"C:\Users\61908\AppData\Local\Temp\torch", "torch 临时"),
        (r"C:\Users\61908\.roboflow", "RF-DETR 权重缓存"),
        (r"C:\Users\61908\AppData\Local\Temp\torchinductor_61908", "inductor 缓存"),
    ]
    log("| 路径 | 说明 | 状态 |")
    log("|---|---|---|")
    for p, desc in targets:
        pp = Path(p)
        if pp.exists():
            files = list(pp.rglob("*"))
            n = sum(1 for f in files if f.is_file())
            mb = sum(f.stat().st_size for f in files if f.is_file()) / 1024**2
            log(f"| `{p}` | {desc} | ⚠️ 存在, {n} 文件 / {mb:.1f} MB |")
        else:
            log(f"| `{p}` | {desc} | ✅ 不存在 |")
    log()


# ===================================================================
def main():
    import platform
    log("# tda 环境验证报告")
    log()
    log(f"- 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"- 解释器: `{sys.executable}`")
    log(f"- Python: `{sys.version.splitlines()[0]}`")
    log(f"- 主机: {platform.platform()}")
    log(f"- 测试图: `{IMG_900}`")
    log()

    gpu_ok = False
    try:
        gpu_ok = section_torch()
    except Exception as e:
        log(f"- ❌ torch 段异常: `{type(e).__name__}: {e}`")
        log("```"); log(traceback.format_exc()[-2000:]); log("```")
    flush_report()

    for nm, fn in [("sam2", lambda: section_sam2(gpu_ok)),
                   ("yolo", lambda: section_yolo(gpu_ok)),
                   ("rfdetr", lambda: section_rfdetr(gpu_ok)),
                   ("sam3", lambda: section_sam3(gpu_ok)),
                   ("dinov3", lambda: section_dinov3(gpu_ok)),
                   ("imports", section_imports),
                   ("cdrive", section_cdrive)]:
        try:
            fn()
        except Exception as e:
            log(f"- ❌ {nm} 段异常: `{type(e).__name__}: {e}`")
            log("```"); log(traceback.format_exc()[-2500:]); log("```")
            log()
        flush_report()

    log("## 9. 关键指标汇总 (JSON)")
    log()
    log("```json")
    log(json.dumps(results, indent=2, ensure_ascii=False))
    log("```")
    flush_report()
    print(f"\n[OK] 报告已写入 {REPORT}")


if __name__ == "__main__":
    main()
