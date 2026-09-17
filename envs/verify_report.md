# tda 环境验证报告

- 生成时间: 2026-09-17 12:17:24
- 解释器: `D:\Anaconda\envs\tda\python.exe`
- Python: `3.11.16 | packaged by Anaconda, Inc. | (main, Aug 27 2026, 14:36:16) [MSC v.1942 64 bit (AMD64)]`
- 主机: Windows-10-10.0.26200-SP0
- 测试图: `C:\Users\61908\AppData\Local\Temp\claude\D--DataSet\3f6dd74d-e1dc-46a6-bbc2-2fe9e89c0835\scratchpad\views\scanner_crop_native.png`

## 1. PyTorch / CUDA

- torch: `2.11.0+cu128`
- torchvision: `0.26.0+cu128`
- torch 编译所用 CUDA: `12.8`
- `torch.cuda.is_available()`: **True**
- 编译支持的架构: `['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']`
- 设备名: **NVIDIA GeForce RTX 5090**
- `get_device_capability()`: **(12, 0)** → **sm_120**
- 显存总量: 31.82 GiB

### 4096x4096 矩阵乘

- fp32: **2.03 ms** (~67.7 TFLOP/s)
- fp16: **0.71 ms** (~194.1 TFLOP/s)
- 正确性: 与 CPU 参考最大绝对误差 `2.747e-04`, 有限值=True
- ✅ **无 sm_120 / 架构不兼容告警或错误**
- matmul 峰值显存: allocated 408 MiB / reserved 422 MiB

## 2. SAM 2.1 (hiera large)

- 权重: `D:\DataSet\models\weights\sam2.1_hiera_large.pt` (856 MiB)
- config: `configs/sam2.1/sam2.1_hiera_l.yaml`
- `SAM2_BUILD_CUDA=0` → 跳过自定义 CUDA 扩展, 连通域后处理走 CPU
- 模型加载: **8572 ms**

### 900x900 图 `scanner_crop_native.png` (shape=(900, 900, 3))

- `set_image` (900x900, 预热后 3 次均值): **33.2 ms**
- `predict` 点提示 `[785, 215]` (5 次均值): **7.6 ms** — 3 候选 mask, 预测 IoU = [0.968999981880188, 0.00800000037997961, 0.10400000214576721], 最佳 mask 面积 5390 px
- `predict` 框提示 `[660, 95, 899, 335]` (5 次均值): **17.3 ms** — 预测 IoU = 0.977, 面积 41796 px
- 叠加图: `sam2_point_overlay.png`, `sam2_box_overlay.png` (均在 `D:\DataSet\envs`)
- `set_image` (1600x1600, 由 900x900 LANCZOS 上采样得到 `_bench_1600.png`): **30.8 ms**
  - 说明: SAM2 内部一律 resize 到 1024x1024, 因此 900 与 1600 输入的 GPU 图像编码耗时几乎相同, 差值主要来自 CPU 端 resize/归一化。
- **SAM 2.1 large 峰值显存: allocated 1292 MiB (1.26 GiB) / reserved 1534 MiB**

## 3. Ultralytics YOLO

- ultralytics: `8.4.155`
- settings 文件: `D:\DataSet\.cache\ultralytics_cfg\Ultralytics\settings.json`
- datasets_dir=`D:\DataSet\models\ultralytics\datasets`  weights_dir=`D:\DataSet\models\ultralytics\weights`  runs_dir=`D:\DataSet\models\ultralytics\runs`
- ✅ 加载模型: **yolo26n-seg.pt**
- `predict` (`scanner_crop_native.png`, imgsz=640, 5 次均值): **26.0 ms** → 检出 0 个实例
  - 内部分解 (ms): `{'preprocess': 1.83, 'inference': 7.51, 'postprocess': 0.57}`
- YOLO 推理峰值显存: allocated **70 MiB**

### 训练冒烟测试 (coco8-seg, 1 epoch, imgsz=640)

- ✅ 训练完成, 总耗时 **11.6 s**
- 训练时 `trainer.device` = `cuda:0` → **GPU 训练: 是**  (训练结束后 `model.device` 会被搬回 `cpu`, 属正常行为)
- 1 epoch 后 mask mAP50-95 = `0.5517`, mAP50 = `0.8266` (仅冒烟, 数值无意义)
- 产物目录: `D:\DataSet\models\ultralytics\runs\segment\tda_smoke`
- **YOLO 训练峰值显存: allocated 940 MiB (0.92 GiB) / reserved 1108 MiB**

## 4. RF-DETR (分割)

- rfdetr: `1.10.1`
- ✅ 加载 **RFDETRSegPreview** 预训练权重, 耗时 **773 ms**
- `predict` (`scanner_crop_native.png`, 5 次均值): **22.2 ms** → 1 个实例
- 叠加图: `rfdetr_overlay.png`  (mask 输出: 有)
- **RF-DETR 峰值显存: allocated 275 MiB (0.27 GiB)**

## 5. SAM 3

- ✅ `sam3` 代码包导入成功 (从 GitHub 源码安装)
- HuggingFace `facebook/sam3` 状态: gated=**manual**, private=False
- ⚠️ **SAM 3 权重不可用** → `GatedRepoError: 401 Client Error. (Request ID: Root=1-6aac20cb-69755c3937eeb3d33f366913;31310b6b-37b0-41a0-ac8c-84ecc536b871)  Cannot access gated repo for url https://huggingface.co/facebook/sam3/resolve/main/config.json. Access to model facebook/sam3 is restricted. You must have access to it and be authenticated to access it. Please log in.`

  **需要用户手动操作 (不能绕过许可):**
  1. 用浏览器登录 HuggingFace, 打开 <https://huggingface.co/facebook/sam3>,
     点击页面上的 *Agree and access repository* 接受 Meta 的许可 (该仓库 `gated=manual`, 需人工审批)。
  2. 在 <https://huggingface.co/settings/tokens> 生成一个 read token。
  3. 在本机执行 (token 由用户自行输入, 不要写进脚本):
     ```
     $Env:HF_HOME="D:\DataSet\models\hf"
     D:\Anaconda\envs\tda\Scripts\hf.exe auth login
     ```
  4. 之后 `build_sam3_image_model(load_from_HF=True)` 会把权重下载到 `D:\DataSet\models\hf\hub` (HF_HOME 已指向 D 盘)。
  备选: 若已从别处拿到 `sam3.pt`, 放到 `D:\DataSet\models\weights\` 下, 本脚本会自动用 `checkpoint_path=` 离线加载。

## 6. DINOv3

- ✅ `dinov3` 代码包导入成功 (从 GitHub 源码安装)
- HuggingFace `facebook/dinov3-vitl16-pretrain-lvd1689m`: gated=**manual**, private=False
- HuggingFace `facebook/dinov3-vitb16-pretrain-lvd1689m`: gated=**manual**, private=False
- HuggingFace `facebook/dinov3-vits16-pretrain-lvd1689m`: gated=**manual**, private=False
- ⚠️ **DINOv3 权重不可用** → `OSError: You are trying to access a gated repo. Make sure to have access to it at https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m. 401 Client Error. (Request ID: Root=1-6aac20cb-2c18b1292bcf6bc123963fbf;be151fcb-87c6-4e97-a0cd-dbc41cc96364)  Cannot access gated repo for url https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m/reso`

  **需要用户手动操作 (不能绕过许可):** 与 SAM 3 相同 —— 登录 HuggingFace 后在 <https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m> 接受 DINOv3 License (`gated=manual`), 再 `hf auth login` 写入 token。
  备选: 若已有 Meta 官方直链的 `.pth`, 可用 `torch.hub.load('facebookresearch/dinov3','dinov3_vitl16', weights='D:/DataSet/models/weights/xxx.pth')` 离线加载。

## 7. 依赖导入自检

| 包 | 状态 | 版本 / 自检 |
|---|---|---|
| PySide6 | ✅ | `6.11.2 (Qt 6.11.2)` |
| numpy | ✅ | `2.4.6` |
| opencv-python | ✅ | `5.0.0` |
| pillow | ✅ | `12.3.0` |
| scipy | ✅ | `1.17.1` |
| scikit-image | ✅ | `0.26.0` |
| pandas | ✅ | `2.3.3` |
| openpyxl | ✅ | `3.1.5` |
| pyyaml | ✅ | `6.0.3` |
| pytest | ✅ | `9.1.1` |
| tqdm | ✅ | `4.70.1` |
| pycocotools | ✅ | `RLE 编解码 OK (area=20)` |
| scikit-learn | ✅ | `1.7.2` |
| transformers | ✅ | `5.17.0` |
| timm | ✅ | `1.0.29` |
| triton | ✅ | `3.8.0` |
| hydra-core | ✅ | `1.3.7` |
| matplotlib | ✅ | `3.11.2` |
| supervision | ✅ | `0.30.4` |

- PySide6 离屏 `QApplication` + `QWidget` 创建: ✅ (120x40)

## 8. C 盘残留扫描

| 路径 | 说明 | 状态 |
|---|---|---|
| `C:\Users\61908\AppData\Local\pip\cache` | pip 缓存 | ✅ 不存在 |
| `C:\Users\61908\AppData\Roaming\Ultralytics` | Ultralytics settings | ✅ 不存在 |
| `C:\Users\61908\AppData\Local\torch` | torch hub 缓存 | ✅ 不存在 |
| `C:\Users\61908\.cache\torch` | torch hub 缓存 | ✅ 不存在 |
| `C:\Users\61908\.cache\huggingface` | HuggingFace 缓存 | ✅ 不存在 |
| `C:\Users\61908\.triton` | triton 编译缓存 | ✅ 不存在 |
| `C:\Users\61908\AppData\Local\Temp\torch` | torch 临时 | ✅ 不存在 |
| `C:\Users\61908\.roboflow` | RF-DETR 权重缓存 | ⚠️ 存在, 0 文件 / 0.0 MB |
| `C:\Users\61908\AppData\Local\Temp\torchinductor_61908` | inductor 缓存 | ⚠️ 存在, 0 文件 / 0.0 MB |

## 9. 关键指标汇总 (JSON)

```json
{
  "torch_version": "2.11.0+cu128",
  "torchvision_version": "0.26.0+cu128",
  "cuda_available": true,
  "device_name": "NVIDIA GeForce RTX 5090",
  "capability": "sm_120",
  "matmul_fp32_ms": 2.03,
  "matmul_fp16_ms": 0.71,
  "matmul_arch_warnings": [],
  "sam2_load_ms": 8572,
  "sam2_set_image_900_ms": 33.2,
  "sam2_predict_point_ms": 7.6,
  "sam2_predict_box_ms": 17.3,
  "sam2_set_image_1600_ms": 30.8,
  "sam2_peak_vram_mib": 1292,
  "yolo_model": "yolo26n-seg.pt",
  "yolo_predict_ms": 26.0,
  "yolo_predict_speed": {
    "preprocess": 1.83,
    "inference": 7.51,
    "postprocess": 0.57
  },
  "yolo_infer_peak_vram_mib": 70,
  "yolo_train_ok": true,
  "yolo_train_seconds": 11.6,
  "yolo_train_device": "cuda:0",
  "yolo_train_peak_vram_mib": 940,
  "rfdetr_model": "RFDETRSegPreview",
  "rfdetr_predict_ms": 22.2,
  "rfdetr_detections": 1,
  "rfdetr_peak_vram_mib": 275,
  "sam3_weights": "gated / 未获授权",
  "dinov3_weights": "gated / 未获授权"
}
```