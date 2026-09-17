#!/usr/bin/env bash
# ==============================================================
#  tda 环境激活脚本 (Git Bash)
#  用法:  source /d/DataSet/envs/activate_tda.sh
#  作用:  把所有缓存 / 权重 / 数据集 / 临时文件都固定在 D 盘
#
#  注意: 在 Git Bash 里给 Windows 程序传路径时用 Windows 形式(D:\...),
#        但 bash 会吃掉反斜杠 -> 这里统一用正斜杠 'D:/...', Windows 侧同样识别。
# ==============================================================

export TDA_ROOT="D:/DataSet"
export TDA_PYTHON="/d/Anaconda/envs/tda/python.exe"

# ---- 缓存与临时文件 (全部 D 盘) ----
export PIP_CACHE_DIR="D:/DataSet/.cache/pip"
export TMP="D:/DataSet/.cache/tmp"
export TEMP="D:/DataSet/.cache/tmp"

# ---- 模型权重缓存 ----
export TORCH_HOME="D:/DataSet/models/torch_home"
export HF_HOME="D:/DataSet/models/hf"
export HUGGINGFACE_HUB_CACHE="D:/DataSet/models/hf/hub"
export TRANSFORMERS_CACHE="D:/DataSet/models/hf/transformers"
export XDG_CACHE_HOME="D:/DataSet/.cache"

# ---- Ultralytics ----
export YOLO_CONFIG_DIR="D:/DataSet/.cache/ultralytics_cfg"

# ---- RF-DETR 权重缓存 (否则默认落在 C:\Users\<user>\.roboflow\models) ----
export RF_HOME="D:/DataSet/models/rfdetr"
export ROBOFLOW_HOME="D:/DataSet/models/rfdetr"

# ---- SAM 2: 不编译自定义 CUDA 扩展 ----
export SAM2_BUILD_CUDA=0
export SAM2_BUILD_ALLOW_ERRORS=1

mkdir -p /d/DataSet/.cache/pip /d/DataSet/.cache/tmp \
         /d/DataSet/.cache/ultralytics_cfg \
         /d/DataSet/models/torch_home /d/DataSet/models/hf/hub \
         /d/DataSet/models/ultralytics/datasets \
         /d/DataSet/models/ultralytics/weights \
         /d/DataSet/models/ultralytics/runs \
         /d/DataSet/models/rfdetr \
         /d/DataSet/models/weights

export PATH="/d/Anaconda/envs/tda:/d/Anaconda/envs/tda/Scripts:/d/Anaconda/envs/tda/Library/bin:$PATH"

echo "[tda] python         = $TDA_PYTHON"
echo "[tda] PIP_CACHE_DIR  = $PIP_CACHE_DIR"
echo "[tda] TORCH_HOME     = $TORCH_HOME"
echo "[tda] HF_HOME        = $HF_HOME"
echo "[tda] YOLO_CONFIG_DIR= $YOLO_CONFIG_DIR"
echo "[tda] RF_HOME       = $RF_HOME"
echo "[tda] TMP/TEMP       = $TMP"
echo "[tda] ready."
