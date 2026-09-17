# ==============================================================
#  tda 环境激活脚本 (PowerShell)
#  用法:  . D:\DataSet\envs\activate_tda.ps1
#  作用:  把所有缓存 / 权重 / 数据集 / 临时文件都固定在 D 盘
# ==============================================================

$Env:TDA_ROOT = "D:\DataSet"
$Env:TDA_PYTHON = "D:\Anaconda\envs\tda\python.exe"

# ---- 缓存与临时文件 (全部 D 盘) ----
$Env:PIP_CACHE_DIR = "D:\DataSet\.cache\pip"
$Env:TMP = "D:\DataSet\.cache\tmp"
$Env:TEMP = "D:\DataSet\.cache\tmp"

# ---- 模型权重缓存 ----
$Env:TORCH_HOME = "D:\DataSet\models\torch_home"
$Env:HF_HOME = "D:\DataSet\models\hf"
$Env:HUGGINGFACE_HUB_CACHE = "D:\DataSet\models\hf\hub"
$Env:TRANSFORMERS_CACHE = "D:\DataSet\models\hf\transformers"
$Env:XDG_CACHE_HOME = "D:\DataSet\.cache"

# ---- Ultralytics ----
# settings.json 本身的存放位置 (否则默认落在 C:\Users\<user>\AppData\Roaming\Ultralytics)
$Env:YOLO_CONFIG_DIR = "D:\DataSet\.cache\ultralytics_cfg"

# ---- RF-DETR 权重缓存 (否则默认落在 C:\Users\<user>\.roboflow\models) ----
$Env:RF_HOME = "D:\DataSet\models\rfdetr"
$Env:ROBOFLOW_HOME = "D:\DataSet\models\rfdetr"

# ---- SAM 2: 不编译自定义 CUDA 扩展, 后处理走 CPU ----
$Env:SAM2_BUILD_CUDA = "0"
$Env:SAM2_BUILD_ALLOW_ERRORS = "1"

# ---- 目录自建 ----
foreach ($d in @($Env:PIP_CACHE_DIR, $Env:TMP, $Env:TORCH_HOME, $Env:HF_HOME,
                 $Env:HUGGINGFACE_HUB_CACHE, $Env:YOLO_CONFIG_DIR, $Env:RF_HOME,
                 "D:\DataSet\models\ultralytics\datasets",
                 "D:\DataSet\models\ultralytics\weights",
                 "D:\DataSet\models\ultralytics\runs",
                 "D:\DataSet\models\weights")) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Force -Path $d | Out-Null }
}

# ---- 把 tda 放到 PATH 前面 (不调用 conda activate, 避免影响其它环境) ----
$Env:PATH = "D:\Anaconda\envs\tda;D:\Anaconda\envs\tda\Scripts;D:\Anaconda\envs\tda\Library\bin;" + $Env:PATH

Write-Host "[tda] python      = $Env:TDA_PYTHON"
Write-Host "[tda] PIP_CACHE_DIR = $Env:PIP_CACHE_DIR"
Write-Host "[tda] TORCH_HOME    = $Env:TORCH_HOME"
Write-Host "[tda] HF_HOME       = $Env:HF_HOME"
Write-Host "[tda] YOLO_CONFIG_DIR = $Env:YOLO_CONFIG_DIR"
Write-Host "[tda] RF_HOME        = $Env:RF_HOME"
Write-Host "[tda] TMP/TEMP      = $Env:TMP"
Write-Host "[tda] Ultralytics datasets/weights/runs -> D:\DataSet\models\ultralytics\*"
Write-Host "[tda] ready."
