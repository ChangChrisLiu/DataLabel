"""把 Ultralytics 的 datasets/weights/runs 目录全部指到 D 盘。
运行前需设置 YOLO_CONFIG_DIR=D:\\DataSet\\.cache\\ultralytics_cfg
（activate_tda.ps1 / activate_tda.sh 已包含）。"""
from ultralytics import settings

settings.update({
    "datasets_dir": r"D:\DataSet\models\ultralytics\datasets",
    "weights_dir": r"D:\DataSet\models\ultralytics\weights",
    "runs_dir": r"D:\DataSet\models\ultralytics\runs",
    "sync": False,
})
for k in ("datasets_dir", "weights_dir", "runs_dir", "sync"):
    print(k, "=", settings[k])
print("settings file:", settings.file)
