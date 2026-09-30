@echo off
rem Teardown Annotator launcher. Double-click, or run from a terminal with extra
rem arguments, e.g.:  start_tda.bat --desktop 13 --view scan
rem Without arguments it reopens the machine / view / step you were on last time.
cd /d D:\DataSet
rem The NVIDIA driver's PTX JIT cache: on D:, and 4 GB instead of the 1 GB
rem default. Its default home is %APPDATA%\NVIDIA\ComputeCache on C:, and at
rem 1 GB it evicts, so the first SAM or detector kernels of a session were
rem recompiled for 8-15 s while every other CUDA call waited (U4 review).
set CUDA_CACHE_PATH=D:\DataSet\.cache\nv_compute
set CUDA_CACHE_MAXSIZE=4294967296
rem ultralytics (the fallback detector backend) keeps its settings on D: and
rem never downloads; rfdetr and torch caches stay on D: too.
set YOLO_CONFIG_DIR=D:\DataSet\.cache\ultralytics_cfg
set YOLO_OFFLINE=1
set HF_HUB_OFFLINE=1
set TORCH_HOME=D:\DataSet\.cache\torch
set HF_HOME=D:\DataSet\.cache\hf
D:\Anaconda\envs\tda\python.exe -m tda.cli app --annotator chang %*
if errorlevel 1 (
    echo.
    echo The annotator exited with an error ^(code %errorlevel%^). Code 3 = the database is locked by another running annotator.
    echo Log: D:\DataSet\.cache\logs\tda_app.log
    pause
)
