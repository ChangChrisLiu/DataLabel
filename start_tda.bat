@echo off
rem Teardown Annotator launcher. Double-click, or run from a terminal with extra
rem arguments, e.g.:  start_tda.bat --desktop 13 --view scan
rem Without arguments it reopens the machine / view / step you were on last time.
cd /d D:\DataSet
D:\Anaconda\envs\tda\python.exe -m tda.cli app --annotator chang %*
if errorlevel 1 (
    echo.
    echo The annotator exited with an error ^(code %errorlevel%^). Code 3 = the database is locked by another running annotator.
    echo Log: D:\DataSet\.cache\logs\tda_app.log
    pause
)
