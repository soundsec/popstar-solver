@echo off
chcp 65001 >nul
cd /d "%~dp0"
title PopStar 求解器

rem 依次尝试常见的 Python 启动方式，取第一个可用的
set "PYEXE="

where py >nul 2>nul && (
    py -3 -c "import sys" >nul 2>nul && set "PYEXE=py -3"
)
if not defined PYEXE (
    where python >nul 2>nul && (
        python -c "import sys" >nul 2>nul && set "PYEXE=python"
    )
)
if not defined PYEXE (
    if exist "%USERPROFILE%\.workbuddy\binaries\python\versions\3.13.12\python.exe" (
        set "PYEXE=%USERPROFILE%\.workbuddy\binaries\python\versions\3.13.12\python.exe"
    )
)
if not defined PYEXE (
    if exist "D:\Anaconda3\python.exe" set "PYEXE=D:\Anaconda3\python.exe"
)

if not defined PYEXE (
    echo.
    echo [!] 没有找到可用的 Python。
    echo     请安装 Python 3.9+ 后重试，或手动运行：python start.py
    echo.
    pause
    exit /b 1
)

echo 使用解释器：%PYEXE%
echo.
echo 正在打开游玩页。分析和图片识别都在这个页面里。
echo.

%PYEXE% start.py

if errorlevel 1 (
    echo.
    echo [!] 启动失败，上面的错误信息即为原因。
    pause
)
