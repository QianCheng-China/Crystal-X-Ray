@echo off
title Build Crystal X-Ray
cd /d "%~dp0"
python -c "import struct; assert struct.calcsize('P')==8"
if errorlevel 1 goto badpython
python -m pip show pyinstaller >nul 2>&1
if errorlevel 1 python -m pip install pyinstaller -i https://pypi.tuna.tsinghua.edu.cn/simple
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
python -m PyInstaller --clean --noconfirm crystal-xray.spec
if errorlevel 1 goto failed
echo @echo off> "dist\Crystal X-Ray\compat_mode.bat"
echo start "" "%%~dp0Crystal X-Ray.exe" --sw>> "dist\Crystal X-Ray\compat_mode.bat"
echo.
echo DONE: dist\Crystal X-Ray\Crystal X-Ray.exe
pause
exit /b 0
:badpython
echo ERROR: need 64-bit Python
pause
exit /b 1
:failed
echo ERROR: build failed - read messages above
pause
exit /b 1
