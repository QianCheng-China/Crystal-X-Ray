@echo off
title Crystal X-Ray v0.1
cd /d "%~dp0"
python -u xibao_app.py
if errorlevel 1 pause
