@echo off
rem Лёгкий запуск меню на Windows (двойной клик или из консоли).
chcp 65001 >nul
cd /d "%~dp0"
python -m src.menu
pause
