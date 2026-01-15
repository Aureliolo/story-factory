@echo off
title Story Factory
cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -File "%~dp0scripts\start.ps1"
