@echo off
echo ========================================
echo    🚀 Iniciando Valerio AI - Full Power
echo ========================================

REM --- Backend ---
echo [1/2] Lanzando Backend (FastAPI)...
start cmd /k "cd /d %~dp0demo && uvicorn app.main:app --reload"

REM --- Frontend ---
echo [2/2] Lanzando Frontend (Vite)...
start cmd /k "cd /d %~dp0valerio-ui && npm run dev"

echo ========================================
echo ✅ Valerio AI corriendo en:
echo   Backend -> http://127.0.0.1:8000/docs
echo   Frontend -> http://127.0.0.1:5173/
echo ========================================
