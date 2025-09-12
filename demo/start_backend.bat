@echo off
echo ==============================
echo  🚀 Iniciando Valerio Backend...
echo ==============================
call .venv\Scripts\activate
cd demo
uvicorn app.main:app --reload


