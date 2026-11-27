@echo off
cd /d "%~dp0"
echo ===================================================
echo Starting Indian Stock Market Predictor Dashboard...
echo ===================================================
call .venv\Scripts\activate.bat
streamlit run app.py --server.port 8501 --server.address 127.0.0.1
pause
