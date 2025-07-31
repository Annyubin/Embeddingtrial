@echo off
echo ======================================
echo PDF Embedding Project Setup
echo ======================================

echo.
echo [1/3] Creating venv_text...
python -m venv venv_text
call venv_text\Scripts\activate
pip install -r requirements_text.txt
call deactivate
echo ✅ venv_text setup complete!

echo.
echo [2/3] Creating venv_table...  
python -m venv venv_table
call venv_table\Scripts\activate
pip install -r requirements_table.txt
call deactivate
echo ✅ venv_table setup complete!

echo.
echo [3/3] Creating venv_web...
python -m venv venv_web
call venv_web\Scripts\activate
pip install -r requirements_web.txt
call deactivate
echo ✅ venv_web setup complete!

echo.
echo ======================================
echo 🎉 All environments ready!
echo ======================================
echo.
echo To run the web app:
echo   venv_web\Scripts\activate
echo   streamlit run streamlit_app.py
echo.
pause 