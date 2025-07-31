@echo off
echo ========================================
echo 하이브리드 추출 시스템 실행
echo ========================================

REM 현재 디렉토리 설정
cd /d "%~dp0.."

REM 1. PyMuPDF 표 영역 감지 (venv_text_new)
echo.
echo [1/4] PyMuPDF 표 영역 감지 시작...
call environments\venv_text_new\Scripts\activate.bat
python src\pymupdf_table_detector.py "input.pdf"
call deactivate

REM 2. Camelot 표 내용 추출 (venv_table_new)
echo.
echo [2/4] Camelot 표 내용 추출 시작...
call environments\venv_table_new\Scripts\activate.bat
python src\camelot_table_extractor.py "input.pdf"
call deactivate

REM 3. 표 영역 제외 텍스트 추출 (venv_text_new)
echo.
echo [3/4] 표 영역 제외 텍스트 추출 시작...
call environments\venv_text_new\Scripts\activate.bat
python src\text_extractor_excluding_tables.py "input.pdf"
call deactivate

REM 4. 하이브리드 통합 및 LLM 처리 (venv_rag_new)
echo.
echo [4/4] 하이브리드 통합 및 LLM 처리 시작...
call environments\venv_rag_new\Scripts\activate.bat
python src\hybrid_extraction_system.py "input.pdf"
call deactivate

echo.
echo ========================================
echo 하이브리드 시스템 실행 완료!
echo ========================================
pause 