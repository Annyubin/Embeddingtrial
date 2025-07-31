#!/bin/bash
echo "========================================"
echo "하이브리드 추출 시스템 실행 (Linux 전용 환경)"
echo "========================================"

# 현재 디렉토리 설정
cd "$(dirname "$0")/.."

# 1. PyMuPDF 표 영역 감지 (venv_text_linux)
echo ""
echo "[1/4] PyMuPDF 표 영역 감지 시작..."
source environments/venv_text_linux/bin/activate
python src/pymupdf_table_detector.py "input.pdf"
deactivate

# 2. Camelot 표 내용 추출 (venv_table_linux)
echo ""
echo "[2/4] Camelot 표 내용 추출 시작..."
source environments/venv_table_linux/bin/activate
python src/camelot_table_extractor.py "input.pdf"
deactivate

# 3. 표 영역 제외 텍스트 추출 (venv_text_linux)
echo ""
echo "[3/4] 표 영역 제외 텍스트 추출 시작..."
source environments/venv_text_linux/bin/activate
python src/text_extractor_excluding_tables.py "input.pdf"
deactivate

# 4. 하이브리드 통합 및 LLM 처리 (venv_rag_linux)
echo ""
echo "[4/4] 하이브리드 통합 및 LLM 처리 시작..."
source environments/venv_rag_linux/bin/activate
python src/hybrid_extraction_system.py "input.pdf"
deactivate

echo ""
echo "========================================"
echo "하이브리드 시스템 실행 완료!"
echo "========================================" 