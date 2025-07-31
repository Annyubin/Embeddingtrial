#!/bin/bash

# 전체 시스템 실행 스크립트
# PDF 임베딩 프로젝트 - Linux 환경
# 실제 사용 가상환경: venv_text_new, venv_table_new, venv_rag_new, venv_web_new

set -e  # 오류 발생 시 스크립트 중단

echo "PDF 임베딩 시스템 시작..."
echo "=================================="

# 1. Ollama 서비스 시작
echo "1. Ollama 서비스 시작..."
if ! pgrep -x "ollama" > /dev/null; then
    echo "   Ollama 서비스를 시작합니다..."
    ollama serve &
    sleep 5
else
    echo "   Ollama 서비스가 이미 실행 중입니다."
fi

# 2. Mistral 모델 확인
echo "2. Mistral 모델 확인..."
if ! ollama list | grep -q "mistral"; then
    echo "   Mistral 모델을 다운로드합니다..."
    ollama pull mistral
else
    echo "   Mistral 모델이 이미 설치되어 있습니다."
fi

# 3. PyMuPDF 표 영역 감지 (venv_text_new 사용)
echo "3. PyMuPDF 표 영역 감지..."
echo "   가상환경 활성화: venv_text_new (PyMuPDF, nltk 포함)"
source environments/venv_text_new/bin/activate
python src/pymupdf_table_detection.py
echo "   표 영역 감지 완료"

# 4. Camelot 표 내용 추출 (venv_table_new 사용)
echo "4. Camelot 표 내용 추출..."
echo "   가상환경 활성화: venv_table_new (camelot-py, ollama 포함)"
source environments/venv_table_new/bin/activate
python src/camelot_table_extraction.py
echo "   표 내용 추출 완료"

# 5. 텍스트 청크 생성 (venv_text_new 사용)
echo "5. 텍스트 청크 생성..."
echo "   가상환경 활성화: venv_text_new (PyMuPDF, nltk 포함)"
source environments/venv_text_new/bin/activate
python src/text_chunking.py
echo "   텍스트 청크 생성 완료"

# 6. LLM 처리 및 통합 (venv_rag_new 사용)
echo "6. LLM 처리 및 통합..."
echo "   가상환경 활성화: venv_rag_new (faiss-cpu, sentence-transformers 포함)"
source environments/venv_rag_new/bin/activate
python src/llm_processing.py
echo "   LLM 처리 완료"

# 7. LLM 컬럼 추론 및 개선
echo "7. LLM 컬럼 추론 및 개선..."
echo "   가상환경 활성화: venv_rag_new (ollama 포함)"
source environments/venv_rag_new/bin/activate
python src/llm_column_inference.py data/final_llm_enhanced_data.json
echo "   LLM 컬럼 추론 및 개선 완료"

# 8. 웹 인터페이스 실행 (venv_web_new 사용)
echo "8. 웹 인터페이스 실행..."
echo "   가상환경 활성화: venv_web_new (streamlit, plotly 포함)"
source environments/venv_web_new/bin/activate
echo "   간소화된 Streamlit 앱을 시작합니다..."
streamlit run src/streamlit_app_simple.py &
echo "   Streamlit 앱 시작 완료"

echo "=================================="
echo "전체 시스템 실행 완료!"
echo ""
echo "생성된 결과 파일:"
echo "   - pymupdf_table_areas.json"
echo "   - camelot_table_content.json"
echo "   - text_chunks_no_tables.json"
echo "   - data/final_llm_enhanced_data.json"
echo ""
echo "웹 인터페이스 접속:"
echo "   - Local URL: http://localhost:8501"
echo "   - Network URL: http://172.20.0.2:8501"
echo "   - External URL: http://61.84.16.185:8501" 