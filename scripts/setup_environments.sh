#!/bin/bash

# 가상환경 설정 스크립트 (Linux)
# 각 가상환경별로 필요한 패키지들을 설치

set -e

echo "가상환경 설정을 시작합니다..."

# 1. venv_text_new 설정 (PDF 텍스트 추출, 청킹, 필터링)
echo ""
echo "1. venv_text_new 설정 중..."
source environments/venv_text_new/bin/activate
pip install --upgrade pip
pip install -r requirements/requirements_text_new.txt
deactivate
echo "venv_text_new setup complete!"

# 2. venv_table_new 설정 (PDF 테이블 추출, LLM 컬럼 추론)
echo ""
echo "2. venv_table_new 설정 중..."
source environments/venv_table_new/bin/activate
pip install --upgrade pip
pip install -r requirements/requirements_table_new.txt
deactivate
echo "venv_table_new setup complete!"

# 3. venv_rag_new 설정 (LLM 섹션/키워드 추출, 임베딩 생성)
echo ""
echo "3. venv_rag_new 설정 중..."
source environments/venv_rag_new/bin/activate
pip install --upgrade pip
pip install -r requirements/requirements_rag_new.txt
deactivate
echo "venv_rag_new setup complete!"

# 4. venv_web_new 설정 (웹 인터페이스)
echo ""
echo "4. venv_web_new 설정 중..."
source environments/venv_web_new/bin/activate
pip install --upgrade pip
pip install -r requirements/requirements_web_new.txt
deactivate
echo "venv_web_new setup complete!"

echo ""
echo "모든 가상환경 설정이 완료되었습니다!"
echo ""
echo "가상환경별 역할:"
echo "- venv_text_new: PDF 텍스트 추출, 청킹, 필터링"
echo "- venv_table_new: PDF 테이블 추출, LLM 컬럼 추론"
echo "- venv_rag_new: LLM 섹션/키워드 추출, 임베딩 생성"
echo "- venv_web_new: 웹 인터페이스, 파일 업로드/다운로드"
echo ""
echo "시스템 실행:"
echo "1. Ollama 서비스 시작: ollama serve &"
echo "2. Mistral 모델 다운로드: ollama pull mistral"
echo "3. 전체 시스템 실행: bash scripts/run_complete_system.sh"
echo "4. 웹 인터페이스: source environments/venv_web_new/bin/activate && streamlit run src/streamlit_app_simple.py" 