# Linux 환경에서 PDF 임베딩 프로젝트 실행 가이드

## 개요
이 프로젝트는 PDF 문서에서 표와 텍스트를 추출하고, LLM을 사용하여 섹션 제목과 키워드를 자동 생성하는 하이브리드 시스템입니다.

## 🚀 전체 실행 과정 (단계별 가이드)

### 1단계: 시스템 준비

#### 1.1 시스템 패키지 설치
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3 python3-pip python3-venv ghostscript libgl1-mesa-glx libglib2.0-0

# CentOS/RHEL
sudo yum install python3 python3-pip ghostscript mesa-libGL
```

#### 1.2 Ollama 설치 및 설정
```bash
# Ollama 설치
curl -fsSL https://ollama.ai/install.sh | sh

# Ollama 서비스 시작 (백그라운드)
ollama serve &

# Mistral 모델 다운로드 (약 4.4GB)
ollama pull mistral

# 설치 확인
ollama list
```

### 2단계: 가상환경 설정

#### 2.1 가상환경 생성
```bash
# 프로젝트 디렉토리로 이동
cd /home/beam23/25075_임베딩

# 가상환경들이 이미 생성되어 있음
ls -la environments/
```

#### 2.2 가상환경별 용도
- `venv_text_new`: PDF 텍스트 추출, 청킹, 필터링 (PyMuPDF 기반)
- `venv_table_new`: PDF 테이블 추출, LLM 컬럼 추론 (Camelot 기반)
- `venv_rag_new`: LLM 섹션/키워드 추출, 임베딩 생성 (Ollama + Mistral)
- `venv_web_new`: 웹 인터페이스, 파일 업로드/다운로드 (Streamlit 기반)

### 3단계: PDF 파일 준비
```bash
# input.pdf 파일을 프로젝트 루트에 배치
cp your_pdf_file.pdf input.pdf
```

### 4단계: 시스템 실행

#### 4.1 Ollama 서비스 확인
```bash
# Ollama 서비스가 실행 중인지 확인
ps aux | grep ollama

# 만약 실행되지 않았다면
ollama serve &
```

#### 4.2 단계별 실행 (각 단계마다 다른 가상환경 필요!)

**1) PyMuPDF 표 영역 감지**
```bash
# venv_text_new 활성화 (PyMuPDF, nltk 포함)
source environments/venv_text_new/bin/activate

# 표 영역 감지 실행
python src/pymupdf_table_detection.py
```

**2) Camelot 표 내용 추출**
```bash
# venv_table_new 활성화 (camelot-py, ollama 포함)
source environments/venv_table_new/bin/activate

# 표 내용 추출 실행
python src/camelot_table_extraction.py
```

**3) 텍스트 청크 생성**
```bash
# venv_text_new 활성화 (PyMuPDF, nltk 포함)
source environments/venv_text_new/bin/activate

# 텍스트 청크 생성
python src/text_chunking.py
```

**4) LLM 처리 및 통합**
```bash
# venv_rag_new 활성화 (faiss-cpu, sentence-transformers 포함)
source environments/venv_rag_new/bin/activate

# LLM 처리 및 통합 실행
python src/llm_processing.py
```

**5) LLM 컬럼 추론 및 개선**
```bash
# venv_rag_new 활성화 (ollama 포함)
source environments/venv_rag_new/bin/activate

# LLM 컬럼 추론 및 개선 실행
python src/llm_column_inference.py data/final_llm_enhanced_data.json
```
- **목적**: 표 구조를 LLM이 검토하고 개선하여 더 정확한 컬럼명 생성
- **기능**: 
  - **빈 컬럼 추론**: `columns`가 `[]`인 경우 LLM이 내용 분석하여 컬럼명 생성
  - **기존 컬럼 검토**: 이미 채워진 컬럼명도 LLM이 검토하여 개선
  - **헤더 분리**: 헤더가 `rows`에 들어간 경우 자동으로 분리하여 `columns`로 이동
  - **패턴 분석**: 숫자, 날짜, 금액, 이름 등 데이터 유형 자동 감지
  - **메타데이터 추적**: 어떤 변경사항이 적용되었는지 추적

### 6단계: 웹 인터페이스 실행
```bash
# venv_web_new 활성화 (streamlit, plotly 포함)
source environments/venv_web_new/bin/activate

# 간소화된 Streamlit 앱 실행
streamlit run src/streamlit_app_simple.py
```

**웹 브라우저에서 접속:**
- Local URL: http://localhost:8501
- Network URL: http://172.20.0.2:8501
- External URL: http://61.84.16.185:8501

**간소화된 Streamlit 앱 기능:**
- **전체 실행**: 한 번의 클릭으로 전체 시스템 실행
- **결과 다운로드**: 모든 결과 파일을 쉽게 다운로드
- **개별 실행**: 각 단계별 개별 실행 가능
- **시스템 상태**: Ollama 및 가상환경 상태 실시간 확인

## 생성되는 결과 파일

실행 완료 후 다음 파일들이 생성됩니다:

- `pymupdf_table_areas.json`: PyMuPDF로 감지된 표 영역 정보
- `camelot_table_content.json`: Camelot으로 추출된 표 내용
- `text_chunks_no_tables.json`: 표를 제외한 텍스트 청크
- `data/final_llm_enhanced_data.json`: LLM 처리된 최종 통합 결과

## 문제 해결

### Ollama 관련 문제
```bash
# Ollama 서비스 재시작
pkill ollama
ollama serve &

# 모델 재다운로드
ollama rm mistral
ollama pull mistral

# 연결 테스트
python -c "import ollama; print(ollama.chat(model='mistral', messages=[{'role': 'user', 'content': 'test'}]))"
```

### 가상환경 관련 문제
```bash
# 가상환경 재생성
rm -rf environments/venv_*
cd environments
python3 -m venv venv_rag_new
python3 -m venv venv_table_new
python3 -m venv venv_text_new
python3 -m venv venv_web_new
```

### 권한 문제
```bash
# 실행 권한 부여
chmod +x scripts/*.sh
chmod +x src/*.py
```

## 빠른 실행 스크립트

### 전체 시스템 한 번에 실행
```bash
#!/bin/bash
# run_complete_system.sh

echo "1. Ollama 서비스 시작..."
ollama serve &
sleep 5

echo "2. PyMuPDF 표 영역 감지..."
source environments/venv_text_new/bin/activate
python src/pymupdf_table_detection.py

echo "3. Camelot 표 내용 추출..."
source environments/venv_table_new/bin/activate
python src/camelot_table_extraction.py

echo "4. 텍스트 청크 생성..."
source environments/venv_text_new/bin/activate
python src/text_chunking.py

echo "5. LLM 처리 및 통합..."
source environments/venv_rag_new/bin/activate
python src/llm_processing.py

echo "6. LLM 컬럼 추론 및 개선..."
source environments/venv_rag_new/bin/activate
python src/llm_column_inference.py data/final_llm_enhanced_data.json

echo "완료! 결과 파일을 확인하세요."
```

## 시스템 요구사항

- **RAM**: 최소 8GB (LLM 처리용)
- **디스크**: 최소 10GB (모델 + 결과 파일)
- **GPU**: CUDA 지원 GPU 권장 (선택사항)
- **Python**: 3.8 이상

## 🔍 모니터링

### 시스템 상태 확인
```bash
# Ollama 서비스 상태
ps aux | grep ollama

# 가상환경 상태
ls -la environments/

# 결과 파일 확인
ls -la *.json
ls -la data/
```

### 로그 확인
```bash
# Ollama 로그
tail -f ~/.ollama/logs/ollama.log

# Python 스크립트 로그
python src/your_script.py 2>&1 | tee log.txt
```

## 🚨 주의사항

1. **메모리 관리**: LLM 처리 시 충분한 RAM 확보
2. **네트워크**: Ollama 모델 다운로드 시 안정적인 인터넷 연결
3. **파일 경로**: 한글 경로 사용 시 인코딩 문제 주의
4. **권한**: 시스템 패키지 설치 시 sudo 권한 필요
5. **가상환경**: 각 단계마다 올바른 가상환경 활성화 필수

## 📞 지원

문제가 발생하면 다음을 확인하세요:
1. 시스템 요구사항 충족 여부
2. Ollama 서비스 실행 상태
3. 가상환경 활성화 상태
4. 파일 권한 설정 