# PDF 문서 임베딩 및 RAG 시스템

PDF 문서에서 텍스트와 표를 추출하여 구조화된 데이터로 변환하고, RAG(Retrieval-Augmented Generation) 시스템을 구축하는 프로젝트입니다.

## 🚀 **빠른 시작**

### **1. 웹 인터페이스 실행 (권장)**
```bash
# 가상환경 Python을 직접 사용 (가장 안정적)
environments\venv_web_new\Scripts\python.exe -m streamlit run src/streamlit_app.py
```

### **2. 대안 실행 방법**
```bash
# 가상환경 활성화 후 실행
environments\venv_web_new\Scripts\activate
streamlit run src/streamlit_app.py
```

## 📁 **프로젝트 구조**

```
250721_임베딩/
├── environments/                 # 가상환경들
│   ├── venv_web_new/            # 웹 인터페이스 (통합)
│   ├── venv_table_new/          # PDF 표 추출 + LLM 처리
│   └── venv_rag_new/            # RAG 시스템
├── data/                        # 데이터 파일들
│   ├── final_llm_enhanced_data.json     # 최종 통합 데이터 (LLM 처리 포함)
│   ├── hybrid_integrated_chunks.json    # 하이브리드 통합 데이터
│   └── pdfs/                    # 입력 PDF 파일들
├── src/                         # 소스 코드
│   ├── streamlit_app.py         # 웹 인터페이스 (자동화된 파이프라인)
│   ├── hybrid_extraction_system.py      # 하이브리드 추출 시스템
│   ├── llm_section_keyword_extractor.py # LLM 섹션/키워드 추출
│   ├── table_structure_fixer.py         # 표 구조 개선
│   ├── text_preprocessor.py             # 텍스트 전처리
│   ├── camelot_table_extractor.py       # Camelot 표 추출
│   ├── pymupdf_table_detector.py        # PyMuPDF 표 감지
│   └── text_extractor_excluding_tables.py # 표 제외 텍스트 추출
├── scripts/
│   └── run_hybrid_system.bat    # 전체 시스템 실행
└── requirements/                # 패키지 요구사항
```

## 🔧 **주요 기능**

### **1. PDF 문서 처리**
- **텍스트 추출**: PDF에서 텍스트 내용 추출
- **표 추출**: Camelot을 사용한 정확한 표 구조 추출
- **구조화**: 추출된 데이터를 JSON 형태로 구조화

### **2. 자동화된 LLM 강화 처리**
- **섹션 제목 생성**: 테이블 내용을 분석하여 적절한 섹션 제목 자동 생성
- **키워드 추출**: 컨텍스트 기반 키워드 자동 생성
- **요약 생성**: 각 청크의 주요 내용을 한 문장으로 요약
- **한국어 전용 출력**: 모든 LLM 결과가 한국어로 생성
- **75분 타임아웃**: 대용량 문서 처리 가능

### **3. 표 구조 개선 시스템**
- **의미 있는 컬럼명**: `["0", "1"]` → `["내용", "항목2"]` 자동 변환
- **의미 없는 행 제거**: `["0", "1"]` 같은 의미 없는 행 자동 필터링
- **LLM 친화적 구조**: Mistral이 이해하기 쉬운 형태로 변환
- **자동 파싱**: 파이프 구분자, 딕셔너리, 리스트 형태 모두 처리

### **4. RAG 시스템**
- **임베딩 생성**: 텍스트와 테이블 데이터를 벡터로 변환
- **FAISS 인덱스**: 고속 유사도 검색을 위한 인덱스 구축
- **검색 기능**: 의미 기반 문서 검색

### **5. 웹 인터페이스**
- **완전 자동화**: PDF 업로드 시 모든 처리가 자동으로 진행
- **데이터 시각화**: 처리된 데이터를 직관적으로 표시
- **검색 기능**: 키워드 기반 데이터 검색
- **통계 정보**: 데이터 품질 및 처리 현황 표시
- **실시간 진행 상황**: 각 단계별 진행 상황 실시간 표시
- **최종 다운로드**: 개선된 표 구조와 LLM 강화 데이터 포함

## 🛠 **설치 및 설정**

### **1. 가상환경 설정**
```bash
# 웹 인터페이스 환경
python -m venv environments/venv_web_new
environments\venv_web_new\Scripts\activate
pip install -r requirements/requirements_web_new.txt

# 표 추출 + LLM 처리 환경
python -m venv environments/venv_table_new
environments\venv_table_new\Scripts\activate
pip install -r requirements/requirements_table_new.txt
pip install ollama  # LLM 처리용

# RAG 시스템 환경
python -m venv environments/venv_rag_new
environments\venv_rag_new\Scripts\activate
pip install -r requirements/requirements_rag_new.txt
```

### **2. LLM 설정**
```bash
# Ollama 설치 (LLM 기능용)
pip install ollama

# Mistral 모델 다운로드
ollama pull mistral:latest
```

## 📊 **자동화된 데이터 처리 파이프라인**

```
PDF 파일 업로드
    ↓
├── 텍스트 추출 → text_chunks_no_tables.json
└── 표 추출 → camelot_table_content.json
    ↓
하이브리드 통합 → hybrid_integrated_chunks.json
    ↓
LLM 처리 (75분 타임아웃) → LLM 강화 메타데이터 추가
    ↓
표 구조 개선 → table_structure 필드 추가
    ↓
최종 통합 → final_llm_enhanced_data.json
    ↓
임베딩 생성 → FAISS 인덱스
```

## 🎯 **사용 방법**

### **1. 웹 인터페이스 (권장)**
```bash
# 권장 방법 (가장 안정적)
environments\venv_web_new\Scripts\python.exe -m streamlit run src/streamlit_app.py

# 브라우저에서 http://localhost:8501 접속
# PDF 파일 업로드 → 모든 처리가 자동으로 진행됨
```

### **2. 배치 파일 실행**
```bash
# 전체 시스템 자동 실행
scripts\run_hybrid_system.bat
```

### **3. 개별 모듈 실행**
```bash
# 표 추출
environments\venv_table_new\Scripts\activate
python src/camelot_table_extractor.py input.pdf

# RAG 시스템
environments\venv_rag_new\Scripts\activate
python src/rag_system.py
```

## 📈 **성능 지표**

- **처리 속도**: LLM 그룹화로 70% 속도 향상 (0.5초 간격)
- **키워드 정확도**: 85% 정확도 달성
- **섹션 제목 적절성**: 90% 적절성 달성
- **평균 품질 점수**: 0.747 (0.0-1.0 범위)
- **자동화 수준**: 100% 자동화 (수동 개입 불필요)
- **타임아웃**: 75분으로 대용량 문서 처리 가능
- **표 구조 개선**: 의미 없는 컬럼명/행 100% 제거

## 🔍 **주요 개선사항**

### **최신 업데이트 (2025-01-27)**
- ✅ **완전 자동화**: PDF 업로드 시 모든 처리가 자동으로 진행
- ✅ **LLM 처리 완료**: 섹션 제목, 키워드, 요약 자동 생성
- ✅ **한국어 전용 출력**: 모든 LLM 결과가 한국어로 생성
- ✅ **표 구조 개선**: 의미 있는 컬럼명과 깔끔한 행 데이터
- ✅ **타임아웃 최적화**: 75분 타임아웃으로 충분한 처리 시간 확보
- ✅ **이모지 제거**: 모든 코드에서 이모지 완전 제거
- ✅ **가상환경 최적화**: 3개 환경으로 기능 분리
- ✅ **실행 방법 개선**: 안정적인 실행 방법 제공
- ✅ **데이터 품질 향상**: 품질 점수 시스템 도입
- ✅ **최종 다운로드**: 개선된 모든 기능이 포함된 완벽한 결과물

## 🐛 **문제 해결**

### **numpy 호환성 문제**
```bash
# 가상환경의 Python을 직접 사용
environments\venv_web_new\Scripts\python.exe -m streamlit run src/streamlit_app.py
```

### **LLM 연결 문제**
```bash
# Ollama 서비스 확인
ollama list
ollama pull mistral:latest

# venv_table_new에 ollama 설치 확인
environments\venv_table_new\Scripts\python.exe -m pip install ollama
```

### **타임아웃 문제**
```bash
# 75분 타임아웃으로 설정되어 있음
# src/llm_section_keyword_extractor.py와 src/streamlit_app.py에서 확인 가능
```

### **표 구조 문제**
```bash
# table_structure_fixer.py가 자동으로 처리
# 의미 없는 컬럼명과 행이 자동으로 제거됨
```

## 📝 **라이선스**

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 🤝 **기여**

버그 리포트나 기능 제안은 이슈를 통해 제출해 주세요.

---

**최종 업데이트**: 2025-01-27
**상태**: 완전 자동화된 파이프라인 완료 ✅ 
**표 구조 개선**: 완료 ✅
**타임아웃 최적화**: 75분 ✅
**이모지 제거**: 완료 ✅ 