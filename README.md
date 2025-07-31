# 가이드라인 문서 분석 시스템

## 프로젝트 개요

이 프로젝트는 PDF 문서에서 텍스트를 추출하고, LLM을 활용하여 의미적 청킹과 요약을 생성하는 시스템입니다. 

## 주요 기능

### 1. PDF 텍스트 추출
- PyMuPDF를 활용한 고품질 텍스트 추출
- 표 제외 처리 (의도적으로 제외) Camelot
- 페이지별 구조화된 데이터 생성

### 2. 의미적 청킹 (Semantic Chunking)
- 의미적 완결성을 고려한 동적 청크 크기 조절
- TF-IDF 기반 문장 중요도 분석
- 최소/최대 청크 크기 제한 (200-800자)

### 3. LLM 기반 요약 생성
- Mistral 모델을 활용한 개선된 요약 생성
- 50자 이내의 간결하고 명확한 요약
- 신뢰도 기반 품질 평가

### 4. 임베딩 및 검색
- Sentence Transformers를 활용한 벡터 생성
- FAISS를 통한 고속 유사도 검색
- 메타데이터 기반 필터링

### 5. 웹 인터페이스
- Streamlit 기반 대시보드
- 실시간 문서 분석 및 검색
- 결과 시각화

## 파일 구조

```
├── improved_summary_pipeline.py    # 개선된 요약 파이프라인
├── requirements_final.txt          # 의존성 패키지 목록
├── .gitignore                     # Git 제외 파일 목록
├── README.md                      # 프로젝트 설명서
├── src/                           # 소스 코드 디렉토리
│   ├── streamlit_app_*.py         # Streamlit 애플리케이션들
│   ├── semantic_chunk_generator.py # 의미적 청킹 생성기
│   ├── llm_section_keyword_extractor.py # LLM 섹션/키워드 추출기
│   └── ...
├── scripts/                       # 스크립트 디렉토리
└── docs/                          # 문서 디렉토리
```

## 사용자가 생성해야 하는 파일들

다음 파일들은 사용자가 직접 생성해야 합니다:

- `input.pdf`: 분석할 PDF 파일
- `full_pipeline_llm_enhanced_chunks.json`: LLM 강화 청크 결과
- `final_text_only_data.json`: 텍스트 전용 추출 결과
- `embeddings_improved.faiss`: 임베딩 벡터 파일
- `embeddings_improved_metadata.pkl`: 임베딩 메타데이터
- `venv/`: 가상환경 디렉토리

## 설치 및 실행

### 1. 가상환경 설정
```bash
# 가상환경 활성화
source environments/venv_web_new/bin/activate

# 의존성 설치
pip install -r requirements_final.txt
```

### 2. Ollama 설정
```bash
# Ollama 설치 (Linux)
curl -fsSL https://ollama.ai/install.sh | sh

# Mistral 모델 다운로드
ollama pull mistral
```

### 3. 개선된 요약 파이프라인 실행
```bash
python improved_summary_pipeline.py
```

### 4. Streamlit 애플리케이션 실행
```bash
streamlit run src/streamlit_app_enhanced.py
```

## 주요 개선사항

### 요약 품질 개선
- 50자 이내의 간결한 요약 생성
- 반복적 표현 제거
- 전문 용어 유지 및 설명 추가
- 신뢰도 기반 품질 평가

### 동적 청크 크기 조절
- 의미적 경계 기반 청크 분할
- 문장 중요도 분석
- 최소/최대 크기 제한으로 품질 보장

### 표 제외 처리
- 의도적으로 표 추출 제외
- 텍스트 중심의 분석에 집중

## 결과 파일

- `improved_summary_chunks.json`: 개선된 요약이 포함된 청크 데이터
- `full_pipeline_llm_enhanced_chunks.json`: LLM 강화 청크 데이터
- `embeddings_improved.faiss`: 임베딩 벡터 파일
- `embeddings_improved_metadata.pkl`: 임베딩 메타데이터

## 기술 스택

- **PDF 처리**: PyMuPDF
- **LLM**: Ollama + Mistral
- **자연어 처리**: NLTK, scikit-learn
- **임베딩**: Sentence Transformers, FAISS
- **웹 인터페이스**: Streamlit
- **데이터 처리**: pandas, numpy

## 라이선스

이 프로젝트는 교육 및 연구 목적으로 개발되었습니다. 
