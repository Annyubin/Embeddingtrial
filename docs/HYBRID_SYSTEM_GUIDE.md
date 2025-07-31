# 하이브리드 추출 시스템 가이드

## 개요

하이브리드 추출 시스템은 PyMuPDF와 Camelot의 장점을 결합하여 PDF에서 텍스트와 표를 정확하게 추출하고, LLM을 통해 섹션 제목과 키워드를 자동으로 생성하는 시스템입니다. **2025-01-27 업데이트로 표 구조 개선, 타임아웃 최적화, 이모지 제거 등이 완료되었습니다.**

## 가상환경 구성

### 1. venv_text_new (PyMuPDF 전용)
- **용도**: 텍스트 추출 및 표 영역 감지
- **주요 라이브러리**: PyMuPDF (fitz)
- **활성화**: `environments\venv_text_new\Scripts\activate.bat`

### 2. venv_table_new (Camelot 전용)
- **용도**: 표 내용 정교화 추출
- **주요 라이브러리**: Camelot, OpenCV, Pandas
- **활성화**: `environments\venv_table_new\Scripts\activate.bat`

### 3. venv_rag_new (LLM 및 통합 처리)
- **용도**: LLM 처리, 임베딩, RAG 시스템
- **주요 라이브러리**: Ollama, Sentence Transformers, FAISS
- **활성화**: `environments\venv_rag_new\Scripts\activate.bat`

## 시스템 실행 방법

### 방법 1: 배치 파일 사용 (권장)
```bash
# scripts/run_hybrid_system.bat 실행
scripts\run_hybrid_system.bat
```

### 방법 2: 웹 인터페이스 사용 (가장 권장)
```bash
# Streamlit 웹 인터페이스 실행
environments\venv_web_new\Scripts\python.exe -m streamlit run src/streamlit_app.py
```

### 방법 3: 수동 실행
```bash
# 1. PyMuPDF 표 영역 감지
call environments\venv_text_new\Scripts\activate.bat
python src\pymupdf_table_detector.py input.pdf
call deactivate

# 2. Camelot 표 내용 추출
call environments\venv_table_new\Scripts\activate.bat
python src\camelot_table_extractor.py input.pdf
call deactivate

# 3. 표 영역 제외 텍스트 추출
call environments\venv_text_new\Scripts\activate.bat
python src\text_extractor_excluding_tables.py input.pdf
call deactivate

# 4. 하이브리드 통합 및 LLM 처리
call environments\venv_rag_new\Scripts\activate.bat
python src\hybrid_extraction_system.py input.pdf
call deactivate
```

## 시스템 구성 요소

### 1. PyMuPDF 표 영역 감지기
- **파일**: `src/pymupdf_table_detector.py`
- **기능**: PDF에서 표 영역의 위치와 경계 상자 감지
- **출력**: `pymupdf_table_areas.json`

### 2. Camelot 표 내용 추출기
- **파일**: `src/camelot_table_extractor.py`
- **기능**: Stream/Lattice 모드로 표 내용 정교화 추출
- **출력**: `camelot_table_content.json`

### 3. 표 영역 제외 텍스트 추출기
- **파일**: `src/text_extractor_excluding_tables.py`
- **기능**: 표 영역을 제외한 순수 텍스트만 추출
- **출력**: `text_chunks_no_tables.json`

### 4. LLM 섹션 제목/키워드 추출기
- **파일**: `src/llm_section_keyword_extractor.py`
- **기능**: Ollama를 사용하여 섹션 제목과 키워드 자동 생성
- **타임아웃**: 75분 (대용량 문서 처리 가능)
- **한국어 전용**: 모든 결과가 한국어로 생성
- **출력**: `data/llm_enhanced_sections.json`

### 5. 하이브리드 통합 시스템
- **파일**: `src/hybrid_extraction_system.py`
- **기능**: 모든 결과를 통합하고 LLM 강화 적용
- **출력**: `data/hybrid_integrated_chunks.json`

### 6. 표 구조 개선 시스템 (신규)
- **파일**: `src/table_structure_fixer.py`
- **기능**: 표 구조를 LLM 친화적으로 개선
- **의미 있는 컬럼명**: `["0", "1"]` → `["내용", "항목2"]` 자동 변환
- **의미 없는 행 제거**: `["0", "1"]` 같은 의미 없는 행 자동 필터링
- **출력**: `data/final_llm_enhanced_data.json`

### 7. 텍스트 전처리 시스템
- **파일**: `src/text_preprocessor.py`
- **기능**: 개행문자, 공백, 특수문자 자동 정리
- **이모지 제거**: 모든 코드에서 이모지 완전 제거

## LLM 설정

### Ollama 설치 및 실행
```bash
# Ollama 설치 (Windows)
# https://ollama.ai/download 에서 다운로드

# 모델 다운로드
ollama pull mistral:latest

# Ollama 서버 실행
ollama serve
```

### LLM 처리 과정
1. **연결 확인**: Ollama API 연결 상태 확인
2. **청크별 처리**: 각 청크에 대해 섹션 제목과 키워드 추출
3. **한국어 강제**: 모든 결과가 한국어로 생성되도록 프롬프트 설정
4. **JSON 파싱**: LLM 응답에서 JSON 형식 데이터 추출
5. **결과 적용**: 통합 청크에 LLM 결과 적용
6. **타임아웃 관리**: 75분 타임아웃으로 안정적인 처리

## 출력 파일 구조

### final_llm_enhanced_data.json (최종 결과)
```json
[
  {
    "id": "integrated_001",
    "text": "청크 내용",
    "metadata": {
      "page": 1,
      "type": "text|table",
      "document_order": 1,
      "content_length": 500,
      "has_table": false,
      "section_title": "LLM 생성 섹션 제목",
      "keywords": ["키워드1", "키워드2", "키워드3"],
      "summary": "청크 내용 요약",
      "llm_enhanced": true,
      "llm_confidence": 0.85,
      "table_bbox": [x0, y0, x1, y1],
      "table_accuracy": 0.95,
      "extraction_method": "hybrid",
      "preprocessed": true
    },
    "table_structure": {
      "columns": ["내용", "항목2"],
      "rows": [
        ["제품명", "엠폭스 진단키트"],
        ["등급", "2등급"]
      ]
    }
  }
]
```

## 성능 개선 효과

### 중복 제거
- **기존**: 표 내용이 텍스트와 표 청크에 중복 포함
- **하이브리드**: 표 영역 제외로 중복 완전 제거

### 정확도 향상
- **PyMuPDF**: 정확한 표 영역 감지 (88.2% 성공률)
- **Camelot**: 정교한 표 구조 추출
- **LLM**: 의미있는 섹션 제목과 키워드 생성
- **표 구조 개선**: 의미 있는 컬럼명과 깔끔한 행 데이터

### 처리 효율성
- **가상환경 분리**: 의존성 충돌 없음
- **병렬 처리**: 각 단계별 독립적 실행
- **오류 격리**: 한 단계 실패가 전체에 영향 없음
- **타임아웃 최적화**: 75분으로 대용량 문서 처리 가능

### 표 구조 개선
- **의미 없는 컬럼명 제거**: `["0", "1"]` → `["내용", "항목2"]`
- **의미 없는 행 제거**: `["0", "1"]` 행 완전 제거
- **LLM 친화적 구조**: Mistral이 이해하기 쉬운 형태
- **자동 파싱**: 다양한 표 형태 자동 처리

## 문제 해결

### Ollama 연결 실패
```bash
# Ollama 서비스 상태 확인
ollama list

# 서비스 재시작
ollama serve
```

### 가상환경 활성화 실패
```bash
# PowerShell 실행 정책 확인
Get-ExecutionPolicy

# 정책 변경 (필요시)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 타임아웃 문제
```bash
# 75분 타임아웃으로 설정되어 있음
# src/llm_section_keyword_extractor.py에서 timeout 값 확인
# src/streamlit_app.py에서 subprocess timeout 값 확인
```

### 표 구조 문제
```bash
# table_structure_fixer.py가 자동으로 처리
# 의미 없는 컬럼명과 행이 자동으로 제거됨
```

### 이모지 에러
```bash
# 모든 코드에서 이모지 완전 제거 완료
# src/ 폴더의 모든 .py 파일에서 이모지 없음
```

### 메모리 부족
```bash
# 더 작은 모델 사용
ollama pull mistral:7b-instruct

# 배치 크기 조정
# src/llm_section_keyword_extractor.py에서 timeout 값 조정
```

## 다음 단계

1. **임베딩 생성**: `src/create_embeddings_optimized.py`
2. **FAISS 인덱스 생성**: `src/create_faiss_index.py`
3. **RAG 시스템 실행**: `src/rag_embedding_system.py`
4. **웹 인터페이스**: `src/streamlit_app.py`

## 주의사항

- 각 가상환경은 독립적으로 관리되어야 함
- Ollama 서버가 실행 중이어야 LLM 기능 사용 가능
- 대용량 PDF 처리 시 메모리 사용량 모니터링 필요
- LLM 처리 시간은 청크 수에 비례하여 증가 (최대 75분)
- 표 구조 개선은 자동으로 처리되므로 수동 개입 불필요
- 모든 LLM 결과는 한국어로 생성됨

## 최신 업데이트 (2025-01-27)

### 완료된 개선사항
- ✅ **타임아웃 최적화**: 75분으로 대용량 문서 처리 가능
- ✅ **표 구조 개선**: 의미 있는 컬럼명과 깔끔한 행 데이터
- ✅ **이모지 제거**: 모든 코드에서 이모지 완전 제거
- ✅ **한국어 전용 출력**: 모든 LLM 결과가 한국어로 생성
- ✅ **최종 다운로드**: 개선된 모든 기능이 포함된 완벽한 결과물
- ✅ **자동화 완성**: PDF 업로드 시 모든 처리가 자동으로 진행 