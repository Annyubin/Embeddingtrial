# 하이브리드 PDF 처리 시스템 로직 보고서

## 📋 개요

본 보고서는 PDF 문서에서 텍스트와 표를 통합적으로 추출하고 구조화하는 하이브리드 시스템의 전체 로직과 아키텍처를 설명합니다.

## 🏗️ 시스템 아키텍처

### 핵심 구성 요소
- **PyMuPDF (fitz)**: PDF 텍스트 추출 및 표 영역 감지
- **Camelot**: 고정밀 표 내용 추출 (Stream/Lattice 모드)
- **Ollama + Mistral**: LLM 기반 의미 분석 및 구조화
- **FAISS**: 벡터 유사도 검색 (향후 RAG 구현용)
- **Streamlit**: 웹 기반 사용자 인터페이스

### 가상환경 구성
```
environments/
├── venv_text_new/     # 텍스트 처리용
├── venv_table_new/    # 표 추출용  
├── venv_rag_new/      # LLM 처리용
└── venv_web_new/      # 웹 인터페이스용
```

## 🔄 처리 단계별 상세 로직

### 1단계: 표 위치 감지 (PyMuPDF)
**목적**: PDF 전체에서 표로 보이는 영역의 좌표를 미리 감지

**처리 과정**:
```python
# pymupdf_table_detector.py
for page in pdf:
    blocks = page.get_text("dict")["blocks"]
    for block in blocks:
        if block["type"] == 1:  # 이미지/표 블록
            if is_table_like(block):
                table_areas.append({
                    "page": page_num,
                    "bbox": block["bbox"],
                    "type": "table"
                })
```

**출력**: `pymupdf_table_areas.json`
```json
[
  {"page": 1, "bbox": [100, 200, 400, 300], "type": "table"},
  {"page": 2, "bbox": [50, 150, 350, 250], "type": "table"}
]
```

### 2단계: 표 내용 추출 (Camelot)
**목적**: 감지된 표 영역에서 정확한 셀 단위 내용 추출

**처리 과정**:
```python
# camelot_table_extractor.py
tables = camelot.read_pdf(
    pdf_path,
    pages='all',
    flavor='stream'  # 선이 없는 표
)
tables += camelot.read_pdf(
    pdf_path, 
    pages='all',
    flavor='lattice'  # 선이 있는 표
)
```

**출력**: `camelot_table_content.json`
```json
[
  {
    "page": 1,
    "bbox": [100, 200, 400, 300],
    "content": "품목 | 내용\n제품명 | 엠폭스 진단키트",
    "accuracy": 95.2
  }
]
```

### 3단계: 일반 텍스트 추출 (PyMuPDF)
**목적**: 표 영역을 제외한 본문 텍스트를 문단 단위로 추출

**처리 과정**:
```python
# text_extractor_excluding_tables.py
for page in pdf:
    # 1단계에서 감지된 표 영역 제외
    excluded_areas = get_table_areas_for_page(page_num)
    
    # 텍스트 블록 추출 (위치 정보 포함)
    text_blocks = page.get_text("dict")["blocks"]
    for block in text_blocks:
        if not overlaps_with_tables(block, excluded_areas):
            text += extract_text_from_block(block)
    
    # Y 위치 정보와 함께 청크 생성
    chunks = create_text_chunks_with_position(text, page_num, page)
```

**출력**: `text_chunks_no_tables.json`
```json
[
  {
    "chunk_id": "text_1_001",
    "content": "의약품의 시판 후 정기적인 안전성정보...",
    "page": 1,
    "y_position": 50,
    "chunk_type": "text_only"
  }
]
```

### 4단계: 위치 기반 통합
**목적**: 표와 텍스트를 문서 순서대로 정확히 통합

**핵심 알고리즘**:
```python
# hybrid_extraction_system.py

# 1. 모든 청크 수집 (Y 위치 정보 포함)
all_chunks = []
for text_chunk in text_chunks:
    all_chunks.append({
        'text': text_chunk['content'],
        'page': text_chunk['page'],
        'y_position': text_chunk['y_position'],
        'type': 'text'
    })

for pymupdf_table in pymupdf_tables:
    # PyMuPDF 표와 Camelot 표 매칭
    camelot_match = find_best_camelot_match(pymupdf_table)
    if camelot_match:
        all_chunks.append({
            'text': camelot_match['content'],
            'page': pymupdf_table['page'],
            'y_position': pymupdf_table['bbox'][1],
            'type': 'table'
        })

# 2. 페이지와 Y 위치로 정렬
integrated_chunks = sorted(all_chunks, key=lambda x: (
    x['page'], 
    x['y_position']
))

# 3. document_order 재할당
for i, chunk in enumerate(integrated_chunks, 1):
    chunk['id'] = f"integrated_{i:03d}"
    chunk['metadata']['document_order'] = i
```

**출력**: `data/hybrid_integrated_chunks.json`
```json
[
  {
    "id": "integrated_001",
    "text": "의약품의 시판 후...",
    "metadata": {
      "page": 1,
      "document_order": 1,
      "type": "text",
      "y_position": 50
    }
  },
  {
    "id": "integrated_002", 
    "text": "품목 | 내용\n제품명 | 엠폭스 진단키트",
    "metadata": {
      "page": 1,
      "document_order": 2,
      "type": "table",
      "y_position": 150
    }
  }
]
```

### 5단계: LLM 구조화 처리
**목적**: 각 청크에 대한 의미 분석 및 구조화 정보 생성

**프롬프트 엔지니어링**:
```python
# llm_section_keyword_extractor.py
prompt = f"""
다음 텍스트를 분석하여 한국어로만 답변하세요:

텍스트: {chunk_text}

다음 형식으로 출력하세요:
- section_title: 섹션 제목 (한국어)
- keywords: 키워드 3-5개 (한국어)
- summary: 요약 (한국어, 50자 이내)

예시:
- section_title: 의약품 안전성 관리
- keywords: ["의약품", "안전성", "관리", "모니터링"]
- summary: 의약품 시판 후 안전성 정보 수집 및 관리 방안
"""
```

**출력**: `data/llm_enhanced_sections.json`
```json
[
  {
    "chunk_id": "integrated_001",
    "section_title": "의약품 안전성 관리",
    "keywords": ["의약품", "안전성", "관리", "모니터링"],
    "summary": "의약품 시판 후 안전성 정보 수집 및 관리 방안"
  }
]
```

### 6단계: 최종 데이터 생성 및 보완
**목적**: 모든 정보를 통합하고 표 구조를 정제

**처리 과정**:
```python
# create_final_llm_enhanced_data.py
for chunk in integrated_chunks:
    llm_info = find_llm_info(chunk['id'])
    final_chunk = {
        **chunk,
        'section_title': llm_info['section_title'],
        'keywords': llm_info['keywords'], 
        'summary': llm_info['summary']
    }
    final_data.append(final_chunk)

# table_structure_fixer.py
for chunk in final_data:
    if chunk['metadata']['type'] == 'table':
        table_structure = parse_table_text(chunk['text'])
        chunk['table_structure'] = table_structure
```

**출력**: `data/final_llm_enhanced_data.json`
```json
[
  {
    "id": "integrated_001",
    "text": "의약품의 시판 후...",
    "metadata": {"page": 1, "type": "text", "y_position": 50},
    "section_title": "의약품 안전성 관리",
    "keywords": ["의약품", "안전성", "관리"],
    "summary": "의약품 시판 후 안전성 정보 수집 및 관리 방안"
  },
  {
    "id": "integrated_002",
    "text": "품목 | 내용\n제품명 | 엠폭스 진단키트",
    "metadata": {"page": 1, "type": "table", "y_position": 150},
    "section_title": "품목분류표",
    "keywords": ["품목", "분류", "제품"],
    "summary": "의료기기 품목 분류 정보",
    "table_structure": {
      "columns": ["품목", "내용"],
      "rows": [["제품명", "엠폭스 진단키트"]]
    }
  }
]
```

## 🎯 핵심 알고리즘

### BBOX 유사도 계산
```python
def calculate_bbox_similarity(bbox1, bbox2):
    """두 BBOX 간의 유사도 계산 (IoU 기반)"""
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    
    # 교집합 영역 계산
    x_left = max(x1, x3)
    y_top = max(y1, y3)
    x_right = min(x2, x4)
    y_bottom = min(y2, y4)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    union = (x2 - x1) * (y2 - y1) + (x4 - x3) * (y4 - y3) - intersection
    
    return intersection / union if union > 0 else 0.0
```

### 표 구조 파싱
```python
def parse_table_text(text):
    """표 텍스트를 columns/rows 구조로 변환"""
    lines = text.strip().split('\n')
    
    if len(lines) < 2:
        return {"columns": ["내용"], "rows": [[text]]}
    
    # 첫 번째 줄을 헤더로 사용
    header = lines[0]
    if " | " in header:
        columns = [col.strip() for col in header.split(" | ")]
    else:
        columns = ["항목", "내용"]
    
    # 나머지 줄을 행으로 처리
    rows = []
    for line in lines[1:]:
        if " | " in line:
            row = [cell.strip() for cell in line.split(" | ")]
        else:
            row = [line.strip()]
        rows.append(row)
    
    return {"columns": columns, "rows": rows}
```

## ⏱️ 성능 특성

### 처리 시간 (PDF 50페이지 기준)
| 단계 | 예상 시간 | 주요 병목 |
|------|-----------|-----------|
| 1-3단계 | 1-2분 | PDF 스캔 속도 |
| 4단계 | 1분 | 정렬 알고리즘 |
| 5단계 | 30-45분 | LLM API 호출 |
| 6단계 | 1분 | 데이터 통합 |

### 메모리 사용량
- **PyMuPDF**: PDF 크기에 비례
- **Camelot**: 표 개수에 비례  
- **LLM 처리**: 청크 수에 비례
- **최종 통합**: 전체 데이터 크기

## 🔧 설정 및 제한사항

### 타임아웃 설정
- **전체 시스템**: 75분 (4500초)
- **LLM 처리**: 75분 (4500초)
- **하이브리드 시스템**: 75분 (4500초)

### LLM 모델
- **모델**: Mistral (Ollama)
- **API 엔드포인트**: http://localhost:11434/api/generate
- **언어**: 한국어 전용 출력

### 파일 크기 제한
- **입력 PDF**: 100MB 이하 권장
- **출력 JSON**: 청크 수에 따라 가변

## 🚀 향후 개선 방안

### 1. PyMuPDF → Camelot 직접 연동
```
현재: PyMuPDF(좌표) + Camelot(전체스캔) → 매칭
개선: PyMuPDF(좌표) → Camelot(table_areas=좌표) → 정확한 추출
```

### 2. 표 구조화 고도화
```
현재: 파이프 구분자 텍스트
개선: columns/rows 구조로 미리 파싱
```

### 3. RAG 시스템 통합
```
현재: 구조화된 데이터 생성
향후: FAISS 인덱싱 + 벡터 검색 기능 추가
```

## 📊 품질 지표

### 정확도 측정
- **표 감지 정확도**: PyMuPDF BBOX vs 실제 표 위치
- **표 내용 정확도**: Camelot 추출 vs 원본 표
- **순서 정확도**: Y 위치 기반 정렬 vs 실제 문서 순서
- **LLM 분석 정확도**: 생성된 section_title/keywords 품질

### 성능 지표
- **처리 속도**: 페이지당 평균 처리 시간
- **메모리 효율성**: 대용량 PDF 처리 능력
- **확장성**: 다양한 PDF 형식 지원

---

**작성일**: 2024년 12월  
**버전**: 1.0  
**담당자**: AI Assistant 