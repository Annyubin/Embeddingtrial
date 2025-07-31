#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM 섹션 제목 및 키워드 추출기
Ollama를 사용하여 청크에서 섹션 제목과 키워드를 추출
"""

import json
import sys
import re
import requests
from pathlib import Path
from typing import Dict, List, Any

def preprocess_text(text: str) -> str:
    """텍스트 전처리 함수"""
    if not text:
        return ""
    
    # 1. 연속된 개행문자 정리 (3개 이상을 2개로)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 2. 연속된 공백 정리 (2개 이상을 1개로)
    text = re.sub(r' {2,}', ' ', text)
    
    # 3. 줄바꿈 후 공백 제거
    text = re.sub(r'\n +', '\n', text)
    
    # 4. 공백 후 줄바꿈 제거
    text = re.sub(r' +\n', '\n', text)
    
    # 5. 문장 시작/끝의 불필요한 공백 제거
    text = text.strip()
    
    # 6. 특수문자 정리 (연속된 특수문자 정리)
    text = re.sub(r'[^\w\s\n\.\,\;\:\!\?\(\)\[\]\{\}\-\+\=\*\/\@\#\$\%\&\*\(\)]+', '', text)
    
    # 7. 연속된 마침표 정리
    text = re.sub(r'\.{3,}', '...', text)
    
    # 8. 연속된 쉼표 정리
    text = re.sub(r',{2,}', ',', text)
    
    # 9. 빈 줄 제거 (문단 구분은 유지)
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        if line.strip():  # 빈 줄이 아닌 경우만 추가
            cleaned_lines.append(line.strip())
    
    # 10. 문단 구분을 위한 개행 추가
    text = '\n\n'.join(cleaned_lines)
    
    # 11. 추가 정리: 연속된 개행을 하나로
    text = re.sub(r'\n{2,}', '\n', text)
    
    # 12. 문장 끝 정리
    text = re.sub(r'\n+$', '', text)  # 끝의 개행 제거
    text = re.sub(r'^\n+', '', text)  # 시작의 개행 제거
    
    # 13. 연속된 공백 최종 정리
    text = re.sub(r' +', ' ', text)
    
    # 14. 특수 유니코드 문자 제거 (Ÿ 등)
    text = re.sub(r'[^\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F\w\s\.\,\;\:\!\?\(\)\[\]\{\}\-\+\=\*\/\@\#\$\%\&\*\(\)]', '', text)
    
    return text

class LLMSectionKeywordExtractor:
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.model = "mistral:latest" 
        
    def check_ollama_connection(self) -> bool:
        """Ollama 연결 확인"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=30)
            return response.status_code == 200
        except Exception as e:
            print(f"[WARNING] Ollama 연결 실패: {e}")
            return False
    
    def extract_section_info(self, chunk_text: str, chunk_type: str, chunk_id: str) -> Dict:
        """개별 청크에서 섹션 제목과 키워드 추출 (전처리 포함)"""
        
        # 텍스트 전처리 적용
        cleaned_text = preprocess_text(chunk_text)
        
        # 빈 텍스트나 의미 없는 텍스트 검증
        if self._is_empty_or_meaningless(cleaned_text):
            print(f"[SKIP] 청크 {chunk_id}: 빈 내용 또는 의미 없는 내용 - LLM 처리 건너뜀")
            return {
                'chunk_id': chunk_id,
                'section_title': '[빈 내용]',
                'keywords': ['빈 내용'],
                'confidence': 0.0,
                'content_type': chunk_type,
                'summary': '[빈 내용] 내용이 없거나 의미가 없는 데이터입니다.',
                'document_section': '',
                'extraction_method': 'empty_content_skip'
            }
        
        # 프롬프트 구성 (전처리된 텍스트 사용)
        if chunk_type == 'table':
            chunk_content = cleaned_text[:2000]  # 표 내용이 길면 앞부분만 사용
            prompt = f"""
다음 표 내용을 분석하여 적절한 섹션 제목, 키워드, 요약을 추출해주세요.

**매우 중요**: 모든 응답은 반드시 한국어로만 작성해주세요. 영어, 라틴어, 약어 등 다른 언어는 절대 사용하지 마세요. 
키워드도 반드시 한국어로만 작성하고, 영어 약어나 전문용어도 한국어로 번역해서 사용해주세요.

**번역 규칙**:
- 모든 영어 용어는 반드시 한국어로 번역해주세요
- 약어(CDC, ATS, HIV 등)도 한국어로 번역해주세요
- 전문 용어도 일반인이 이해할 수 있는 한국어로 번역해주세요

표 내용:
{chunk_content}

**분석 지침:**
1. 표의 전체적인 목적과 주제를 파악하세요.
2. 개별 데이터가 아닌 표의 의미를 추출하세요.
3. 섹션 제목은 표의 종류나 용도를 나타내야 합니다.
4. 키워드는 표에서 다루는 주요 개념(주제, 분류, 속성 등)을 5~10개로 뽑으세요.
5. summary는 표 전체 맥락을 한 문장으로 명확하게 요약하세요.
6. 모든 키워드는 반드시 한국어로만 작성해주세요.

다음 JSON 형식으로 응답해주세요 (모든 필드는 한국어로만):
{{
    "section_title": "표의 주제를 나타내는 간결한 한국어 제목 (예: 일자 및 담당자 정보, 품목분류표, 안전성평가표)",
    "keywords": ["한국어키워드1", "한국어키워드2", "한국어키워드3", ...],
    "confidence": 0.85,
    "content_type": "table",
    "summary": "표의 주요 내용을 한 문장으로 한국어 요약"
}}

**섹션 제목 예시 (한국어만):**
- "일자 및 담당자 정보" (날짜와 담당자 정보가 있는 표)
- "품목분류표" (품목을 분류하는 표)
- "안전성평가표" (안전성을 평가하는 표)
- "허가요건표" (허가 요건을 정리한 표)
- "시험방법표" (시험 방법을 설명하는 표)

**키워드 예시 (한국어만):**
- "임상시험", "안전성", "유효성", "약물동태", "투여량", "부작용"
- "결핵치료", "항생제", "미생물학", "배양검사", "약물내성"
"""
        else:
            chunk_content = cleaned_text[:2000]  # 텍스트가 길면 앞부분만 사용
            prompt = f"""
다음 텍스트 내용을 분석하여 적절한 섹션 제목과 키워드를 추출해주세요.

**매우 중요**: 모든 응답은 반드시 한국어로만 작성해주세요. 영어, 라틴어, 약어 등 다른 언어는 절대 사용하지 마세요. 
키워드도 반드시 한국어로만 작성하고, 영어 약어나 전문용어도 한국어로 번역해서 사용해주세요.

**번역 규칙**:
- 모든 영어 용어는 반드시 한국어로 번역해주세요
- 약어(CDC, ATS, HIV 등)도 한국어로 번역해주세요
- 전문 용어도 일반인이 이해할 수 있는 한국어로 번역해주세요

텍스트 내용:
{chunk_content}

다음 JSON 형식으로 응답해주세요 (모든 필드는 한국어로만):
{{
    "section_title": "텍스트의 주제를 나타내는 간결한 한국어 제목",
    "keywords": ["한국어키워드1", "한국어키워드2", "한국어키워드3", "한국어키워드4", "한국어키워드5"],
    "confidence": 0.85,
    "content_type": "text",
    "summary": "텍스트의 주요 내용을 한 문장으로 한국어 요약",
    "document_section": "문서의 어느 부분인지 한국어로 설명"
}}

섹션 제목 예시 (한국어만):
- "서문" (문서 시작 부분)
- "목적" (목적 설명 부분)
- "방법" (방법론 설명 부분)
- "결과" (결과 설명 부분)
- "결론" (결론 부분)

키워드 예시 (한국어만):
- "임상시험", "안전성", "유효성", "약물동태", "투여량", "부작용"
- "결핵치료", "항생제", "미생물학", "배양검사", "약물내성"
- "시험방법", "평가기준", "통계분석", "데이터수집", "결과해석"
"""
        
        try:
            print(f"[DEBUG] Ollama API 호출 시작: {self.ollama_url}")
            print(f"[DEBUG] 모델: {self.model}")
            print(f"[DEBUG] 프롬프트 길이: {len(prompt)}")
            
            # Ollama API 호출
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "max_tokens": 500,
                        "num_ctx": 4096
                    }
                },
                timeout=4500 # 75 minutes
            )
            
            print(f"[DEBUG] API 응답 상태: {response.status_code}")
            if response.status_code != 200:
                print(f"[DEBUG] API 오류 응답: {response.text}")
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '')
                
                print(f"[DEBUG] LLM 응답: {response_text[:200]}...")
                
                # JSON 파싱 시도
                try:
                    # JSON 부분만 추출
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    
                    if json_start != -1 and json_end > json_start:
                        json_str = response_text[json_start:json_end]
                        print(f"[DEBUG] 추출된 JSON: {json_str}")
                        parsed_result = json.loads(json_str)
                        
                        return {
                            'chunk_id': chunk_id,
                            'section_title': parsed_result.get('section_title', '기본 섹션'),
                            'keywords': parsed_result.get('keywords', []),
                            'confidence': parsed_result.get('confidence', 0.5),
                            'content_type': parsed_result.get('content_type', chunk_type),
                            'summary': parsed_result.get('summary', ''),
                            'document_section': parsed_result.get('document_section', ''),
                            'extraction_method': 'llm_ollama'
                        }
                except json.JSONDecodeError as e:
                    print(f"[WARNING] JSON 파싱 실패: {e}")
                    print(f"[WARNING] 원본 응답: {response_text}")
                    # fallback 발생 시 원본 응답 저장
                    safe_chunk_id = chunk_id.replace('/', '_').replace('\\', '_').replace(':', '_')
                    fallback_filename = f"llm_fallback_{safe_chunk_id}.txt"
                    with open(fallback_filename, "w", encoding="utf-8") as f:
                        f.write(response_text)
                    print(f"[FALLBACK] 청크 {chunk_id} LLM 파싱 실패. 원본 응답 {fallback_filename}에 저장됨.")
                    # fallback 정보 명확히 표시
                    return {
                        'chunk_id': chunk_id,
                        'section_title': '[LLM 파싱 실패] ' + self._generate_default_section_title(chunk_text, chunk_type),
                        'keywords': ['LLM 파싱 실패'],
                        'confidence': 0.0,
                        'content_type': chunk_type,
                        'summary': '[LLM 파싱 실패] LLM 응답 파싱에 실패했습니다.',
                        'document_section': '',
                        'extraction_method': 'llm_fallback'
                    }
            # 기본값 반환
            print(f"[FALLBACK] 청크 {chunk_id} LLM 호출 실패 또는 응답 없음.")
            return {
                'chunk_id': chunk_id,
                'section_title': '[LLM 파싱 실패] ' + self._generate_default_section_title(chunk_text, chunk_type),
                'keywords': ['LLM 파싱 실패'],
                'confidence': 0.0,
                'content_type': chunk_type,
                'summary': '[LLM 파싱 실패] LLM 호출 실패 또는 응답 없음.',
                'document_section': '',
                'extraction_method': 'llm_fallback'
            }
        except Exception as e:
            print(f"[ERROR] LLM 처리 실패: {e}")
            # 예외 발생 시에도 fallback 정보 명확히 표시
            return {
                'chunk_id': chunk_id,
                'section_title': '[LLM 파싱 실패] ' + self._generate_default_section_title(chunk_text, chunk_type),
                'keywords': ['LLM 파싱 실패'],
                'confidence': 0.0,
                'content_type': chunk_type,
                'summary': '[LLM 파싱 실패] LLM 처리 중 예외 발생.',
                'document_section': '',
                'extraction_method': 'llm_fallback'
            }
    
    def _is_empty_or_meaningless(self, text: str) -> bool:
        """텍스트가 빈 내용이거나 의미가 없는지 확인"""
        if not text:
            return True
        
        # 문자열이 아닌 경우 문자열로 변환
        if not isinstance(text, str):
            text = str(text)
        
        # 공백만 있는 경우
        if text.strip() == '':
            return True
        
        # 빈 리스트나 딕셔너리 문자열
        if text.strip() in ['[]', '{}', '[""]', '[\'\']', '["", ""]', '[\'\', \'\']']:
            return True
        
        # 빈 문자열만 포함된 리스트 패턴
        import re
        empty_list_patterns = [
            r'^\s*\[\s*["\']?\s*["\']?\s*\]\s*$',  # [""] 또는 [''] 또는 []
            r'^\s*\[\s*["\']\s*,\s*["\']\s*\]\s*$',  # ["", ""] 또는 ['', '']
            r'^\s*\[\s*["\']\s*["\']\s*\]\s*$',  # ["" ""] 또는 ['' '']
        ]
        
        for pattern in empty_list_patterns:
            if re.match(pattern, text.strip()):
                return True
        
        # 의미 없는 단일 문자나 숫자만 있는 경우
        meaningful_chars = 0
        for char in text:
            if char.isalnum() or char in '가-힣':
                meaningful_chars += 1
        
        # 의미 있는 문자가 3개 미만이면 의미 없음
        if meaningful_chars < 3:
            return True
        
        return False
    
    def extract_section_info_semantic(self, chunk_text: str, chunk_type: str, chunk_id: str, cluster_category: str) -> Dict:
        """의미 기반 청크에서 섹션 제목과 키워드 추출"""
        
        # 빈 텍스트나 의미 없는 텍스트 검증
        if self._is_empty_or_meaningless(chunk_text):
            print(f"[SKIP] 청크 {chunk_id}: 빈 내용 또는 의미 없는 내용 - LLM 처리 건너뜀")
            return {
                'chunk_id': chunk_id,
                'section_title': f'[{cluster_category}] 빈 내용',
                'keywords': [cluster_category, '빈 내용'],
                'confidence': 0.0,
                'content_type': chunk_type,
                'summary': f'[{cluster_category}] 내용이 없거나 의미가 없는 데이터입니다.',
                'document_section': cluster_category,
                'extraction_method': 'empty_content_skip',
                'cluster_category': cluster_category
            }
        
        # Ollama 연결 확인
        if not self.check_ollama_connection():
            print(f"[WARNING] Ollama 연결 실패 - 기본 결과 생성: {chunk_id}")
            return self._create_default_result_semantic(chunk_text, chunk_type, chunk_id, cluster_category)
        
        # LLM 프롬프트 생성 (클러스터 카테고리 정보 포함)
        prompt = self._create_semantic_prompt(chunk_text, cluster_category)
        
        try:
            # Ollama API 호출
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '')
                
                # 응답 파싱
                section_info = self._parse_semantic_response(response_text, chunk_id, cluster_category, chunk_text)
                return section_info
            else:
                print(f"[ERROR] Ollama API 호출 실패: {response.status_code}")
                return self._create_default_result_semantic(chunk_text, chunk_type, chunk_id, cluster_category)
                
        except Exception as e:
            print(f"[ERROR] LLM 처리 실패: {e}")
            return self._create_default_result_semantic(chunk_text, chunk_type, chunk_id, cluster_category)
    
    def _create_semantic_prompt(self, chunk_text: str, cluster_category: str) -> str:
        """의미 기반 청크용 프롬프트 생성"""
        chunk_content = chunk_text[:2000]  # 텍스트가 길면 앞부분만 사용
        
        return f"""
다음 {cluster_category} 관련 텍스트 내용을 분석하여 JSON 형식으로 응답해주세요.

**중요**: 반드시 JSON 형식으로만 응답하고, 다른 설명은 포함하지 마세요.

**클러스터 카테고리**: {cluster_category}

텍스트 내용:
{chunk_content}

응답 형식 (JSON만):
{{
    "section_title": "{cluster_category} 관련 텍스트의 주제를 나타내는 간결한 한국어 제목",
    "keywords": ["{cluster_category}관련키워드1", "{cluster_category}관련키워드2", "{cluster_category}관련키워드3", "키워드4", "키워드5"],
    "confidence": 0.85,
    "content_type": "semantic_clustered",
    "summary": "텍스트의 주요 내용을 한 문장으로 한국어 요약",
    "document_section": "{cluster_category} 관련 섹션"
}}

JSON 형식으로만 응답해주세요. 다른 텍스트는 포함하지 마세요.
"""
    
    def _parse_semantic_response(self, response_text: str, chunk_id: str, cluster_category: str, original_content: str) -> Dict:
        """의미 기반 응답 파싱"""
        try:
            # JSON 부분 추출 (더 정확한 방법)
            import re
            
            # JSON 패턴 찾기
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            matches = re.findall(json_pattern, response_text)
            
            if not matches:
                print(f"[WARNING] JSON 패턴을 찾을 수 없음: {chunk_id}")
                return self._create_default_result_semantic(original_content, "semantic_clustered", chunk_id, cluster_category)
            
            # 가장 긴 JSON 문자열 선택
            json_text = max(matches, key=len)
            
            # JSON 파싱
            data = json.loads(json_text)
            
            return {
                'chunk_id': chunk_id,
                'content': original_content,  # 원본 텍스트 유지
                'section_title': data.get('section_title', f'{cluster_category} 관련 내용'),
                'keywords': data.get('keywords', [cluster_category]),
                'confidence': data.get('confidence', 0.8),
                'content_type': 'semantic_clustered',
                'summary': data.get('summary', f'{cluster_category} 관련 내용입니다.'),
                'document_section': data.get('document_section', f'{cluster_category} 섹션'),
                'extraction_method': 'llm_semantic',
                'cluster_category': cluster_category
            }
            
        except json.JSONDecodeError as e:
            print(f"[ERROR] JSON 파싱 실패: {e}")
            print(f"[DEBUG] 응답 텍스트: {response_text[:200]}...")
            return self._create_default_result_semantic(original_content, "semantic_clustered", chunk_id, cluster_category)
        except Exception as e:
            print(f"[ERROR] 응답 파싱 실패: {e}")
            return self._create_default_result_semantic(original_content, "semantic_clustered", chunk_id, cluster_category)
    
    def _create_default_result_semantic(self, chunk_text: str, chunk_type: str, chunk_id: str, cluster_category: str) -> Dict:
        """의미 기반 기본 결과 생성"""
        # 간단한 섹션 제목 추출
        section_title = self._extract_section_title_from_text_simple(chunk_text, chunk_type)
        
        return {
            'chunk_id': chunk_id,
            'content': chunk_text,  # 원본 텍스트 유지
            'section_title': f'{cluster_category}: {section_title}',
            'keywords': [cluster_category, '기본키워드'],
            'confidence': 0.5,
            'content_type': chunk_type,
            'summary': f'{cluster_category} 관련 기본 요약',
            'document_section': f'{cluster_category} 섹션',
            'extraction_method': 'default_semantic_fallback',
            'cluster_category': cluster_category
        }
    
    def _extract_section_title_from_text_simple(self, chunk_text: str, chunk_type: str) -> str:
        """간단한 섹션 제목 추출 (LLM 없이)"""
        if not chunk_text.strip():
            return "빈 내용"
        
        # 첫 번째 의미있는 줄 찾기
        lines = chunk_text.split('\n')
        for line in lines[:5]:  # 처음 5줄 확인
            line = line.strip()
            if line and len(line) < 100 and len(line) > 3:
                # 특수문자 제거하고 간단하게
                clean_line = ''.join(c for c in line if c.isalnum() or c.isspace() or c in '가-힣')
                if clean_line and len(clean_line) > 2:
                    return clean_line[:50]  # 50자로 제한
        
        return f"{chunk_type} 관련 내용"
    
    def _create_default_results_semantic(self, chunks: List[Dict]) -> List[Dict]:
        """Ollama가 없을 때 기본 결과 생성 (의미 기반)"""
        results = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = chunk.get('chunk_id', f'chunk_{i}')
            chunk_text = chunk.get('content', '')
            cluster_category = chunk.get('cluster_category', '기타')
            
            results.append(self._create_default_result_semantic(chunk_text, 'semantic_clustered', chunk_id, cluster_category))
        
        return results
    
    def process_chunks(self, chunks_file: str) -> List[Dict]:
        """청크 파일을 처리하여 섹션 제목과 키워드 추출"""
        print(f"[INFO] 청크 파일 처리 시작: {chunks_file}")
        
        try:
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            
            print(f"[INFO] 청크 로드 완료: {len(chunks)}개")
            
            # Ollama 연결 확인
            if not self.check_ollama_connection():
                print("[WARNING] Ollama 연결 실패 - 기본 결과 생성")
                return self._create_default_results_semantic(chunks)
            
            results = []
            total_chunks = len(chunks)
            
            for i, chunk in enumerate(chunks, 1):
                print(f"[PROCESS] 청크 {i}/{total_chunks}: {chunk.get('chunk_id', 'unknown')}")
                
                # 의미 기반 청크 형식에 맞게 데이터 추출
                chunk_id = chunk.get('chunk_id', f'chunk_{i}')
                chunk_text = chunk.get('content', '')
                chunk_type = chunk.get('chunk_type', 'semantic_clustered')
                cluster_category = chunk.get('cluster_category', '기타')
                
                # 섹션 정보 추출 (클러스터 카테고리 정보 포함)
                section_info = self.extract_section_info_semantic(chunk_text, chunk_type, chunk_id, cluster_category)
                results.append(section_info)
                
                # 진행률 표시
                if i % 5 == 0 or i == total_chunks:
                    print(f"[PROGRESS] {i}/{total_chunks} 완료 ({i/total_chunks*100:.1f}%)")
            
            print(f"[SUCCESS] 청크 처리 완료: {len(results)}개")
            return results
            
        except Exception as e:
            print(f"[ERROR] 청크 처리 실패: {e}")
            return []
    
    def _extract_section_title_from_text(self, response_text: str, chunk_text: str, chunk_type: str) -> str:
        """LLM 응답 텍스트에서 섹션 제목 추출"""
        # "제목:" 또는 "section_title:" 패턴 찾기
        patterns = [
            r'제목[:\s]*([^\n\r]+)',
            r'section_title[:\s]*([^\n\r]+)',
            r'섹션[:\s]*([^\n\r]+)',
            r'제목[:\s]*"([^"]+)"',
            r'section_title[:\s]*"([^"]+)"'
        ]
        
        for pattern in patterns:
            import re
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                title = match.group(1).strip()
                if title and len(title) > 2 and title not in ['알 수 없음', '기본']:
                    return title
        
        # 패턴을 찾지 못한 경우 청크 텍스트에서 추출
        return self._generate_default_section_title(chunk_text, chunk_type)
    
    def _extract_keywords_from_text(self, response_text: str, chunk_text: str) -> List[str]:
        """LLM 응답 텍스트에서 키워드 추출 (한국어만 허용)"""
        # "키워드:" 또는 "keywords:" 패턴 찾기
        patterns = [
            r'키워드[:\s]*([^\n\r]+)',
            r'keywords[:\s]*([^\n\r]+)',
            r'키워드[:\s]*\[([^\]]+)\]',
            r'keywords[:\s]*\[([^\]]+)\]'
        ]
        
        for pattern in patterns:
            import re
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                keywords_text = match.group(1).strip()
                keywords = [kw.strip().strip('"\'') for kw in keywords_text.split(',') if kw.strip()]
                
                # 한국어 키워드만 필터링
                korean_keywords = []
                for kw in keywords:
                    if len(kw) > 1:
                        # 한국어 문자가 포함된 키워드만 허용
                        if re.search(r'[가-힣]', kw):
                            korean_keywords.append(kw)
                        # 영어만 있는 키워드는 제외
                        elif re.search(r'[a-zA-Z]', kw) and not re.search(r'[가-힣]', kw):
                            continue
                        else:
                            korean_keywords.append(kw)
                
                return korean_keywords[:5]  # 최대 5개
        
        return []
    
    def _generate_default_section_title(self, chunk_text: str, chunk_type: str) -> str:
        """청크 텍스트를 기반으로 기본 섹션 제목 생성"""
        if chunk_type == 'table':
            # 테이블의 경우 첫 번째 행이나 헤더에서 추출
            lines = chunk_text.split('\n')
            for line in lines[:3]:  # 처음 3줄 확인
                line = line.strip()
                if line and len(line) < 50:  # 짧은 줄이 헤더일 가능성
                    # 특수문자 제거
                    clean_line = ''.join(c for c in line if c.isalnum() or c.isspace() or c in '가-힣')
                    if clean_line and len(clean_line) > 2:
                        return clean_line[:20]  # 20자로 제한
            return "표 데이터"
        else:
            # 텍스트의 경우 첫 번째 줄에서 추출
            lines = chunk_text.split('\n')
            for line in lines[:2]:  # 처음 2줄 확인
                line = line.strip()
                if line and len(line) < 100:  # 너무 긴 줄 제외
                    # 특수문자 제거
                    clean_line = ''.join(c for c in line if c.isalnum() or c.isspace() or c in '가-힣')
                    if clean_line and len(clean_line) > 3:
                        return clean_line[:30]  # 30자로 제한
            return "문서 내용"
    
    def _create_default_results(self, chunks: List[Dict]) -> List[Dict]:
        """Ollama가 없을 때 기본 결과 생성"""
        results = []
        
        for chunk in chunks:
            chunk_id = chunk.get('id', 'unknown')
            chunk_type = chunk.get('metadata', {}).get('type', 'text')
            
            results.append({
                'chunk_id': chunk_id,
                'section_title': '[Ollama 연결 실패] ' + f'{chunk_type.capitalize()} 섹션',
                'keywords': ['Ollama 연결 실패'],
                'confidence': 0.0,
                'content_type': chunk_type,
                'summary': '[Ollama 연결 실패] Ollama 서버에 연결할 수 없습니다.',
                'document_section': '',
                'extraction_method': 'ollama_connection_failed'
            })
        
        return results
    
    def _create_default_results_text_only(self, chunks: List[Dict]) -> List[Dict]:
        """텍스트 전용 형식에 맞는 기본 결과 생성"""
        results = []
        
        for chunk in chunks:
            chunk_id = chunk.get('chunk_id', 'unknown')
            chunk_type = chunk.get('chunk_type', 'text_only')
            chunk_text = chunk.get('content', '')
            
            # 기본 섹션 제목 생성
            default_title = self._generate_default_section_title(chunk_text, chunk_type)
            
            results.append({
                'chunk_id': chunk_id,
                'section_title': f'[Ollama 연결 실패] {default_title}',
                'keywords': ['Ollama 연결 실패', '기본 처리'],
                'confidence': 0.0,
                'content_type': 'text_only',
                'summary': f'[Ollama 연결 실패] {default_title} 섹션입니다.',
                'document_section': '',
                'extraction_method': 'ollama_connection_failed'
            })
        
        return results
    
    def save_results(self, results: List[Dict], output_file: str = "data/llm_enhanced_sections.json"):
        """결과 저장"""
        try:
            # 출력 디렉토리 생성
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"[SUCCESS] 결과 저장: {output_file}")
            
            # 통계 출력
            total = len(results)
            llm_enhanced = len([r for r in results if r.get('extraction_method') == 'llm_ollama'])
            llm_fallback = len([r for r in results if r.get('extraction_method') == 'llm_fallback'])
            connection_failed = len([r for r in results if r.get('extraction_method') == 'ollama_connection_failed'])
            empty_content_skip = len([r for r in results if r.get('extraction_method') == 'empty_content_skip'])
            
            print(f"[STATS] 처리 결과:")
            print(f"  - 총 청크: {total}개")
            print(f"  - LLM 강화: {llm_enhanced}개")
            print(f"  - LLM 파싱 실패: {llm_fallback}개")
            print(f"  - Ollama 연결 실패: {connection_failed}개")
            print(f"  - 빈 내용 건너뛰기: {empty_content_skip}개")
            
        except Exception as e:
            print(f"[ERROR] 결과 저장 실패: {e}")

def main():
    if len(sys.argv) < 2:
        print("사용법: python llm_section_keyword_extractor.py <청크파일경로>")
        sys.exit(1)
    
    chunks_file = sys.argv[1]
    
    if not Path(chunks_file).exists():
        print(f"[ERROR] 파일을 찾을 수 없습니다: {chunks_file}")
        sys.exit(1)
    
    # LLM 추출기 실행
    extractor = LLMSectionKeywordExtractor()
    
    print("=" * 60)
    print("LLM 섹션 제목 및 키워드 추출기")
    print("=" * 60)
    
    # 청크 처리
    results = extractor.process_chunks(chunks_file)
    
    # 결과 저장
    extractor.save_results(results)
    
    print("=" * 60)
    print("LLM 추출 완료")
    print("=" * 60)

if __name__ == "__main__":
    main() 