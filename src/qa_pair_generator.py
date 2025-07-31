#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
질문-답변 쌍 자동 생성기
LLM을 사용하여 의약품 도메인 특화 질문-답변 쌍을 자동 생성
"""
import json
import pickle
import numpy as np
from typing import List, Dict, Tuple
import os
import sys
import requests
import time
import re

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
    
    return text

class QAPairGenerator:
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        """질문-답변 쌍 생성기 초기화"""
        self.ollama_url = ollama_url
        self.model = "mistral"
        
    def check_ollama_connection(self) -> bool:
        """Ollama 연결 확인"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def generate_questions_with_llm(self, metadata: List[Dict], questions_per_chunk: int = None) -> List[Dict]:
        """LLM을 사용하여 chunk당 질문-답변 쌍 생성 (동적 조정)"""
        
        # 문서 크기에 따라 questions_per_chunk 동적 조정
        if questions_per_chunk is None:
            if len(metadata) <= 10:
                questions_per_chunk = 5  # 작은 문서: 더 많은 QA
            elif len(metadata) <= 30:
                questions_per_chunk = 3  # 중간 문서: 기본값
            elif len(metadata) <= 100:
                questions_per_chunk = 2  # 큰 문서: 적은 QA
            else:
                questions_per_chunk = 1  # 매우 큰 문서: 최소 QA
        """LLM을 사용하여 chunk당 질문-답변 쌍 생성"""
        print(f"[INFO] LLM을 사용하여 chunk당 {questions_per_chunk}개의 질문-답변 쌍 생성 시작...")
        
        if not self.check_ollama_connection():
            print("[ERROR] Ollama 연결 실패")
            return self._generate_fallback_questions(metadata, len(metadata) * questions_per_chunk)
        
        # 각 chunk별로 질문-답변 쌍 생성
        all_qa_pairs = []
        
        for i, chunk in enumerate(metadata):
            print(f"[INFO] Chunk {i+1}/{len(metadata)} 처리 중... ({(i+1)/len(metadata)*100:.1f}%)")
            
            try:
                # 개별 chunk에 대한 질문 생성 (타임아웃 60초)
                chunk_qa_pairs = self._generate_qa_for_chunk(chunk, questions_per_chunk)
                print(f"[DEBUG] Chunk {i+1}에서 생성된 QA 쌍: {len(chunk_qa_pairs)}개")
                all_qa_pairs.extend(chunk_qa_pairs)
            except Exception as e:
                print(f"[ERROR] Chunk {i+1} 처리 실패: {e}")
                # 실패해도 폴백 QA 생성
                fallback_qa = self._generate_fallback_qa_for_chunk(chunk, questions_per_chunk)
                all_qa_pairs.extend(fallback_qa)
                print(f"[DEBUG] Chunk {i+1} 폴백 QA 쌍: {len(fallback_qa)}개")
            
            # 진행상황 업데이트
            current_count = len(all_qa_pairs)
            expected_count = len(metadata) * questions_per_chunk
            print(f"[PROGRESS] 현재 생성된 QA 쌍: {current_count}/{expected_count}개")
        
        try:
            # 모든 chunk 처리 완료 후 결과 반환
            print(f"[SUCCESS] 총 {len(all_qa_pairs)}개의 질문-답변 쌍 생성 완료")
            return all_qa_pairs
                
        except Exception as e:
            print(f"[ERROR] LLM 질문 생성 실패: {e}")
            return self._generate_fallback_questions(metadata, len(metadata) * questions_per_chunk)
    
    def _create_content_summary(self, metadata: List[Dict]) -> str:
        """문서 내용 요약 생성 (전처리 포함)"""
        summary_parts = []
        
        # 섹션별로 내용 요약
        sections = {}
        for item in metadata:
            section = item.get('section_title', '기타')
            if section not in sections:
                sections[section] = []
            # 전처리된 내용 사용
            content = preprocess_text(item.get('content', ''))
            sections[section].append(content[:200])  # 처음 200자만
        
        for section, contents in sections.items():
            summary_parts.append(f"## {section}")
            summary_parts.append(f"주요 내용: {' '.join(contents[:3])}")  # 섹션당 3개 내용만
        
        return "\n".join(summary_parts)
    
    def _generate_qa_for_chunk(self, chunk: Dict, questions_per_chunk: int) -> List[Dict]:
        """개별 chunk에 대한 질문-답변 쌍 생성"""
        try:
            # chunk 정보 추출
            content = chunk.get('content', '')
            section_title = chunk.get('section_title', '')
            keywords = chunk.get('keywords', [])
            summary = chunk.get('summary', '')
            
            # 청크 내용 전처리
            preprocessed_content = preprocess_text(content)
            
            # chunk별 질문 생성 프롬프트
            prompt = self._create_chunk_qa_prompt(preprocessed_content, section_title, keywords, summary, questions_per_chunk)
            
            # Ollama API 호출
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=1200
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('response', '')
                
                # 생성된 텍스트에서 질문-답변 쌍 파싱
                qa_pairs = self._parse_qa_pairs(generated_text)
                
                # chunk 정보 추가
                for qa in qa_pairs:
                    qa['chunk_id'] = chunk.get('chunk_id', '')
                    qa['section_title'] = section_title
                    qa['cluster_category'] = chunk.get('cluster_category', '')
                
                return qa_pairs
            else:
                print(f"[ERROR] Chunk 질문 생성 실패: {response.status_code}")
                return self._generate_fallback_qa_for_chunk(chunk, questions_per_chunk)
                
        except Exception as e:
            print(f"[ERROR] Chunk 질문 생성 중 오류: {e}")
            return self._generate_fallback_qa_for_chunk(chunk, questions_per_chunk)
    
    def _create_chunk_qa_prompt(self, content: str, section_title: str, keywords: List[str], summary: str, questions_per_chunk: int) -> str:
        """chunk별 질문 생성 프롬프트"""
        return f"""다음 의약품 관련 문서 청크를 바탕으로 정확히 {questions_per_chunk}개의 질문-답변 쌍을 생성해주세요.

섹션 제목: {section_title}
키워드: {', '.join(keywords)}
요약: {summary}
내용: {content[:800]}  # 처음 800자만

반드시 다음 JSON 형식으로만 응답해주세요. 다른 텍스트는 포함하지 마세요:

[
  {{
    "question": "질문 내용",
    "answer": "답변 내용", 
    "difficulty": "easy"
  }},
  {{
    "question": "질문 내용",
    "answer": "답변 내용",
    "difficulty": "medium"
  }},
  {{
    "question": "질문 내용", 
    "answer": "답변 내용",
    "difficulty": "hard"
  }}
]

의약품 도메인에 특화된 구체적인 질문을 생성해주세요. 반드시 한국어로 질문과 답변을 작성해주세요. JSON 형식만 출력하세요."""

    def _parse_qa_pairs(self, generated_text: str) -> List[Dict]:
        """생성된 텍스트에서 질문-답변 쌍 파싱"""
        try:
            print(f"[DEBUG] 파싱할 텍스트 길이: {len(generated_text)}")
            print(f"[DEBUG] 텍스트 미리보기: {generated_text[:200]}...")
            
            # JSON 배열 패턴: [ ... ]
            array_pattern = r'\[[^\]]*\]'
            matches = re.findall(array_pattern, generated_text, re.DOTALL)
            
            print(f"[DEBUG] JSON 배열 패턴 매치 수: {len(matches)}")
            
            if matches:
                # 가장 긴 JSON 배열 선택
                json_text = max(matches, key=len)
                print(f"[DEBUG] 선택된 JSON 텍스트 길이: {len(json_text)}")
                data = json.loads(json_text)
                
                # 각 항목이 올바른 형식인지 확인
                valid_qa_pairs = []
                for item in data:
                    if isinstance(item, dict) and 'question' in item and 'answer' in item:
                        valid_qa_pairs.append(item)
                
                print(f"[DEBUG] 유효한 QA 쌍 수: {len(valid_qa_pairs)}")
                return valid_qa_pairs
            
            # JSON 객체 패턴도 시도
            object_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            matches = re.findall(object_pattern, generated_text, re.DOTALL)
            
            if matches:
                json_text = max(matches, key=len)
                data = json.loads(json_text)
                
                if 'questions' in data:
                    return data['questions']
            
            # 실패 시 텍스트에서 직접 추출
            return self._extract_qa_from_text(generated_text)
            
        except Exception as e:
            print(f"[ERROR] JSON 파싱 실패: {e}")
            return self._extract_qa_from_text(generated_text)
    
    def _extract_qa_from_text(self, text: str) -> List[Dict]:
        """텍스트에서 질문-답변 쌍 직접 추출"""
        qa_pairs = []
        
        # 질문-답변 패턴 찾기
        lines = text.split('\n')
        current_question = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 질문 패턴 (숫자로 시작하거나 "Q:", "질문:" 등)
            if re.match(r'^\d+\.', line) or line.startswith('Q:') or line.startswith('질문:'):
                if current_question:
                    qa_pairs.append(current_question)
                current_question = {'question': line, 'answer': '', 'difficulty': 'medium'}
            
            # 답변 패턴 ("A:", "답변:" 등)
            elif line.startswith('A:') or line.startswith('답변:'):
                if current_question:
                    current_question['answer'] = line
            elif current_question and current_question['answer']:
                # 답변 계속
                current_question['answer'] += ' ' + line
        
        if current_question:
            qa_pairs.append(current_question)
        
        return qa_pairs
    
    def _generate_fallback_qa_for_chunk(self, chunk: Dict, questions_per_chunk: int) -> List[Dict]:
        """chunk별 폴백 질문 생성"""
        print(f"[INFO] Chunk {chunk.get('chunk_id', 'unknown')} 폴백 질문 생성 중...")
        
        section_title = chunk.get('section_title', '')
        keywords = chunk.get('keywords', [])
        
        # 기본 질문 템플릿
        question_templates = [
            f"{section_title}에서 {keywords[0] if keywords else '주요 내용'}은 무엇인가요?",
            f"{section_title}의 핵심 요구사항은 무엇인가요?",
            f"{section_title}에서 {keywords[1] if len(keywords) > 1 else '중요한 점'}은 어떻게 처리되나요?"
        ]
        
        questions = []
        for i in range(min(questions_per_chunk, len(question_templates))):
            questions.append({
                "question": question_templates[i],
                "answer": f"{section_title} 관련 내용입니다.",
                "difficulty": "medium",
                "chunk_id": chunk.get('chunk_id', ''),
                "section_title": section_title,
                "cluster_category": chunk.get('cluster_category', '')
            })
        
        return questions
    
    def _generate_fallback_questions(self, metadata: List[Dict], num_questions: int) -> List[Dict]:
        """LLM 실패 시 폴백 질문 생성"""
        print("[INFO] 폴백 질문 생성 중...")
        
        # 기본 질문 템플릿
        question_templates = {
            "임상시험": [
                "임상시험 {phase}상에서 {aspect}는 어떻게 진행되나요?",
                "임상시험 {phase}상의 주요 평가 항목은 무엇인가요?",
                "임상시험 {phase}상에서 {aspect} 기준은 무엇인가요?"
            ],
            "심사": [
                "의약품 심사 과정에서 {aspect}는 어떻게 처리되나요?",
                "심사부에서 {aspect}를 평가하는 기준은 무엇인가요?",
                "의약품 승인 시 {aspect} 관련 요구사항은 무엇인가요?"
            ],
            "가이드라인": [
                "{topic} 가이드라인의 핵심 내용은 무엇인가요?",
                "가이드라인에서 {aspect}에 대한 기준은 무엇인가요?",
                "{topic} 가이드라인 준수 여부는 어떻게 확인하나요?"
            ],
            "민원": [
                "민원인 {aspect} 절차는 어떻게 되나요?",
                "민원 처리 시 {aspect}는 어떻게 진행되나요?",
                "민원 관련 {aspect} 정보는 어디서 확인할 수 있나요?"
            ],
            "도메인용어": [
                "{term1}와 {term2}의 차이점은 무엇인가요?",
                "{term1}의 주요 특징은 무엇인가요?",
                "{term1}와 {term2}의 관계는 어떻게 되나요?"
            ]
        }
        
        # 키워드 추출
        keywords = []
        for item in metadata:
            item_keywords = item.get('keywords', [])
            keywords.extend(item_keywords)
        
        # 섹션 제목 추출
        sections = list(set([item.get('section_title', '') for item in metadata if item.get('section_title')]))
        
        questions = []
        categories = list(question_templates.keys())
        
        for i in range(num_questions):
            category = categories[i % len(categories)]
            template = question_templates[category][i % len(question_templates[category])]
            
            # 템플릿 변수 치환
            if "{phase}" in template:
                template = template.replace("{phase}", ["1", "2", "3"][i % 3])
            if "{aspect}" in template:
                template = template.replace("{aspect}", ["안전성", "유효성", "품질", "효과"][i % 4])
            if "{topic}" in template:
                template = template.replace("{topic}", ["임상시험", "심사", "평가"][i % 3])
            if "{term1}" in template and "{term2}" in template:
                if len(keywords) >= 2:
                    template = template.replace("{term1}", keywords[i % len(keywords)])
                    template = template.replace("{term2}", keywords[(i + 1) % len(keywords)])
            
            questions.append({
                "question": template,
                "category": "외재적" if i % 2 == 0 else "내재적",
                "subcategory": category,
                "expected_keywords": keywords[i % len(keywords):(i % len(keywords)) + 2] if keywords else []
            })
        
        return questions

    def extract_domain_keywords(self, text: str) -> List[str]:
        """LLM을 사용하여 도메인 특화 키워드 추출"""
        try:
            if not self.check_ollama_connection():
                print("[WARNING] Ollama 연결 실패, 기본 키워드 추출 사용")
                return self._extract_basic_keywords(text)
            
            # LLM 프롬프트 구성
            prompt = f"""
다음 텍스트에서 도메인 특화 키워드들을 추출해주세요.
텍스트의 주제와 관련된 전문적인 용어, 개념, 기술적 용어들을 찾아주세요.

텍스트: {text[:500]}

추출된 키워드들을 쉼표로 구분하여 리스트로 반환해주세요.
예시: 키워드1, 키워드2, 키워드3, 키워드4, 키워드5

키워드만 반환하고 다른 설명은 하지 마세요.
"""
            
            # Ollama API 호출 (타임아웃 10분으로 증가)
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "max_tokens": 100  # 토큰 수 줄임
                    }
                },
                timeout=600  # 타임아웃 10분으로 증가
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('response', '')
                
                # 응답에서 키워드 파싱
                keywords = self._parse_keywords_from_response(generated_text)
                
                if keywords:
                    return keywords[:8]  # 최대 8개 반환
            
        except Exception as e:
            print(f"[WARNING] LLM 키워드 추출 실패: {e}")
        
        # 실패 시 기본 키워드 추출
        return self._extract_basic_keywords(text)
    
    def _parse_keywords_from_response(self, response: str) -> List[str]:
        """LLM 응답에서 키워드 파싱"""
        try:
            # 응답에서 키워드 부분 추출
            lines = response.split('\n')
            for line in lines:
                if ',' in line and len(line.strip()) > 10:
                    # 쉼표로 구분된 키워드 추출
                    keywords = [kw.strip() for kw in line.split(',')]
                    # 의미있는 키워드만 필터링
                    meaningful_keywords = [kw for kw in keywords if len(kw) > 1 and not kw.isdigit()]
                    return meaningful_keywords
        
        except Exception as e:
            print(f"[WARNING] 키워드 파싱 실패: {e}")
        
        return []
    
    def _extract_basic_keywords(self, text: str) -> List[str]:
        """기본 키워드 추출 (LLM 실패 시 사용)"""
        import re
        from collections import Counter
        
        # 한글 단어 추출
        korean_words = re.findall(r'[가-힣]{2,}', text)
        word_freq = Counter(korean_words)
        
        # 빈도순으로 정렬하여 상위 키워드 반환
        top_keywords = [word for word, freq in word_freq.most_common(10) if freq > 1]
        
        return top_keywords

def main():
    if len(sys.argv) < 2:
        print("사용법: python qa_pair_generator.py <metadata_file> [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "qa_pairs.json"
    
    if not os.path.exists(input_file):
        print(f"[ERROR] 파일을 찾을 수 없습니다: {input_file}")
        sys.exit(1)
    
    # 메타데이터 로드 (파일 확장자에 따라)
    if input_file.endswith('.json'):
        with open(input_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    elif input_file.endswith('.pkl'):
        import pickle
        with open(input_file, 'rb') as f:
            metadata = pickle.load(f)
    else:
        print(f"[ERROR] 지원하지 않는 파일 형식입니다: {input_file}")
        sys.exit(1)
    
    # 질문-답변 쌍 생성 (chunk당 3개씩)
    generator = QAPairGenerator()
    qa_pairs = generator.generate_questions_with_llm(metadata, 3)
    
    # 결과 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "total_questions": len(qa_pairs),
            "generation_timestamp": str(np.datetime64('now')),
            "questions": qa_pairs
        }, f, ensure_ascii=False, indent=2)
    
    print(f"[SUCCESS] 질문-답변 쌍 생성 완료: {output_file}")
    print(f"- 총 질문 수: {len(qa_pairs)}")

if __name__ == "__main__":
    main() 