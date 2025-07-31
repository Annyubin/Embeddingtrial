#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
동적 키워드 생성기
LLM을 사용하여 문서에서 키워드를 동적으로 추출
"""

import json
import requests
from typing import List, Dict, Any
import re

class DynamicKeywordGenerator:
    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "mistral"):
        """동적 키워드 생성기 초기화"""
        self.ollama_url = ollama_url
        self.model = model
    
    def generate_keywords_from_text(self, text: str, max_keywords: int = 10) -> List[str]:
        """텍스트에서 키워드 동적 생성"""
        prompt = f"""다음 텍스트에서 가장 중요한 키워드 {max_keywords}개를 추출해주세요.

텍스트: {text[:500]}

다음 JSON 형식으로만 응답해주세요:
{{
    "keywords": ["키워드1", "키워드2", "키워드3", ...]
}}

반드시 한국어로 키워드를 작성해주세요."""

        try:
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
                
                # JSON 파싱
                try:
                    import re
                    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
                    matches = re.findall(json_pattern, response_text)
                    
                    if matches:
                        json_text = max(matches, key=len)
                        data = json.loads(json_text)
                        return data.get('keywords', [])
                    else:
                        # JSON 파싱 실패 시 텍스트에서 직접 추출
                        return self._extract_keywords_fallback(response_text)
                        
                except json.JSONDecodeError:
                    return self._extract_keywords_fallback(response_text)
            else:
                return self._extract_keywords_fallback(text)
                
        except Exception as e:
            print(f"[ERROR] 키워드 생성 실패: {e}")
            return self._extract_keywords_fallback(text)
    
    def _extract_keywords_fallback(self, text: str) -> List[str]:
        """키워드 추출 실패 시 대체 방법"""
        import re
        
        # 기본 키워드 추출
        korean_keywords = re.findall(r'[가-힣]{2,}', text)
        english_keywords = re.findall(r'[a-zA-Z]{3,}', text.lower())
        
        # 의미있는 단어만 필터링
        meaningful_korean = [word for word in korean_keywords if len(word) >= 2]
        meaningful_english = [word for word in english_keywords if word not in ['the', 'and', 'for', 'are', 'was', 'were']]
        
        return meaningful_korean[:5] + meaningful_english[:3]
    
    def generate_qa_keywords(self, question: str, answer: str) -> Dict[str, List[str]]:
        """질문과 답변에서 키워드 생성"""
        question_keywords = self.generate_keywords_from_text(question, 5)
        answer_keywords = self.generate_keywords_from_text(answer, 8)
        
        return {
            'question_keywords': question_keywords,
            'answer_keywords': answer_keywords,
            'combined_keywords': list(set(question_keywords + answer_keywords))
        }

def main():
    """테스트 함수"""
    generator = DynamicKeywordGenerator()
    
    # 테스트 텍스트
    test_text = "임상시험 1상에서는 주로 안전성을 평가합니다. 약물의 독성과 부작용을 확인하고, 적절한 용량을 결정합니다."
    
    keywords = generator.generate_keywords_from_text(test_text)
    print(f"생성된 키워드: {keywords}")
    
    # QA 키워드 테스트
    qa_keywords = generator.generate_qa_keywords(
        "임상시험 1상에서 무엇을 평가하나요?",
        "안전성과 독성을 평가합니다."
    )
    print(f"QA 키워드: {qa_keywords}")

if __name__ == "__main__":
    main() 