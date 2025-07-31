#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
텍스트 전처리기
PDF에서 추출된 텍스트의 개행문자, 공백, 특수문자 등을 정리
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Any

class TextPreprocessor:
    def __init__(self):
        self.whitespace_pattern = re.compile(r'\s+')
        self.newline_pattern = re.compile(r'\n+')
        self.multiple_spaces = re.compile(r' +')
        
    def clean_text(self, text) -> str:
        """텍스트 정리"""
        if not text:
            return ""
        
        # 문자열이 아닌 경우 변환
        if not isinstance(text, str):
            text = str(text)
        
        # 1. 개행문자 정리
        text = self.newline_pattern.sub(' ', text)
        
        # 2. 연속된 공백 정리
        text = self.multiple_spaces.sub(' ', text)
        
        # 3. 앞뒤 공백 제거
        text = text.strip()
        
        # 4. 특수 공백 문자 정리
        text = text.replace('\u3000', ' ')  # 전각 공백
        text = text.replace('\u00a0', ' ')  # non-breaking space
        text = text.replace('\u200b', '')   # zero-width space
        
        # 5. 연속된 공백 다시 정리
        text = self.multiple_spaces.sub(' ', text)
        
        return text
    
    def clean_table_text(self, text) -> str:
        """표 텍스트 정리 (표 형식 유지)"""
        if not text:
            return ""
        
        # 리스트인 경우 문자열로 변환
        if isinstance(text, list):
            text = ' '.join(str(item) for item in text)
        elif not isinstance(text, str):
            text = str(text)
        
        # 1. 개행문자를 공백으로 변경 (표 셀 구분 유지)
        text = text.replace('\n', ' ')
        
        # 2. 연속된 공백 정리
        text = self.multiple_spaces.sub(' ', text)
        
        # 3. 앞뒤 공백 제거
        text = text.strip()
        
        # 4. 특수 공백 문자 정리
        text = text.replace('\u3000', ' ')  # 전각 공백
        text = text.replace('\u00a0', ' ')  # non-breaking space
        text = text.replace('\u200b', '')   # zero-width space
        
        return text
    
    def process_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """청크 리스트 전체 전처리"""
        processed_chunks = []
        
        for chunk in chunks:
            processed_chunk = chunk.copy()
            
            # 텍스트 타입에 따라 다른 전처리 적용
            if chunk.get('metadata', {}).get('type') == 'table':
                processed_chunk['text'] = self.clean_table_text(chunk['text'])
            else:
                processed_chunk['text'] = self.clean_text(chunk['text'])
            
            # 빈 내용 필터링
            if self._is_empty_or_meaningless(processed_chunk['text']):
                print(f"[FILTER] 빈 내용 청크 제거: {chunk.get('id', 'unknown')} - '{processed_chunk['text'][:50]}...'")
                continue
            
            # content_length 업데이트
            if 'metadata' in processed_chunk:
                processed_chunk['metadata']['content_length'] = len(processed_chunk['text'])
                processed_chunk['metadata']['preprocessed'] = True
            
            processed_chunks.append(processed_chunk)
        
        print(f"[INFO] 빈 내용 필터링 완료: {len(chunks)}개 → {len(processed_chunks)}개")
        return processed_chunks
    
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
    
    def process_json_file(self, input_file: str, output_file: str = None) -> str:
        """JSON 파일 전처리"""
        if output_file is None:
            input_path = Path(input_file)
            output_file = str(input_path.parent / f"{input_path.stem}_preprocessed{input_path.suffix}")
        
        print(f"[INFO] 텍스트 전처리 시작: {input_file}")
        
        # 파일 읽기
        with open(input_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        # 전처리
        processed_chunks = self.process_chunks(chunks)
        
        # 파일 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_chunks, f, ensure_ascii=False, indent=2)
        
        print(f"[SUCCESS] 전처리 완료: {output_file}")
        print(f"  - 원본 청크: {len(chunks)}개")
        print(f"  - 전처리 청크: {len(processed_chunks)}개")
        
        return output_file

def main():
    if len(sys.argv) < 2:
        print("사용법: python text_preprocessor.py <입력JSON파일> [출력JSON파일]")
        sys.exit(1)
    
    preprocessor = TextPreprocessor()
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if Path(input_file).exists():
        preprocessor.process_json_file(input_file, output_file)
    else:
        print(f"[ERROR] 파일을 찾을 수 없습니다: {input_file}")
        sys.exit(1)

if __name__ == "__main__":
    main() 