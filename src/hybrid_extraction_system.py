#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
하이브리드 추출 시스템 (텍스트 전용)
PyMuPDF로 표 영역 감지 + 텍스트만 추출 + LLM으로 섹션 제목/키워드 추출
표 내용 추출 제외
"""

import json
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Tuple
from text_preprocessor import TextPreprocessor

# 한글 인코딩 설정
import locale
import codecs

# Windows 환경에서 한글 출력을 위한 설정
if sys.platform.startswith('win'):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

class HybridExtractionSystem:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.pymupdf_tables = []
        self.text_chunks = []
        self.integrated_chunks = []
        
    def run_pymupdf_detection(self) -> List[Dict]:
        """PyMuPDF로 표 영역 감지 (venv_text_new 환경)"""
        print("[INFO] PyMuPDF 표 영역 감지 시작...")
        
        try:
            # venv_text_new 환경에서 PyMuPDF 실행
            if sys.platform.startswith('win'):
                venv_python = "environments/venv_text_new/Scripts/python.exe"
            else:
                venv_python = "environments/venv_text_new/bin/python"
            script_path = "src/pymupdf_table_detector.py"
            
            cmd = [venv_python, script_path, self.pdf_path]
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            
            if result.returncode == 0:
                # 결과 파일 읽기
                with open("pymupdf_table_areas.json", "r", encoding="utf-8") as f:
                    self.pymupdf_tables = json.load(f)
                print(f"[SUCCESS] PyMuPDF 표 영역 감지 완료: {len(self.pymupdf_tables)}개")
                return self.pymupdf_tables
            else:
                print(f"[ERROR] PyMuPDF 실행 실패: {result.stderr}")
                return []
                
        except Exception as e:
            print(f"[ERROR] PyMuPDF 표 감지 오류: {e}")
            return []
    
    def extract_semantic_chunks(self) -> List[Dict]:
        """의미 기반 청크 추출 (표 제외)"""
        print("[INFO] 의미 기반 청크 추출 시작...")
        try:
            # venv_text_new 환경에서 의미 기반 청크 생성기 실행
            if sys.platform.startswith('win'):
                venv_python = "environments/venv_text_new/Scripts/python.exe"
            else:
                venv_python = "environments/venv_text_new/bin/python"
            script_path = "src/semantic_chunk_generator.py"
            
            cmd = [venv_python, script_path, self.pdf_path]
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            
            if result.returncode == 0:
                # 결과 파일 읽기
                with open("semantic_chunks.json", "r", encoding="utf-8") as f:
                    self.text_chunks = json.load(f)
                print(f"[SUCCESS] 의미 기반 청크 추출 완료: {len(self.text_chunks)}개 청크")
                return self.text_chunks
            else:
                print(f"[ERROR] 의미 기반 청크 추출 실패: {result.stderr}")
                return []
                
        except Exception as e:
            print(f"[ERROR] 의미 기반 청크 추출 오류: {e}")
            return []
    
    def run_llm_enhancement(self) -> Dict:
        """LLM으로 섹션 제목과 키워드 추출 (venv_rag_new 환경)"""
        print("[INFO] LLM 섹션 제목/키워드 추출 시작...")
        try:
            # venv_rag_new 환경에서 LLM 실행
            if sys.platform.startswith('win'):
                venv_python = "environments/venv_rag_new/Scripts/python.exe"
            else:
                venv_python = "environments/venv_rag_new/bin/python"
            script_path = "src/llm_section_keyword_extractor.py"
            
            cmd = [venv_python, script_path, "semantic_chunks.json"]
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            
            if result.returncode == 0:
                # 결과 파일 읽기
                with open("data/llm_enhanced_sections.json", "r", encoding="utf-8") as f:
                    llm_results = json.load(f)
                print(f"[SUCCESS] LLM 섹션 제목/키워드 추출 완료")
                return llm_results
            else:
                print(f"[ERROR] LLM 실행 실패: {result.stderr}")
                return {}
                
        except Exception as e:
            print(f"[ERROR] LLM 섹션 제목/키워드 추출 오류: {e}")
            return {}
    
    def integrate_results(self) -> List[Dict]:
        """결과 통합 (텍스트 전용)"""
        print("[INFO] 결과 통합 시작...")
        
        integrated_chunks = []
        
        # 텍스트 청크들을 통합
        for chunk in self.text_chunks:
            integrated_chunk = {
                'chunk_id': chunk['chunk_id'],
                'content': chunk['content'],
                'page': chunk.get('page', 0),  # page 키가 없을 경우 기본값 0
                'content_length': chunk['content_length'],
                'chunk_type': 'text_only_integrated',
                'y_position': chunk.get('y_position', 0),
                'section_title': None,
                'keywords': [],
                'confidence': 0.0
            }
            integrated_chunks.append(integrated_chunk)
        
        self.integrated_chunks = integrated_chunks
        print(f"[SUCCESS] 결과 통합 완료: {len(integrated_chunks)}개 청크")
        
        return integrated_chunks
    
    def apply_llm_enhancements(self, integrated_chunks: List[Dict]) -> List[Dict]:
        """LLM 강화 결과를 적용"""
        print("[INFO] LLM 강화 결과 적용 중...")
        
        # LLM 결과 로드
        try:
            with open("data/llm_enhanced_sections.json", "r", encoding="utf-8") as f:
                llm_results = json.load(f)
        except Exception as e:
            print(f"[WARNING] LLM 결과 로드 실패: {e}")
            return integrated_chunks
        
        # LLM 결과를 청크 ID로 매핑
        llm_map = {}
        for result in llm_results:
            chunk_id = result.get('chunk_id')
            if chunk_id:
                llm_map[chunk_id] = result
        
        # 통합된 청크에 LLM 결과 적용
        enhanced_chunks = []
        for chunk in integrated_chunks:
            chunk_id = chunk.get('chunk_id')
            llm_result = llm_map.get(chunk_id, {})
                
            # LLM 결과 적용
            enhanced_chunk = chunk.copy()
            enhanced_chunk['section_title'] = llm_result.get('section_title', chunk.get('section_title'))
            enhanced_chunk['keywords'] = llm_result.get('keywords', chunk.get('keywords', []))
            enhanced_chunk['confidence'] = llm_result.get('confidence', chunk.get('confidence', 0.0))
            enhanced_chunk['summary'] = llm_result.get('summary', chunk.get('summary', ''))
            
            enhanced_chunks.append(enhanced_chunk)
        
        print(f"[SUCCESS] LLM 강화 결과 적용 완료: {len(enhanced_chunks)}개 청크")
        return enhanced_chunks
    
    def preprocess_text(self):
        """텍스트 전처리"""
        print("[INFO] 텍스트 전처리 시작...")
        
        preprocessor = TextPreprocessor()
        
        for chunk in self.integrated_chunks:
            if 'content' in chunk:
                # 텍스트 정리
                cleaned_text = preprocessor.clean_text(chunk['content'])
                chunk['content'] = cleaned_text
                chunk['content_length'] = len(cleaned_text)
        
        print("[SUCCESS] 텍스트 전처리 완료")
    
    def save_results(self, final_chunks: List[Dict]):
        """최종 결과 저장"""
        print("[INFO] 최종 결과 저장 중...")
        
        # 최종 결과 저장
        with open("final_text_only_data.json", "w", encoding="utf-8") as f:
            json.dump(final_chunks, f, ensure_ascii=False, indent=2)
        
        # 통계 정보
        total_chunks = len(final_chunks)
        total_length = sum(chunk.get('content_length', 0) for chunk in final_chunks)
        chunks_with_section = sum(1 for chunk in final_chunks if chunk.get('section_title'))
        
        print(f"[SUCCESS] 최종 결과 저장 완료")
        print(f"- 총 청크 수: {total_chunks}")
        print(f"- 총 텍스트 길이: {total_length:,}자")
        print(f"- 섹션 제목 포함: {chunks_with_section}개")
        print(f"- 파일: final_text_only_data.json")

    def run_full_extraction(self) -> Dict:
        """전체 추출 프로세스 실행 (의미 기반 청크 추출 + LLM 강화 + 임베딩 생성)"""
        print("=" * 60)
        print("전체 추출 프로세스 시작")
        print("=" * 60)
        
        # 1. 의미 기반 청크 추출
        print("[STEP 1] 의미 기반 청크 추출")
        text_chunks = self.extract_semantic_chunks()
        
        # 2. LLM 강화
        print("[STEP 2] LLM 섹션 제목/키워드 추출")
        llm_results = self.run_llm_enhancement()
        
        # 3. 결과 통합
        print("[STEP 3] 결과 통합")
        integrated_chunks = self.integrate_results()
        
        # 4. LLM 강화 적용
        print("[STEP 4] LLM 강화 적용")
        final_chunks = self.apply_llm_enhancements(integrated_chunks)
        
        # 5. 최종 결과 저장
        print("[STEP 5] 최종 결과 저장")
        self.save_results(final_chunks)
        
        # 6. 임베딩 생성
        print("[STEP 6] 임베딩 생성")
        self.generate_embeddings()
        
        print("=" * 60)
        print("전체 추출 프로세스 완료!")
        print("=" * 60)
        
        return {
            'text_chunks': len(text_chunks),
            'integrated_chunks': len(integrated_chunks),
            'final_chunks': len(final_chunks),
            'embedding_generated': True
        }

    def generate_embeddings(self):
        """임베딩 생성"""
        print("[INFO] 임베딩 생성 시작...")
        try:
            # venv_rag_new 환경에서 임베딩 생성기 실행
            if sys.platform.startswith('win'):
                venv_python = "environments/venv_rag_new/Scripts/python.exe"
            else:
                venv_python = "environments/venv_rag_new/bin/python"
            script_path = "src/embedding_generator.py"
            
            cmd = [venv_python, script_path, "final_text_only_data.json", "embeddings"]
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            
            if result.returncode == 0:
                print("[SUCCESS] 임베딩 생성 완료")
                return True
            else:
                print(f"[ERROR] 임베딩 생성 실패: {result.stderr}")
                return False
        except Exception as e:
            print(f"[ERROR] 임베딩 생성 오류: {e}")
            return False

def main():
    if len(sys.argv) < 2:
        print("사용법: python hybrid_extraction_system.py <PDF파일경로>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    if not Path(pdf_path).exists():
        print(f"[ERROR] 파일을 찾을 수 없습니다: {pdf_path}")
        sys.exit(1)
    
    # 하이브리드 추출 시스템 실행
    system = HybridExtractionSystem(pdf_path)
    system.run_full_extraction()

if __name__ == "__main__":
    main() 