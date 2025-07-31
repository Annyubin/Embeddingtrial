#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit 통합 스크립트
새로운 구조화된 결과를 기존 Streamlit 앱과 호환되도록 변환
"""

import json
import os
import sys
from typing import Dict, List, Any

def convert_structured_to_streamlit_format(structured_file: str, output_file: str = "final_text_only_data.json"):
    """
    새로운 구조화된 결과를 Streamlit 앱이 기대하는 형식으로 변환
    
    Args:
        structured_file: 새로운 구조화된 결과 파일 (full_pipeline_llm_enhanced_chunks.json)
        output_file: Streamlit 앱이 기대하는 출력 파일명
    """
    print(f"[INFO] 구조화된 결과를 Streamlit 형식으로 변환 중...")
    print(f"[INFO] 입력 파일: {structured_file}")
    print(f"[INFO] 출력 파일: {output_file}")
    
    try:
        # 새로운 구조화된 결과 읽기
        with open(structured_file, 'r', encoding='utf-8') as f:
            structured_data = json.load(f)
        
        print(f"[INFO] 구조화된 데이터 로드 완료: {len(structured_data)}개 청크")
        
        # Streamlit 앱이 기대하는 형식으로 변환
        streamlit_format = []
        
        for i, chunk in enumerate(structured_data):
            # 기존 Streamlit 앱이 기대하는 형식
            streamlit_chunk = {
                "id": chunk.get("chunk_id", f"chunk_{i+1:03d}"),
                "content": chunk.get("content", ""),
                "page": chunk.get("page", 0),
                "y_position": chunk.get("y_position", 0),
                "content_length": chunk.get("content_length", 0),
                "source_blocks": chunk.get("source_blocks", []),
                "overlap_ratio": chunk.get("overlap_ratio", 0.3),
                
                # LLM 메타데이터
                "section_title": chunk.get("section_title", ""),
                "keywords": chunk.get("keywords", []),
                "summary": chunk.get("summary", ""),
                "confidence": chunk.get("confidence", 0.5),
                "content_type": chunk.get("content_type", "text"),
                "document_section": chunk.get("document_section", ""),
                "extraction_method": chunk.get("extraction_method", "llm_ollama"),
                
                # 추가 메타데이터
                "metadata": {
                    "section_title": chunk.get("section_title", ""),
                    "keywords": chunk.get("keywords", []),
                    "summary": chunk.get("summary", ""),
                    "confidence": chunk.get("confidence", 0.5),
                    "content_type": chunk.get("content_type", "text"),
                    "document_section": chunk.get("document_section", ""),
                    "extraction_method": chunk.get("extraction_method", "llm_ollama"),
                    "overlap_ratio": chunk.get("overlap_ratio", 0.3),
                    "source_blocks": chunk.get("source_blocks", [])
                }
            }
            
            streamlit_format.append(streamlit_chunk)
        
        # Streamlit 앱이 기대하는 형식으로 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(streamlit_format, f, ensure_ascii=False, indent=2)
        
        print(f"[SUCCESS] 변환 완료: {len(streamlit_format)}개 청크")
        print(f"[INFO] 저장된 파일: {output_file}")
        
        # 통계 정보 출력
        total_content_length = sum(chunk.get("content_length", 0) for chunk in streamlit_format)
        avg_content_length = total_content_length / len(streamlit_format) if streamlit_format else 0
        
        print(f"[INFO] 통계:")
        print(f"  - 전체 청크 수: {len(streamlit_format)}개")
        print(f"  - 총 텍스트 길이: {total_content_length:,}자")
        print(f"  - 평균 청크 길이: {avg_content_length:.1f}자")
        print(f"  - LLM 처리된 청크: {sum(1 for c in streamlit_format if c.get('extraction_method') == 'llm_ollama')}개")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] 변환 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_streamlit_ready_pipeline():
    """
    Streamlit에서 사용할 수 있는 완전한 파이프라인 생성
    """
    print("[INFO] Streamlit용 완전 파이프라인 생성 중...")
    
    # 1. 새로운 구조화된 파이프라인 실행
    print("[STEP 1] 새로운 구조화된 파이프라인 실행...")
    try:
        import subprocess
        cmd = ["python", "test_full_pipeline_with_llm.py"]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        
        if result.returncode == 0:
            print("[SUCCESS] 새로운 구조화된 파이프라인 완료!")
        else:
            print(f"[ERROR] 새로운 파이프라인 실패: {result.stderr}")
            return False
    except Exception as e:
        print(f"[ERROR] 파이프라인 실행 오류: {e}")
        return False
    
    # 2. Streamlit 형식으로 변환
    print("[STEP 2] Streamlit 형식으로 변환...")
    if convert_structured_to_streamlit_format("full_pipeline_llm_enhanced_chunks.json"):
        print("[SUCCESS] Streamlit 형식 변환 완료!")
    else:
        print("[ERROR] Streamlit 형식 변환 실패!")
        return False
    
    # 3. 임베딩 생성 (기존 시스템 사용)
    print("[STEP 3] 임베딩 생성...")
    try:
        cmd = ["python", "src/embedding_generator.py"]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        
        if result.returncode == 0:
            print("[SUCCESS] 임베딩 생성 완료!")
        else:
            print(f"[ERROR] 임베딩 생성 실패: {result.stderr}")
            return False
    except Exception as e:
        print(f"[ERROR] 임베딩 생성 오류: {e}")
        return False
    
    print("[SUCCESS] Streamlit용 완전 파이프라인 완료!")
    return True

def main():
    """메인 함수"""
    print("=" * 60)
    print("Streamlit 통합 스크립트")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "convert":
            # 기존 구조화된 결과를 Streamlit 형식으로 변환
            input_file = sys.argv[2] if len(sys.argv) > 2 else "full_pipeline_llm_enhanced_chunks.json"
            convert_structured_to_streamlit_format(input_file)
            
        elif command == "pipeline":
            # 완전한 파이프라인 실행
            create_streamlit_ready_pipeline()
            
        else:
            print("사용법:")
            print("  python src/streamlit_integration.py convert [입력파일]")
            print("  python src/streamlit_integration.py pipeline")
    else:
        print("사용법:")
        print("  python src/streamlit_integration.py convert [입력파일]")
        print("  python src/streamlit_integration.py pipeline")

if __name__ == "__main__":
    main() 