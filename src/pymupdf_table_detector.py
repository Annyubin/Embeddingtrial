#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyMuPDF 표 영역 감지기
표의 위치와 영역 정보만 추출 (내용 제외)
"""

import fitz
import json
import sys
from pathlib import Path

def detect_table_areas(pdf_path):
    """PDF에서 표 영역만 감지 (내용 제외)"""
    print(f"[INFO] PyMuPDF 표 영역 감지 시작: {pdf_path}")
    
    try:
        doc = fitz.open(pdf_path)
        table_areas = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            try:
                # PyMuPDF로 표 감지
                table_finder = page.find_tables()
                tables = table_finder.tables
                
                for i, table in enumerate(tables):
                    table_info = {
                        'page': page_num + 1,
                        'table_index': i,
                        'bbox': list(table.bbox),  # (x0, y0, x1, y1)
                        'rows': len(table.rows),
                        'detection_method': 'pymupdf_find_tables',
                        'content': None  # 표 내용은 제외
                    }
                    
                    table_areas.append(table_info)
                
                if tables:
                    print(f"  [PyMuPDF] 페이지 {page_num + 1}: {len(tables)}개 표 영역 감지")
                    
            except Exception as e:
                print(f"  [ERROR] 페이지 {page_num + 1} 표 감지 실패: {e}")
        
        doc.close()
        
        # 결과 저장
        output_file = "pymupdf_table_areas.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(table_areas, f, ensure_ascii=False, indent=2)
        
        print(f"\n[SUCCESS] 표 영역 감지 완료: {len(table_areas)}개")
        print(f"[INFO] 결과 저장: {output_file}")
        
        return table_areas
        
    except Exception as e:
        print(f"[ERROR] PDF 처리 실패: {e}")
        return []

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python pymupdf_table_detector.py <PDF파일경로>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    if not Path(pdf_path).exists():
        print(f"[ERROR] 파일을 찾을 수 없습니다: {pdf_path}")
        sys.exit(1)
    
    detect_table_areas(pdf_path) 