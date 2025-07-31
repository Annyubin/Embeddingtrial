#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
텍스트 전용 추출기
표와 표 내용을 완전히 제외하고 텍스트만 추출
"""

import fitz
import json
import sys
import re
from pathlib import Path
from typing import List

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
    
    return text

def extract_text_only(pdf_path):
    """표를 완전히 제외한 텍스트만 추출"""
    print(f"[INFO] 텍스트 전용 추출 시작: {pdf_path}")
    
    try:
        doc = fitz.open(pdf_path)
        text_chunks = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            print(f"[INFO] 페이지 {page_num + 1} 처리 중...")
            
            # 1. 표 영역 감지 (PyMuPDF + 추가 감지)
            table_areas = detect_all_table_areas(page)
            print(f"  [INFO] 감지된 표 영역: {len(table_areas)}개")
            
            # 2. 표 영역을 제외한 텍스트만 추출
            text_blocks = page.get_text("dict")["blocks"]
            text_only_blocks = filter_out_table_blocks(text_blocks, table_areas)
            
            # 3. 텍스트 추출
            page_text = extract_text_from_blocks(text_only_blocks)
            
            if page_text.strip():
                # 4. 청크 생성
                chunks = create_text_chunks(page_text, page_num + 1, page, text_only_blocks)
                text_chunks.extend(chunks)
                print(f"  [INFO] 텍스트 청크 {len(chunks)}개 생성")
            
        doc.close()
        
        # 결과 저장
        output_file = "text_only_chunks.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(text_chunks, f, ensure_ascii=False, indent=2)
        
        print(f"\n[SUCCESS] 텍스트 전용 추출 완료: {len(text_chunks)}개 청크")
        print(f"[INFO] 결과 저장: {output_file}")
        
        return text_chunks
        
    except Exception as e:
        print(f"[ERROR] PDF 처리 실패: {e}")
        return []

def detect_all_table_areas(page):
    """모든 표 영역을 감지"""
    table_areas = []
    
    try:
        # 1. PyMuPDF 표 감지
        table_finder = page.find_tables()
        tables = table_finder.tables
        
        for table in tables:
            table_areas.append(table.bbox)
        
        # 2. 텍스트 블록에서 표 형태 감지
        text_blocks = page.get_text("dict")["blocks"]
        for block in text_blocks:
            if "lines" in block and is_table_like_block(block):
                table_areas.append(block["bbox"])
        
        # 3. 중복 제거 및 병합
        table_areas = merge_overlapping_areas(table_areas)
        
    except Exception as e:
        print(f"  [WARNING] 표 영역 감지 실패: {e}")
    
    return table_areas

def is_table_like_block(block):
    """블록이 표 형태인지 판단"""
    if "lines" not in block:
        return False
    
    lines = block["lines"]
    if len(lines) < 2:
        return False
    
    # 각 줄의 텍스트 분석
    line_texts = []
    line_positions = []
    
    for line in lines:
        line_text = ""
        for span in line["spans"]:
            line_text += span["text"]
        
        if line_text.strip():
            line_texts.append(line_text.strip())
            line_positions.append(line["bbox"][1])
    
    if len(line_texts) < 2:
        return False
    
    # 표 형태 판단 기준
    # 1. 비슷한 길이의 줄들
    lengths = [len(text) for text in line_texts]
    avg_length = sum(lengths) / len(lengths)
    similar_lengths = sum(1 for length in lengths if abs(length - avg_length) < avg_length * 0.4)
    
    # 2. 일정한 간격
    if len(line_positions) > 1:
        intervals = [line_positions[i+1] - line_positions[i] for i in range(len(line_positions)-1)]
        avg_interval = sum(intervals) / len(intervals)
        regular_intervals = sum(1 for interval in intervals if abs(interval - avg_interval) < avg_interval * 0.6)
    else:
        regular_intervals = 0
    
    # 3. 특정 패턴 (숫자, 날짜, 구분자 등)
    has_table_patterns = any(
        any(char in text for char in ['│', '│', '─', '┌', '┐', '└', '┘', '├', '┤', '┬', '┴'])
        for text in line_texts
    )
    
    # 표로 판단하는 조건
    return (
        (similar_lengths >= len(lengths) * 0.6 and len(lengths) >= 3) or
        (regular_intervals >= len(intervals) * 0.6 and len(intervals) >= 2) or
        has_table_patterns
    )

def merge_overlapping_areas(areas, threshold=0.5):
    """겹치는 영역들을 병합"""
    if not areas:
        return areas
    
    merged = []
    used = [False] * len(areas)
    
    for i in range(len(areas)):
        if used[i]:
            continue
        
        current_area = areas[i]
        used[i] = True
        
        # 겹치는 영역들 찾기
        overlapping = [current_area]
        for j in range(i + 1, len(areas)):
            if not used[j] and bbox_overlap(current_area, areas[j], threshold):
                overlapping.append(areas[j])
                used[j] = True
        
        # 병합
        if len(overlapping) > 1:
            merged_area = merge_bboxes(overlapping)
            merged.append(merged_area)
        else:
            merged.append(current_area)
    
    return merged

def bbox_overlap(bbox1, bbox2, threshold=0.5):
    """두 BBOX가 겹치는지 확인"""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # 겹치는 영역 계산
    x_overlap = max(0, min(x2_1, x2_2) - max(x1_1, x1_2))
    y_overlap = max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
    
    overlap_area = x_overlap * y_overlap
    bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    
    if bbox1_area == 0:
        return False
    
    overlap_ratio = overlap_area / bbox1_area
    return overlap_ratio > threshold

def merge_bboxes(bboxes):
    """여러 BBOX를 하나로 병합"""
    if not bboxes:
        return None
    
    x1 = min(bbox[0] for bbox in bboxes)
    y1 = min(bbox[1] for bbox in bboxes)
    x2 = max(bbox[2] for bbox in bboxes)
    y2 = max(bbox[3] for bbox in bboxes)
    
    return [x1, y1, x2, y2]

def filter_out_table_blocks(text_blocks, table_areas):
    """표 영역을 제외한 텍스트 블록만 반환"""
    filtered_blocks = []
    
    for block in text_blocks:
        if "lines" not in block:
            continue
        
        block_bbox = block["bbox"]
        
        # 표 영역과 겹치는지 확인
        is_in_table = False
        for table_bbox in table_areas:
            if bbox_overlap(block_bbox, table_bbox, threshold=0.3):
                is_in_table = True
                break
        
        if not is_in_table:
            filtered_blocks.append(block)
    
    return filtered_blocks

def extract_text_from_blocks(blocks):
    """블록들에서 텍스트 추출 및 전처리"""
    text = ""
    
    for block in blocks:
        if "lines" in block:
            for line in block["lines"]:
                for span in line["spans"]:
                    text += span["text"] + " "
                text += "\n"
            text += "\n"
    
    # 전처리 적용
    cleaned_text = preprocess_text(text.strip())
    return cleaned_text

def create_text_chunks(text, page_num, page, blocks, max_length=500, overlap_ratio=0.5):
    """텍스트를 청크로 분할 (overlap 포함)"""
    chunks = []
    
    current_chunk = ""
    chunk_id = 1
    current_y_position = 0
    overlap_text = ""  # 이전 청크의 마지막 부분을 저장
    
    for i, block in enumerate(blocks):
        if "lines" not in block:
            continue
        
        block_text = ""
        for line in block["lines"]:
            for span in line["spans"]:
                block_text += span["text"] + " "
            block_text += "\n"
        
        # 현재 청크에 블록을 추가했을 때 길이 확인
        if len(current_chunk + block_text) <= max_length:
            current_chunk += block_text
            # Y 위치는 첫 번째 블록의 위치 사용
            if current_y_position == 0:
                current_y_position = block["bbox"][1]
        else:
            # 현재 청크가 있으면 저장 (전처리 적용)
            if current_chunk.strip():
                cleaned_chunk = preprocess_text(current_chunk.strip())
                chunks.append({
                    'chunk_id': f"text_only_{page_num}_{chunk_id:03d}",
                    'content': cleaned_chunk,
                    'page': page_num,
                    'content_length': len(cleaned_chunk),
                    'chunk_type': 'text_only',
                    'y_position': current_y_position,
                    'overlap_ratio': overlap_ratio
                })
                chunk_id += 1
            
            # overlap 텍스트 생성 (이전 청크의 마지막 부분)
            overlap_text = create_overlap_text(current_chunk, overlap_ratio)
            
            # 새 청크 시작 (overlap 텍스트 포함)
            current_chunk = overlap_text + block_text
            current_y_position = block["bbox"][1]
    
    # 마지막 청크 처리 (전처리 적용)
    if current_chunk.strip():
        cleaned_chunk = preprocess_text(current_chunk.strip())
        chunks.append({
            'chunk_id': f"text_only_{page_num}_{chunk_id:03d}",
            'content': cleaned_chunk,
            'page': page_num,
            'content_length': len(cleaned_chunk),
            'chunk_type': 'text_only',
            'y_position': current_y_position,
            'overlap_ratio': overlap_ratio
        })
    
    return chunks

def create_overlap_text(text: str, overlap_ratio: float = 0.5) -> str:
    """이전 청크의 마지막 부분을 overlap 텍스트로 생성"""
    if not text.strip():
        return ""
    
    # 문장 단위로 분할
    sentences = split_into_sentences(text)
    
    if not sentences:
        return ""
    
    # overlap_ratio에 따라 마지막 문장들 선택
    overlap_count = max(1, int(len(sentences) * overlap_ratio))
    overlap_sentences = sentences[-overlap_count:]
    
    # overlap 텍스트 생성
    overlap_text = " ".join(overlap_sentences)
    
    # 최대 길이 제한 (전체 텍스트의 50% 이하)
    max_overlap_length = len(text) * 0.5
    if len(overlap_text) > max_overlap_length:
        # 단어 단위로 자르기
        words = overlap_text.split()
        overlap_text = " ".join(words[:int(max_overlap_length / 5)])  # 평균 단어 길이 5자 가정
    
    return overlap_text + "\n\n"

def split_into_sentences(text: str) -> List[str]:
    """텍스트를 문장 단위로 분할"""
    # 한국어 문장 구분 패턴
    sentence_pattern = r'[^.!?。]*[.!?。]'
    sentences = re.findall(sentence_pattern, text)
    
    # 패턴에 매치되지 않은 부분도 추가
    remaining = re.sub(sentence_pattern, '', text)
    if remaining.strip():
        sentences.append(remaining.strip())
    
    # 빈 문장 제거 및 정리
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python text_only_extractor.py <PDF파일경로>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    if not Path(pdf_path).exists():
        print(f"[ERROR] 파일을 찾을 수 없습니다: {pdf_path}")
        sys.exit(1)
    
    extract_text_only(pdf_path) 