#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
의미 기반 청크 생성기
QA 의미 기반 클러스터링으로 먼저 그룹화하고, 클러스터 내에서 500자 기준 후분할
"""

import json
import fitz
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple
import re
from pathlib import Path

class SemanticChunkGenerator:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        """의미 기반 청크 생성기 초기화"""
        # 모델 로딩을 지연시킴 (필요할 때만 로드)
        self.model_name = model_name
        self.model = None
        
        # 일반적인 문서 섹션 카테고리 (하드코딩된 도메인 키워드 제거)
        self.general_categories = [
            '서론', '배경', '목적', '범위', '정의', '절차', '방법', 
            '결과', '결론', '참고', '부록', '기타'
        ]
    
    def _load_model(self):
        """모델을 필요할 때 로드"""
        if self.model is None:
            print("[INFO] SentenceTransformer 모델 로딩 중...")
            try:
                # CPU 모드로 모델 초기화 (CUDA 호환성 문제 해결)
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name, device='cpu')
                print("[SUCCESS] 모델 로딩 완료")
            except Exception as e:
                print(f"[ERROR] 모델 로딩 실패: {e}")
                self.model = None

    def _preprocess_text(self, text: str) -> str:
        """텍스트 전처리 (완화된 버전)"""
        if not text:
            return ""
        
        # 1. 기본 정리
        text = text.strip()
        
        # 2. 연속된 개행을 하나로
        text = re.sub(r'\n{2,}', '\n', text)
        
        # 3. 문장 끝 정리
        text = re.sub(r'\n+$', '', text)  # 끝의 개행 제거
        text = re.sub(r'^\n+', '', text)  # 시작의 개행 제거
        
        # 4. 연속된 공백 정리
        text = re.sub(r' +', ' ', text)
        
        # 5. 특수 유니코드 문자 제거 (Ÿ 등)
        text = re.sub(r'[^\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F\w\s\.\,\;\:\!\?\(\)\[\]\{\}\-\+\=\*\/\@\#\$\%\&\*\(\)]', '', text)
        
        # 6. 추가 정리: 불필요한 개행 제거
        text = re.sub(r'\n\s*\n', '\n', text)  # 빈 줄 제거
        text = re.sub(r'\n\s+', '\n', text)    # 개행 후 공백 제거
        text = re.sub(r'\s+\n', '\n', text)    # 개행 전 공백 제거
        
        # 7. 체크박스나 라디오 버튼이 많은 텍스트 제거 (완화된 기준)
        checkbox_count = text.count('□') + text.count('■') + text.count('☐') + text.count('☑')
        if checkbox_count > 10:  # 3개 → 10개로 완화
            return ""  # 표/폼 내용은 빈 문자열로 반환
        
        # 8. "예 아니오" 패턴이 많은 텍스트 제거 (완화된 기준)
        yes_no_count = text.count('예') + text.count('아니오')
        if yes_no_count > 15:  # 5개 → 15개로 완화
            return ""  # 체크박스 폼 내용은 제거
        
        # 9. 문장 단위로 정리
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line:  # 빈 줄 제외
                cleaned_lines.append(line)
        
        # 10. 최종 조합
        text = '\n'.join(cleaned_lines)
        
        # 11. 마지막 정리
        text = text.strip()
        
        return text

    def extract_text_blocks(self, pdf_path: str) -> List[Dict]:
        """PDF에서 텍스트 블록 추출 (표 제외)"""
        print(f"[INFO] PDF 텍스트 블록 추출: {pdf_path}")
        
        doc = fitz.open(pdf_path)
        all_blocks = []
        
        print(f"[INFO] 총 페이지 수: {len(doc)}")
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            print(f"[INFO] 페이지 {page_num + 1}/{len(doc)} 처리 중...")
            
            try:
                # 표 영역 감지
                print(f"  [DEBUG] 표 영역 감지 시작...")
                table_areas = self._detect_table_areas(page)
                print(f"  [DEBUG] 감지된 표 영역: {len(table_areas)}개")
                
                # 텍스트 블록 추출 (표 제외)
                print(f"  [DEBUG] 텍스트 블록 추출 시작...")
                text_blocks = page.get_text("dict")["blocks"]
                print(f"  [DEBUG] 원본 텍스트 블록: {len(text_blocks)}개")
                
                filtered_blocks = self._filter_out_table_blocks(text_blocks, table_areas)
                print(f"  [DEBUG] 필터링 후 블록: {len(filtered_blocks)}개")
                
                # 블록 정보 저장
                page_blocks = 0
                for i, block in enumerate(filtered_blocks):
                    if "lines" in block:
                        block_text = self._extract_text_from_block(block)
                        if block_text.strip():
                            all_blocks.append({
                                'block_id': f"page_{page_num + 1}_block_{i}",
                                'content': block_text.strip(),
                                'page': page_num + 1,
                                'bbox': block['bbox'],
                                'y_position': block['bbox'][1]
                            })
                            page_blocks += 1
                
                print(f"  [INFO] 페이지 {page_num + 1} 완료: {page_blocks}개 블록 추출")
                
            except Exception as e:
                print(f"  [ERROR] 페이지 {page_num + 1} 처리 실패: {e}")
                continue
        
        doc.close()
        print(f"[SUCCESS] 텍스트 블록 추출 완료: {len(all_blocks)}개")
        return all_blocks
    
    def _detect_table_areas(self, page) -> List:
        """표 영역 감지 (강화된 버전)"""
        table_areas = []
        try:
            print(f"    [DEBUG] PyMuPDF 표 감지 시작...")
            # PyMuPDF의 표 감지 (더 엄격한 설정)
            table_finder = page.find_tables(table_settings={"vertical_strategy": "text", "horizontal_strategy": "text"})
            tables = table_finder.tables
            print(f"    [DEBUG] PyMuPDF 감지된 표: {len(tables)}개")
            
            for i, table in enumerate(tables):
                try:
                    # 표 크기와 내용을 확인하여 실제 표인지 검증
                    table_content = table.extract()
                    if self._is_real_table(table_content, table.bbox):
                        table_areas.append(table.bbox)
                        print(f"    [DEBUG] 실제 표 감지됨: {table.bbox}")
                        print(f"    [DEBUG] 표 내용 미리보기: {str(table_content)[:100]}...")
                    else:
                        print(f"    [DEBUG] 가짜 표 제외됨: {table.bbox}")
                except Exception as e:
                    print(f"    [WARNING] 표 {i} 처리 실패: {e}")
                    continue
            
            # 추가적인 표 감지 방법 (텍스트 블록 패턴 기반)
            print(f"    [DEBUG] 텍스트 블록 패턴 감지 시작...")
            text_blocks = page.get_text("dict")["blocks"]
            pattern_detected = 0
            for j, block in enumerate(text_blocks):
                try:
                    if "lines" in block:
                        block_text = ""
                        for line in block["lines"]:
                            for span in line["spans"]:
                                block_text += span["text"] + " "
                        
                        # 표 패턴 감지 (더 엄격한 기준)
                        if self._is_table_pattern(block_text):
                            table_areas.append(block["bbox"])
                            pattern_detected += 1
                            print(f"    [DEBUG] 표 패턴 감지됨: {block['bbox']}")
                except Exception as e:
                    print(f"    [WARNING] 블록 {j} 패턴 감지 실패: {e}")
                    continue
            
            print(f"    [DEBUG] 패턴 기반 감지: {pattern_detected}개")
                        
        except Exception as e:
            print(f"    [WARNING] 표 감지 실패: {e}")
        
        print(f"    [DEBUG] 총 감지된 표 영역: {len(table_areas)}개")
        return table_areas
    
    def _is_real_table(self, table_content, bbox) -> bool:
        """실제 표인지 검증 (완화된 기준)"""
        if not table_content or len(table_content) == 0:
            return False
        
        # 표 크기 확인 (완화된 기준)
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        # 너무 큰 영역은 제외 (완화된 기준)
        if width > 500 or height > 400:  # 300x200 → 500x400으로 완화
            print(f"[DEBUG] 너무 큰 영역 제외: {width:.1f} x {height:.1f}")
            return False
        
        # 표 내용 분석
        content_str = str(table_content)
        
        # 체크박스나 라디오 버튼이 많은 경우 (완화된 기준)
        checkbox_count = content_str.count('□') + content_str.count('■') + content_str.count('☐') + content_str.count('☑')
        if checkbox_count > 8:  # 3개 → 8개로 완화
            print(f"[DEBUG] 체크박스가 많은 폼 제외: {checkbox_count}개")
            return False
        
        # 실제 표 구조 확인 (행과 열이 있는지)
        if isinstance(table_content, list) and len(table_content) > 0:
            # 첫 번째 행의 열 개수 확인
            first_row = table_content[0]
            if isinstance(first_row, list) and len(first_row) > 1:
                # 모든 행이 비슷한 열 개수를 가지는지 확인
                col_counts = [len(row) if isinstance(row, list) else 1 for row in table_content]
                if len(set(col_counts)) <= 2:  # 열 개수가 일정하면 표
                    return True
        
        return False
    
    def _is_table_pattern(self, text: str) -> bool:
        """텍스트가 표 패턴인지 확인 (완화된 기준)"""
        # 표 패턴 감지 규칙
        lines = text.strip().split('\n')
        if len(lines) < 2:
            return False
        
        # 숫자와 텍스트가 반복되는 패턴 확인
        has_numbers = any(any(c.isdigit() for c in line) for line in lines)
        has_text = any(any(c.isalpha() for c in line) for line in lines)
        
        # 탭이나 여러 공백으로 구분된 패턴 확인
        has_tab_separated = any('\t' in line or '  ' in line for line in lines)
        
        # 짧은 라인들이 반복되는 패턴 (표의 특징) - 완화된 기준
        short_lines = [line for line in lines if len(line.strip()) < 80]  # 50 → 80으로 완화
        has_short_lines = len(short_lines) >= len(lines) * 0.8  # 0.7 → 0.8로 완화
        
        # 체크박스나 라디오 버튼이 많은 경우 제외 (완화된 기준)
        checkbox_count = text.count('□') + text.count('■') + text.count('☐') + text.count('☑')
        if checkbox_count > 8:  # 3개 → 8개로 완화
            return False
        
        return (has_numbers and has_text and (has_tab_separated or has_short_lines))
    
    def _filter_out_table_blocks(self, text_blocks: List, table_areas: List) -> List:
        """표 영역을 제외한 텍스트 블록 필터링 (완화된 버전)"""
        filtered_blocks = []
        
        for block in text_blocks:
            if "lines" not in block:
                continue
            
            block_bbox = block["bbox"]
            block_text = ""
            for line in block["lines"]:
                for span in line["spans"]:
                    block_text += span["text"] + " "
            
            # 표 영역과 겹치는지 확인 (완화된 기준)
            is_in_table = False
            for table_bbox in table_areas:
                if self._bbox_overlap(block_bbox, table_bbox, threshold=0.3):  # 10% → 30%로 완화
                    is_in_table = True
                    print(f"[DEBUG] 표 영역 제외: {block_bbox}")
                    break
            
            # 표 패턴인지 추가 확인 (완화된 기준)
            if not is_in_table and self._is_table_pattern(block_text):
                is_in_table = True
                print(f"[DEBUG] 표 패턴 제외: {block_bbox}")
            
            # 체크박스나 라디오 버튼이 많은 블록 제외 (완화된 기준)
            checkbox_count = block_text.count('□') + block_text.count('■') + block_text.count('☐') + block_text.count('☑')
            if checkbox_count > 5:  # 2개 → 5개로 완화
                is_in_table = True
                print(f"[DEBUG] 체크박스 블록 제외: {block_bbox} ({checkbox_count}개)")
            
            if not is_in_table:
                filtered_blocks.append(block)
        
        print(f"[INFO] 필터링 결과: {len(text_blocks)}개 블록 중 {len(filtered_blocks)}개 유지")
        return filtered_blocks
    
    def _extract_text_from_block(self, block: Dict) -> str:
        """블록에서 텍스트 추출 및 전처리"""
        text = ""
        for line in block["lines"]:
            for span in line["spans"]:
                text += span["text"] + " "
            # 개행 제거하고 공백으로 대체
            text += " "
        
        # 전처리 적용
        cleaned_text = self._preprocess_text(text.strip())
        return cleaned_text
    
    def _bbox_overlap(self, bbox1: List, bbox2: List, threshold: float = 0.3) -> bool:
        """두 bbox가 겹치는지 확인"""
        x1, y1, x2, y2 = bbox1
        x3, y3, x4, y4 = bbox2
        
        overlap_x = max(0, min(x2, x4) - max(x1, x3))
        overlap_y = max(0, min(y2, y4) - max(y1, y3))
        
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x4 - x3) * (y4 - y3)
        overlap_area = overlap_x * overlap_y
        
        if area1 == 0 or area2 == 0:
            return False
        
        overlap_ratio = overlap_area / min(area1, area2)
        return overlap_ratio > threshold
    
    def semantic_clustering(self, text_blocks: List[Dict]) -> List[List[Dict]]:
        """의미 기반 클러스터링 (임베딩 기반)"""
        print("[INFO] 의미 기반 클러스터링 시작...")
        
        if len(text_blocks) < 2:
            # 블록이 1개 이하면 단일 클러스터로 처리
            return [text_blocks]
        
        # 텍스트 내용 추출
        texts = [block['content'] for block in text_blocks]
        
        # 임베딩 생성
        embeddings = self.model.encode(texts)
        
        # 클러스터 수 결정 (문서 크기에 따라 동적 조정)
        n_clusters = min(len(text_blocks) // 3, 8)  # 최대 8개 클러스터
        n_clusters = max(n_clusters, 2)  # 최소 2개 클러스터
        
        # K-means 클러스터링
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # 클러스터별로 블록 그룹화
        clusters = [[] for _ in range(n_clusters)]
        for i, label in enumerate(cluster_labels):
            clusters[label].append(text_blocks[i])
        
        # 클러스터 정보 출력
        for i, cluster in enumerate(clusters):
            print(f"[INFO] 클러스터 {i+1}: {len(cluster)}개 블록")
        
        return clusters
    
    def content_based_clustering(self, text_blocks: List[Dict]) -> List[List[Dict]]:
        """내용 기반 클러스터링 (더 일반적인 방식)"""
        print("[INFO] 내용 기반 클러스터링 시작...")
        
        if len(text_blocks) < 2:
            return [text_blocks]
        
        # 텍스트 내용 추출
        texts = [block['content'] for block in text_blocks]
        
        # 임베딩 생성
        embeddings = self.model.encode(texts)
        
        # 클러스터 수 결정
        n_clusters = min(len(text_blocks) // 2, 6)  # 더 적은 클러스터 수
        n_clusters = max(n_clusters, 2)
        
        # K-means 클러스터링
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # 클러스터별로 블록 그룹화
        clusters = [[] for _ in range(n_clusters)]
        for i, label in enumerate(cluster_labels):
            clusters[label].append(text_blocks[i])
        
        # 빈 클러스터 제거
        result_clusters = []
        for i, cluster in enumerate(clusters):
            if cluster:
                result_clusters.append(cluster)
                print(f"[INFO] 클러스터 {i+1}: {len(cluster)}개 블록")
        
        return result_clusters
    
    def split_cluster_by_length(self, cluster: List[Dict], max_length: int = 500, overlap_ratio: float = 0.5) -> List[Dict]:
        """클러스터를 길이 기준으로 분할 (overlap 포함)"""
        chunks = []
        current_chunk = ""
        chunk_id = 1
        overlap_text = ""  # 이전 청크의 마지막 부분을 저장
        
        # 클러스터 내 블록들을 Y 위치 순으로 정렬
        sorted_blocks = sorted(cluster, key=lambda x: x['y_position'])
        
        for i, block in enumerate(sorted_blocks):
            block_text = block['content']
            
            # 현재 청크에 블록을 추가했을 때 길이 확인
            if len(current_chunk + block_text) <= max_length:
                current_chunk += block_text + " "  # 개행 대신 공백 사용
            else:
                # 현재 청크가 있으면 저장
                if current_chunk.strip():
                    # 전처리 적용
                    final_chunk = self._preprocess_text(current_chunk.strip())
                    chunks.append({
                        'chunk_id': f"semantic_{chunk_id:03d}",
                        'content': final_chunk,
                        'content_length': len(final_chunk),
                        'chunk_type': 'semantic_clustered',
                        'page': sorted_blocks[0]['page'] if sorted_blocks else 0,  # 첫 번째 블록의 페이지 번호
                        'y_position': sorted_blocks[0]['y_position'] if sorted_blocks else 0,  # 첫 번째 블록의 Y 위치
                        'source_blocks': [b['block_id'] for b in sorted_blocks[:i]],
                        'cluster_category': self._identify_content_category(final_chunk),
                        'overlap_ratio': overlap_ratio
                    })
                    chunk_id += 1
                
                # overlap 텍스트 생성 (이전 청크의 마지막 부분)
                overlap_text = self._create_overlap_text(current_chunk, overlap_ratio)
                
                # 새 청크 시작 (overlap 텍스트 포함)
                current_chunk = overlap_text + block_text + " "  # 개행 대신 공백 사용
        
        # 마지막 청크 처리
        if current_chunk.strip():
            # 전처리 적용
            final_chunk = self._preprocess_text(current_chunk.strip())
            chunks.append({
                'chunk_id': f"semantic_{chunk_id:03d}",
                'content': final_chunk,
                'content_length': len(final_chunk),
                'chunk_type': 'semantic_clustered',
                'page': sorted_blocks[0]['page'] if sorted_blocks else 0,  # 첫 번째 블록의 페이지 번호
                'y_position': sorted_blocks[0]['y_position'] if sorted_blocks else 0,  # 첫 번째 블록의 Y 위치
                'source_blocks': [b['block_id'] for b in sorted_blocks],
                'cluster_category': self._identify_content_category(final_chunk),
                'overlap_ratio': overlap_ratio
            })
        
        return chunks
    
    def _create_overlap_text(self, text: str, overlap_ratio: float = 0.5) -> str:
        """이전 청크의 마지막 부분을 overlap 텍스트로 생성"""
        if not text.strip():
            return ""
        
        # 문장 단위로 분할
        sentences = self._split_into_sentences(text)
        
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
        
        return overlap_text + " "  # 개행 대신 공백 사용
    
    def _split_into_sentences(self, text: str) -> List[str]:
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
    
    def _identify_content_category(self, text: str) -> str:
        """텍스트의 내용 카테고리 식별 (더 일반적인 방식)"""
        text_lower = text.lower()
        
        # 일반적인 문서 섹션 키워드로 카테고리 식별
        if any(word in text_lower for word in ['서론', '개요', '배경', '목적']):
            return '서론'
        elif any(word in text_lower for word in ['방법', '절차', '과정', '단계']):
            return '방법'
        elif any(word in text_lower for word in ['결과', '성과', '효과', '성능']):
            return '결과'
        elif any(word in text_lower for word in ['결론', '요약', '정리']):
            return '결론'
        elif any(word in text_lower for word in ['참고', '부록', '첨부']):
            return '참고'
        else:
            return '본문'
    
    def sequential_chunking(self, text_blocks: List[Dict], max_length: int = 500, overlap_ratio: float = 0.5) -> List[Dict]:
        """순차적 청킹 (문서 순서 그대로)"""
        chunks = []
        current_chunk = ""
        chunk_id = 1
        overlap_text = ""
        
        for i, block in enumerate(text_blocks):
            block_text = block['content']
            
            # 현재 청크에 블록을 추가했을 때 길이 확인
            if len(current_chunk + block_text) <= max_length:
                current_chunk += block_text + " "
            else:
                # 현재 청크가 있으면 저장
                if current_chunk.strip():
                    final_chunk = self._preprocess_text(current_chunk.strip())
                    
                    chunks.append({
                        'chunk_id': f"sequential_{chunk_id:03d}",
                        'content': final_chunk,
                        'content_length': len(final_chunk),
                        'chunk_type': 'sequential_chunked',
                        'page': text_blocks[i-1]['page'] if i > 0 else 0,
                        'y_position': text_blocks[i-1]['y_position'] if i > 0 else 0,
                        'source_blocks': [b['block_id'] for b in text_blocks[max(0, i-len(current_chunk.split())):i]],
                        'overlap_ratio': overlap_ratio
                    })
                    chunk_id += 1
                
                # overlap 텍스트 생성
                overlap_text = self._create_overlap_text(current_chunk, overlap_ratio)
                current_chunk = overlap_text + block_text + " "
        
        # 마지막 청크 처리
        if current_chunk.strip():
            final_chunk = self._preprocess_text(current_chunk.strip())
            
            chunks.append({
                'chunk_id': f"sequential_{chunk_id:03d}",
                'content': final_chunk,
                'content_length': len(final_chunk),
                'chunk_type': 'sequential_chunked',
                'page': text_blocks[-1]['page'] if text_blocks else 0,
                'y_position': text_blocks[-1]['y_position'] if text_blocks else 0,
                'source_blocks': [b['block_id'] for b in text_blocks[-len(current_chunk.split()):]],
                'overlap_ratio': overlap_ratio
            })
        
        return chunks
    
    def generate_semantic_chunks(self, pdf_path: str) -> List[Dict]:
        """순차적 청크 생성 (의미적 보정 포함)"""
        print("=" * 60)
        print("순차적 청크 생성 시작")
        print("=" * 60)
        
        try:
            # 1. 텍스트 블록 추출
            print("[STEP 1] 텍스트 블록 추출 시작...")
            text_blocks = self.extract_text_blocks(pdf_path)
            
            if not text_blocks:
                print("[WARNING] 추출된 텍스트 블록이 없습니다.")
                return []
            
            print(f"[STEP 1] 완료: {len(text_blocks)}개 블록 추출")
            
            # 2. 페이지 및 위치 순으로 정렬
            print("[STEP 2] 블록 정렬 시작...")
            sorted_blocks = sorted(text_blocks, key=lambda x: (x['page'], x['y_position']))
            print(f"[STEP 2] 완료: {len(sorted_blocks)}개 블록 정렬")
            
            # 3. 순차적 청킹 (의미적 보정 포함)
            print("[STEP 3] 순차적 청킹 시작...")
            all_chunks = self.sequential_chunking(sorted_blocks, max_length=300, overlap_ratio=0.5)
            print(f"[STEP 3] 완료: {len(all_chunks)}개 청크 생성")
            
            # 4. 결과 저장
            print("[STEP 4] 결과 저장 시작...")
            output_file = "semantic_chunks.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_chunks, f, ensure_ascii=False, indent=2)
            
            print(f"\n[SUCCESS] 순차적 청크 생성 완료: {len(all_chunks)}개 청크")
            print(f"[INFO] 결과 저장: {output_file}")
            print(f"[INFO] 청크 길이: 300자 (overlap: 50%)")
            
            # 청크 길이 통계
            if all_chunks:
                lengths = [chunk['content_length'] for chunk in all_chunks]
                avg_length = sum(lengths) / len(lengths) if lengths else 0
                print(f"\n📊 청크 통계:")
                print(f"  - 평균 길이: {avg_length:.1f}자")
                print(f"  - 최소 길이: {min(lengths)}자")
                print(f"  - 최대 길이: {max(lengths)}자")
            
            return all_chunks
            
        except Exception as e:
            print(f"[ERROR] 청크 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            return []

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("사용법: python semantic_chunk_generator.py <PDF파일경로>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    if not Path(pdf_path).exists():
        print(f"[ERROR] 파일을 찾을 수 없습니다: {pdf_path}")
        sys.exit(1)
    
    # 의미 기반 청크 생성기 실행
    generator = SemanticChunkGenerator()
    generator.generate_semantic_chunks(pdf_path)

if __name__ == "__main__":
    main() 