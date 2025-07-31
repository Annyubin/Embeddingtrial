#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
임베딩 생성기
텍스트 청크를 임베딩하여 FAISS 인덱스와 PKL 파일을 생성
"""

import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
import sys
from typing import List, Dict, Tuple
import os

class EmbeddingGenerator:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """임베딩 생성기 초기화"""
        self.model_name = model_name
        # CPU 모드로 강제 실행 (CUDA 호환성 문제 해결)
        import torch
        device = torch.device('cpu')
        self.model = SentenceTransformer(model_name, device=device)
        print(f"[INFO] 임베딩 모델 로드 (CPU 모드): {model_name}")
    
    def load_text_chunks(self, json_file: str) -> List[Dict]:
        """JSON 파일에서 텍스트 청크 로드"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            print(f"[INFO] 텍스트 청크 로드 완료: {len(chunks)}개")
            return chunks
        except Exception as e:
            print(f"[ERROR] 텍스트 청크 로드 실패: {e}")
            return []
    
    def prepare_texts_for_embedding(self, chunks: List[Dict]) -> Tuple[List[str], List[Dict]]:
        """임베딩을 위한 텍스트 준비 (개선된 전처리)"""
        texts = []
        metadata = []
        
        for chunk in chunks:
            # 텍스트 내용 추출 및 전처리 (LLM 강화된 데이터는 summary 사용)
            content = chunk.get('content', '')
            summary = chunk.get('summary', '')
            
            # content가 없으면 summary 사용
            if not content and summary:
                content = summary
            elif not content and not summary:
                continue  # 둘 다 없으면 건너뛰기
            
            if len(content.strip()) < 10:  # 너무 짧은 텍스트 제외
                continue
            
            # 텍스트 정제
            content = self._clean_text(content)
            
            # 섹션 제목이 있으면 함께 사용
            section_title = chunk.get('section_title', '')
            if section_title and not section_title.startswith('[Ollama 연결 실패]'):
                section_title = self._clean_text(section_title)
                # 섹션 제목과 내용을 조합 (더 강한 가중치)
                combined_text = f"제목: {section_title} | 내용: {content}"
            else:
                combined_text = content
            
            # 키워드가 있으면 추가 정보로 활용
            keywords = chunk.get('keywords', [])
            if keywords:
                keywords_text = ', '.join(keywords)
                combined_text = f"{combined_text} | 키워드: {keywords_text}"
            
            # 클러스터 카테고리 정보 추가
            cluster_category = chunk.get('cluster_category', '')
            if cluster_category:
                combined_text = f"{combined_text} | 카테고리: {cluster_category}"
            
            texts.append(combined_text)
            metadata.append(chunk)
        
        print(f"[INFO] 임베딩용 텍스트 준비 완료: {len(texts)}개")
        return texts, metadata
    
    def _clean_text(self, text: str) -> str:
        """텍스트 정제"""
        import re
        
        # 불필요한 공백 제거
        text = re.sub(r'\s+', ' ', text.strip())
        
        # 특수문자 정리 (한글, 영문, 숫자, 기본 문장부호만 유지)
        text = re.sub(r'[^\w\s가-힣\-\.\,\;\:\!\?\(\)\[\]]', ' ', text)
        
        # 연속된 공백을 하나로
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """텍스트 임베딩 생성"""
        print(f"[INFO] 임베딩 생성 시작: {len(texts)}개 텍스트")
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)
        print(f"[INFO] 임베딩 생성 완료: {embeddings.shape}")
        return embeddings
    
    def create_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """FAISS 인덱스 생성 (이미 정규화된 벡터 사용)"""
        print("[INFO] FAISS 인덱스 생성 시작")
        
        # 이미 정규화된 벡터이므로 추가 정규화 불필요
        
        # IVF 인덱스 생성 (클러스터 수는 데이터 크기에 따라 조정)
        n_clusters = min(100, len(embeddings) // 10) if len(embeddings) > 100 else 10
        index = faiss.IndexIVFFlat(faiss.IndexFlatIP(embeddings.shape[1]), embeddings.shape[1], n_clusters)
        
        # 인덱스 훈련
        index.train(embeddings)
        index.add(embeddings)
        
        print(f"[INFO] FAISS 인덱스 생성 완료: {index.ntotal}개 벡터")
        return index
    
    def save_faiss_index(self, index: faiss.Index, filename: str = "embeddings.faiss"):
        """FAISS 인덱스 저장"""
        try:
            faiss.write_index(index, filename)
            print(f"[SUCCESS] FAISS 인덱스 저장: {filename}")
            return True
        except Exception as e:
            print(f"[ERROR] FAISS 인덱스 저장 실패: {e}")
            return False
    
    def save_metadata_pkl(self, metadata: List[Dict], filename: str = "embeddings_metadata.pkl"):
        """메타데이터 PKL 파일 저장"""
        try:
            with open(filename, 'wb') as f:
                pickle.dump(metadata, f)
            print(f"[SUCCESS] 메타데이터 PKL 저장: {filename}")
            return True
        except Exception as e:
            print(f"[ERROR] 메타데이터 PKL 저장 실패: {e}")
            return False
    
    def generate_embeddings_from_file(self, input_file: str, output_prefix: str = "embeddings"):
        """파일에서 임베딩 생성"""
        print("=" * 60)
        print("임베딩 생성 시작")
        print("=" * 60)
        
        # 1. 텍스트 청크 로드
        chunks = self.load_text_chunks(input_file)
        if not chunks:
            print("[ERROR] 텍스트 청크를 로드할 수 없습니다.")
            return False
        
        # 2. 임베딩용 텍스트 준비
        texts, metadata = self.prepare_texts_for_embedding(chunks)
        if not texts:
            print("[ERROR] 임베딩용 텍스트가 없습니다.")
            return False
        
        # 3. 임베딩 생성
        embeddings = self.generate_embeddings(texts)
        
        # 4. L2 정규화 적용 (코사인 유사도 사용)
        normalized_embeddings = embeddings.copy()
        faiss.normalize_L2(normalized_embeddings)
        
        # 5. 메타데이터에 정규화된 임베딩 벡터 추가
        for i, chunk in enumerate(metadata):
            chunk['embedding'] = normalized_embeddings[i].tolist()  # 정규화된 numpy 배열을 리스트로 변환
        
        # 6. FAISS 인덱스 생성 (이미 정규화된 벡터 사용)
        index = self.create_faiss_index(normalized_embeddings)
        
        # 6. 파일 저장
        faiss_filename = f"{output_prefix}.faiss"
        pkl_filename = f"{output_prefix}_metadata.pkl"
        
        faiss_success = self.save_faiss_index(index, faiss_filename)
        pkl_success = self.save_metadata_pkl(metadata, pkl_filename)
        
        if faiss_success and pkl_success:
            print("=" * 60)
            print("임베딩 생성 완료!")
            print(f"- FAISS 인덱스: {faiss_filename}")
            print(f"- 메타데이터 PKL: {pkl_filename}")
            print(f"- 총 벡터 수: {index.ntotal}")
            print(f"- 벡터 차원: {embeddings.shape[1]}")
            print("=" * 60)
            return True
        else:
            print("[ERROR] 파일 저장 중 오류가 발생했습니다.")
            return False

def main():
    if len(sys.argv) < 2:
        print("사용법: python embedding_generator.py <입력JSON파일> [출력파일접두사]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_prefix = sys.argv[2] if len(sys.argv) > 2 else "embeddings"
    
    if not os.path.exists(input_file):
        print(f"[ERROR] 파일을 찾을 수 없습니다: {input_file}")
        sys.exit(1)
    
    # 임베딩 생성기 실행
    generator = EmbeddingGenerator()
    success = generator.generate_embeddings_from_file(input_file, output_prefix)
    
    if success:
        print("임베딩 생성이 성공적으로 완료되었습니다!")
    else:
        print("임베딩 생성에 실패했습니다.")
        sys.exit(1)

if __name__ == "__main__":
    main() 