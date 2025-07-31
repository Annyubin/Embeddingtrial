#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import fitz
import json
import sys
import re
import requests
import os
from pathlib import Path
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# NLTK 데이터 다운로드
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def check_ollama_connection():
    """Ollama 연결 상태 확인"""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        return response.status_code == 200
    except:
        return False

def create_improved_summary_prompt(content):
    """개선된 요약 생성을 위한 프롬프트 생성"""
    prompt = f"""
다음 텍스트를 간결하고 명확하게 요약해주세요. 다음 규칙을 따라주세요:

1. 핵심 내용만 포함 (50자 이내)
2. 구체적이고 명확한 표현 사용
3. 반복적인 표현 제거
4. 전문 용어는 그대로 유지하되 설명 추가
5. 문장은 간결하게 구성

텍스트:
{content}

요약 (50자 이내):
"""
    return prompt

def extract_semantic_boundaries(text):
    """의미적 경계를 찾아 동적 청크 크기 결정"""
    sentences = sent_tokenize(text)
    
    # 문장별 중요도 계산
    tfidf = TfidfVectorizer(max_features=100, stop_words='english')
    try:
        tfidf_matrix = tfidf.fit_transform(sentences)
        sentence_importance = np.sum(tfidf_matrix.toarray(), axis=1)
    except:
        # TF-IDF 실패시 문장 길이 기반 중요도
        sentence_importance = [len(sent) for sent in sentences]
    
    # 의미적 경계 찾기
    boundaries = []
    current_chunk = []
    current_length = 0
    max_chunk_length = 800  # 기본 최대 길이
    min_chunk_length = 200  # 최소 길이
    
    for i, sentence in enumerate(sentences):
        sentence_length = len(sentence)
        
        # 문장이 너무 길면 강제로 분할
        if sentence_length > 400:
            if current_chunk:
                boundaries.append(current_chunk)
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk = [sentence]
                current_length = sentence_length
            continue
        
        # 현재 청크에 추가할지 결정
        if current_length + sentence_length <= max_chunk_length:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            # 최소 길이 확인
            if current_length >= min_chunk_length:
                boundaries.append(current_chunk)
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                # 최소 길이 미달시 강제로 추가
                current_chunk.append(sentence)
                current_length += sentence_length
    
    # 마지막 청크 추가
    if current_chunk:
        boundaries.append(current_chunk)
    
    return boundaries

def create_semantic_chunks(blocks, overlap_ratio=0.2):
    """의미적 완결성을 고려한 동적 청크 생성"""
    # 모든 텍스트 블록을 하나로 합치기
    full_text = " ".join([block['text'] for block in blocks])
    
    # 의미적 경계로 분할
    semantic_chunks = extract_semantic_boundaries(full_text)
    
    chunks = []
    chunk_id = 1
    
    for i, chunk_sentences in enumerate(semantic_chunks):
        chunk_text = " ".join(chunk_sentences)
        
        # 청크 정보 생성
        chunk_info = {
            'chunk_id': f"chunk_{chunk_id:03d}",
            'content': chunk_text,
            'content_length': len(chunk_text),
            'page': blocks[0]['page'] if blocks else 1,
            'y_position': blocks[0]['y'] if blocks else 0,
            'source_blocks': [block['block_id'] for block in blocks],
            'overlap_ratio': overlap_ratio,
            'semantic_boundary': True,
            'sentence_count': len(chunk_sentences)
        }
        
        chunks.append(chunk_info)
        chunk_id += 1
    
    return chunks

def process_chunk_with_improved_summary(chunk_content, chunk_id):
    """개선된 요약으로 청크 처리"""
    if not check_ollama_connection():
        return create_default_improved_result(chunk_content, chunk_id)
    
    try:
        # 개선된 프롬프트 생성
        prompt = create_improved_summary_prompt(chunk_content)
        
        # Ollama API 호출
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": False
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            summary = result['response'].strip()
            
            # 요약 품질 검증
            if len(summary) > 100 or len(summary) < 10:
                summary = create_default_summary(chunk_content)
            
            return {
                'chunk_id': chunk_id,
                'content': chunk_content,
                'improved_summary': summary,
                'summary_length': len(summary),
                'confidence': calculate_summary_confidence(summary, chunk_content),
                'extraction_method': 'improved_llm_ollama'
            }
        else:
            return create_default_improved_result(chunk_content, chunk_id)
            
    except Exception as e:
        print(f"LLM 처리 오류: {e}")
        return create_default_improved_result(chunk_content, chunk_id)

def create_default_summary(content):
    """기본 요약 생성"""
    # 핵심 키워드 추출
    words = word_tokenize(content)
    stop_words = set(stopwords.words('english'))
    
    # 한국어 키워드 패턴
    korean_keywords = [
        '임상시험', '폐결핵', '치료', '약물', '평가', '가이드라인',
        '안전성', '유효성', '환자', '시험', '방법', '고려사항'
    ]
    
    # 영어 키워드 패턴
    english_keywords = [
        'clinical', 'trial', 'tuberculosis', 'treatment', 'drug',
        'evaluation', 'guideline', 'safety', 'efficacy', 'patient'
    ]
    
    # 키워드 기반 요약 생성
    found_keywords = []
    for word in words:
        if word.lower() in english_keywords or word in korean_keywords:
            found_keywords.append(word)
    
    if found_keywords:
        return f"{found_keywords[0]} 관련 {found_keywords[1] if len(found_keywords) > 1 else '내용'} 설명"
    else:
        return "문서 내용 요약"

def calculate_summary_confidence(summary, content):
    """요약 신뢰도 계산"""
    # 요약 길이 기반 신뢰도
    length_confidence = min(len(summary) / 50, 1.0)
    
    # 키워드 일치도 기반 신뢰도
    content_words = set(word_tokenize(content.lower()))
    summary_words = set(word_tokenize(summary.lower()))
    
    if content_words:
        keyword_overlap = len(content_words.intersection(summary_words)) / len(content_words)
    else:
        keyword_overlap = 0
    
    # 최종 신뢰도 계산
    confidence = (length_confidence * 0.4 + keyword_overlap * 0.6)
    return min(confidence, 1.0)

def create_default_improved_result(content, chunk_id):
    """기본 개선된 결과 생성"""
    summary = create_default_summary(content)
    return {
        'chunk_id': chunk_id,
        'content': content,
        'improved_summary': summary,
        'summary_length': len(summary),
        'confidence': 0.7,
        'extraction_method': 'default_improved'
    }

def process_blocks_with_semantic_chunking(blocks):
    """의미적 청킹으로 블록 처리"""
    print("의미적 완결성을 고려한 동적 청크 생성 중...")
    
    # 의미적 청크 생성
    semantic_chunks = create_semantic_chunks(blocks)
    
    print(f"생성된 의미적 청크 수: {len(semantic_chunks)}")
    
    # 각 청크에 대해 개선된 요약 생성
    improved_results = []
    
    for chunk in semantic_chunks:
        print(f"청크 {chunk['chunk_id']} 처리 중... (길이: {chunk['content_length']}자)")
        
        # 개선된 요약 생성
        improved_result = process_chunk_with_improved_summary(
            chunk['content'], 
            chunk['chunk_id']
        )
        
        # 청크 정보와 요약 정보 병합
        final_result = {**chunk, **improved_result}
        improved_results.append(final_result)
    
    return improved_results

def analyze_summary_quality(results):
    """요약 품질 분석"""
    print("\n=== 요약 품질 분석 ===")
    
    summary_lengths = [r.get('summary_length', 0) for r in results]
    confidences = [r.get('confidence', 0) for r in results]
    
    print(f"총 청크 수: {len(results)}")
    print(f"평균 요약 길이: {np.mean(summary_lengths):.1f}자")
    print(f"평균 신뢰도: {np.mean(confidences):.2f}")
    
    # 품질별 분류
    high_quality = [r for r in results if r.get('confidence', 0) >= 0.8]
    medium_quality = [r for r in results if 0.6 <= r.get('confidence', 0) < 0.8]
    low_quality = [r for r in results if r.get('confidence', 0) < 0.6]
    
    print(f"고품질 요약: {len(high_quality)}개")
    print(f"중간품질 요약: {len(medium_quality)}개")
    print(f"저품질 요약: {len(low_quality)}개")
    
    return {
        'high_quality': high_quality,
        'medium_quality': medium_quality,
        'low_quality': low_quality
    }

def save_improved_results(results, output_file):
    """개선된 결과 저장"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n개선된 결과가 {output_file}에 저장되었습니다.")

def main():
    """메인 함수"""
    print("=== 개선된 요약 및 동적 청크 파이프라인 ===")
    
    # 기존 결과 파일 로드
    input_file = "full_pipeline_llm_enhanced_chunks.json"
    
    if not os.path.exists(input_file):
        print(f"입력 파일 {input_file}을 찾을 수 없습니다.")
        return
    
    with open(input_file, 'r', encoding='utf-8') as f:
        existing_results = json.load(f)
    
    print(f"기존 결과 로드 완료: {len(existing_results)}개 청크")
    
    # 기존 청크를 블록 형태로 변환
    blocks = []
    for result in existing_results:
        block = {
            'text': result['content'],
            'page': result['page'],
            'y': result['y_position'],
            'block_id': result['chunk_id']
        }
        blocks.append(block)
    
    # 의미적 청킹 및 개선된 요약 생성
    improved_results = process_blocks_with_semantic_chunking(blocks)
    
    # 품질 분석
    quality_analysis = analyze_summary_quality(improved_results)
    
    # 결과 저장
    output_file = "improved_summary_chunks.json"
    save_improved_results(improved_results, output_file)
    
    print("\n=== 처리 완료 ===")
    print(f"개선된 청크 수: {len(improved_results)}")
    print(f"평균 요약 길이: {np.mean([r.get('summary_length', 0) for r in improved_results]):.1f}자")
    print(f"평균 신뢰도: {np.mean([r.get('confidence', 0) for r in improved_results]):.2f}")

if __name__ == "__main__":
    main() 