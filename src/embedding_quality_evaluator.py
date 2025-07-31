#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
임베딩 품질 평가기
임베딩 벡터의 품질을 다양한 관점에서 평가
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
import torch
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from collections import Counter
import re

class EmbeddingQualityEvaluator:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """평가 모델 초기화"""
        # CPU 모드로 강제 설정
        device = torch.device('cpu')
        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        print(f"[INFO] 평가 모델 로드: {model_name} (CPU 모드)")

    def load_experiment_data(self, experiment_dir: str) -> Tuple[List[np.ndarray], List[Dict], List[Dict]]:
        """실험 데이터 로드 (임베딩 벡터는 PKL에서 추출)"""
        print(f"[INFO] 실험 데이터 로드: {experiment_dir}")

        # 메타데이터 PKL 로드 (최신 파일 우선)
        possible_pkl_files = ["embeddings_metadata.pkl", "embeddings_improved_metadata.pkl", "실험.pkl", "metadata.pkl"]
        pkl_path = None
        for pkl_file in possible_pkl_files:
            test_path = os.path.join(experiment_dir, pkl_file)
            if os.path.exists(test_path):
                pkl_path = test_path
                break
        
        if pkl_path is None:
            raise FileNotFoundError(f"PKL 파일을 찾을 수 없습니다. 시도한 파일들: {possible_pkl_files}")

        with open(pkl_path, 'rb') as f:
            metadata = pickle.load(f)
        print(f"[INFO] 메타데이터 로드: {len(metadata)}개 항목")

        # 벡터 추출 (embedding 필드가 있다고 가정)
        vectors = []
        for item in metadata:
            emb = item.get('embedding')
            if emb is not None:
                vectors.append(np.array(emb))
        vectors = np.stack(vectors)
        print(f"[INFO] 임베딩 벡터 추출: {vectors.shape}")

        # JSON 데이터 로드 (선택사항)
        json_path = os.path.join(experiment_dir, "실험.json")
        json_data = []
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            print(f"[INFO] JSON 데이터 로드: {len(json_data)}개 항목")

        return vectors, metadata, json_data

    def analyze_vector_distribution(self, vectors: np.ndarray) -> Dict:
        """벡터 분포 분석"""
        print("[INFO] 벡터 분포 분석 중...")

        stats = {
            'total_vectors': vectors.shape[0],
            'vector_dimension': vectors.shape[1],
            'mean_values': np.mean(vectors, axis=0),
            'std_values': np.std(vectors, axis=0),
            'min_values': np.min(vectors, axis=0),
            'max_values': np.max(vectors, axis=0),
            'vector_norms': np.linalg.norm(vectors, axis=1)
        }

        norms = stats['vector_norms']
        is_normalized = np.allclose(norms, 1.0, atol=1e-6)
        stats['is_normalized'] = is_normalized
        stats['norm_mean'] = np.mean(norms)
        stats['norm_std'] = np.std(norms)

        print(f"[INFO] 벡터 분포 분석 완료")
        print(f"- 정규화 상태: {'정규화됨' if is_normalized else '정규화되지 않음'}")
        print(f"- 평균 노름: {stats['norm_mean']:.6f}")
        print(f"- 노름 표준편차: {stats['norm_std']:.6f}")

        return stats

    def evaluate_semantic_similarity(self, metadata: List[Dict], vectors: np.ndarray) -> Dict:
        """의미적 유사도 평가 (내재적 평가 포함)"""
        print("[INFO] 의미적 유사도 평가 중...")

        # 1. 섹션별 클러스터링 평가
        section_groups = {}
        for i, item in enumerate(metadata):
            section_title = item.get('section_title', 'Unknown')
            if section_title not in section_groups:
                section_groups[section_title] = []
            section_groups[section_title].append(i)

        intra_similarities = []
        inter_similarities = []

        for section, indices in section_groups.items():
            if len(indices) > 1:
                for i in range(len(indices)):
                    for j in range(i+1, len(indices)):
                        sim = self._calculate_similarity_vec(vectors[indices[i]], vectors[indices[j]])
                        intra_similarities.append(sim)

        sections = list(section_groups.keys())
        for i in range(len(sections)):
            for j in range(i+1, len(sections)):
                for idx1 in section_groups[sections[i]]:
                    for idx2 in section_groups[sections[j]]:
                        sim = self._calculate_similarity_vec(vectors[idx1], vectors[idx2])
                        inter_similarities.append(sim)

        # 2. 도메인 전문 용어 유사도 평가 (의약품 도메인 특화)
        domain_pairs = [
            ("신청서", "유효성정보"),
            ("안전성정보", "유효성정보"),
            ("임상시험", "승인절차"),
            ("가이드라인", "평가기준"),
            ("민원인", "안내서")
        ]
        
        domain_similarities = []
        for term1, term2 in domain_pairs:
            # 키워드가 포함된 문서들 찾기
            term1_docs = [i for i, item in enumerate(metadata) 
                         if term1 in item.get('content', '') or term1 in item.get('section_title', '')]
            term2_docs = [i for i, item in enumerate(metadata) 
                         if term2 in item.get('content', '') or term2 in item.get('section_title', '')]
            
            if term1_docs and term2_docs:
                # 각 용어의 평균 벡터 계산
                term1_avg = np.mean([vectors[i] for i in term1_docs], axis=0)
                term2_avg = np.mean([vectors[i] for i in term2_docs], axis=0)
                
                sim = self._calculate_similarity_vec(term1_avg, term2_avg)
                domain_similarities.append(sim)

        # 3. 유추(analogy) 평가
        analogy_score = 0
        if len(domain_similarities) >= 3:
            # "신청서 - 유효성정보 + 안전성정보" 유추 평가
            analogy_score = np.mean(domain_similarities[:3])

        results = {
            'intra_similarity_mean': np.mean(intra_similarities) if intra_similarities else 0,
            'intra_similarity_std': np.std(intra_similarities) if intra_similarities else 0,
            'inter_similarity_mean': np.mean(inter_similarities) if inter_similarities else 0,
            'inter_similarity_std': np.std(inter_similarities) if inter_similarities else 0,
            'separation_score': np.mean(intra_similarities) - np.mean(inter_similarities) if intra_similarities and inter_similarities else 0,
            'domain_similarity_mean': np.mean(domain_similarities) if domain_similarities else 0,
            'analogy_score': analogy_score
        }

        print(f"[INFO] 의미적 유사도 평가 완료")
        print(f"- 같은 섹션 내 유사도: {results['intra_similarity_mean']:.4f} ± {results['intra_similarity_std']:.4f}")
        print(f"- 다른 섹션 간 유사도: {results['inter_similarity_mean']:.4f} ± {results['inter_similarity_std']:.4f}")
        print(f"- 분리 점수: {results['separation_score']:.4f}")
        print(f"- 도메인 용어 유사도: {results['domain_similarity_mean']:.4f}")
        print(f"- 유추 점수: {results['analogy_score']:.4f}")

        return results

    def _calculate_similarity_vec(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """두 벡터 간 코사인 유사도 계산"""
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    def evaluate_content_quality(self, metadata: List[Dict]) -> Dict:
        """콘텐츠 품질 평가"""
        print("[INFO] 콘텐츠 품질 평가 중...")

        total_items = len(metadata)
        items_with_section = sum(1 for item in metadata if item.get('section_title'))
        items_with_keywords = sum(1 for item in metadata if item.get('keywords'))
        items_with_confidence = sum(1 for item in metadata if item.get('confidence'))

        content_lengths = [len(item.get('content', '')) for item in metadata]
        avg_length = np.mean(content_lengths) if content_lengths else 0
        std_length = np.std(content_lengths) if content_lengths else 0

        results = {
            'total_items': total_items,
            'items_with_section': items_with_section,
            'items_with_keywords': items_with_keywords,
            'items_with_confidence': items_with_confidence,
            'section_coverage': items_with_section / total_items if total_items > 0 else 0,
            'keyword_coverage': items_with_keywords / total_items if total_items > 0 else 0,
            'confidence_coverage': items_with_confidence / total_items if total_items > 0 else 0,
            'avg_content_length': avg_length,
            'content_length_std': std_length,
            'min_content_length': str(min(content_lengths)) if content_lengths else "0",
            'max_content_length': str(max(content_lengths)) if content_lengths else "0"
        }

        print(f"[INFO] 콘텐츠 품질 평가 완료")
        print(f"- 총 항목: {total_items}")
        print(f"- 섹션 제목 포함: {items_with_section} ({results['section_coverage']*100:.1f}%)")
        print(f"- 키워드 포함: {items_with_keywords} ({results['keyword_coverage']*100:.1f}%)")
        print(f"- 신뢰도 포함: {items_with_confidence} ({results['confidence_coverage']*100:.1f}%)")
        print(f"- 평균 텍스트 길이: {avg_length:.0f}자")

        return results

    def evaluate_qa_retrieval(self, metadata: List[Dict], vectors: np.ndarray, qa_pairs_file: str = None) -> Dict:
        """질문-답변 검색 품질 평가 (의약품 도메인 특화)"""
        print("[INFO] 질문-답변 검색 품질 평가 중...")

        # QA 쌍 파일이 있으면 사용, 없으면 기본 질문 사용
        if qa_pairs_file and os.path.exists(qa_pairs_file):
            try:
                with open(qa_pairs_file, 'r', encoding='utf-8') as f:
                    qa_data = json.load(f)
                questions = [q.get('question', '') for q in qa_data.get('questions', [])]
                print(f"[INFO] QA 쌍 파일에서 {len(questions)}개 질문 로드")
                used_qa_file = True
            except Exception as e:
                print(f"[WARNING] QA 쌍 파일 로드 실패: {e}")
                questions = self._get_default_questions()
                used_qa_file = False
        else:
            questions = self._get_default_questions()
            print(f"[INFO] 기본 질문 {len(questions)}개 사용")
            used_qa_file = False
        
        results = self._evaluate_questions(questions, metadata, vectors)
        results['used_qa_file'] = used_qa_file
        results['qa_file_path'] = qa_pairs_file if used_qa_file else None
        
        return results

    def evaluate_extrinsic_metrics(self, metadata: List[Dict], vectors: np.ndarray) -> Dict:
        """외재적 평가 지표 (Precision@k, MRR, Exact Match, F1, 샘플 효율성)"""
        print("[INFO] 외재적 평가 지표 계산 중...")
        
        # 1. Precision@k, MRR 계산
        retrieval_metrics = self._calculate_retrieval_metrics(metadata, vectors)
        
        # 2. Exact Match, F1 계산
        qa_accuracy = self._calculate_qa_accuracy(metadata, vectors)
        
        # 3. 샘플 효율성 테스트
        sample_efficiency = self._evaluate_sample_efficiency(metadata, vectors)
        
        results = {
            'retrieval_metrics': retrieval_metrics,
            'qa_accuracy': qa_accuracy,
            'sample_efficiency': sample_efficiency
        }
        
        print(f"[INFO] 외재적 평가 완료")
        print(f"- Precision@1: {retrieval_metrics['precision_at_1']:.4f}")
        print(f"- Precision@3: {retrieval_metrics['precision_at_3']:.4f}")
        print(f"- MRR: {retrieval_metrics['mrr']:.4f}")
        print(f"- Exact Match: {qa_accuracy['exact_match']:.4f}")
        print(f"- F1 Score: {qa_accuracy['f1_score']:.4f}")
        
        return results

    def evaluate_intrinsic_metrics(self, metadata: List[Dict], vectors: np.ndarray) -> Dict:
        """내재적 평가 지표 (의미 유사도 상관관계, 클러스터링 일관성, 유추 QA)"""
        print("[INFO] 내재적 평가 지표 계산 중...")
        
        # 1. 의미 유사도 상관관계 (Spearman)
        semantic_correlation = self._calculate_semantic_correlation(metadata, vectors)
        
        # 2. 클러스터링 일관성 (Purity, NMI)
        clustering_consistency = self._evaluate_clustering_consistency(metadata, vectors)
        
        # 3. 유추 QA 성능
        analogy_performance = self._evaluate_analogy_performance(metadata, vectors)
        
        results = {
            'semantic_correlation': semantic_correlation,
            'clustering_consistency': clustering_consistency,
            'analogy_performance': analogy_performance
        }
        
        print(f"[INFO] 내재적 평가 완료")
        print(f"- 의미 유사도 상관관계: {semantic_correlation['spearman_correlation']:.4f}")
        print(f"- 클러스터 Purity: {clustering_consistency['purity']:.4f}")
        print(f"- 클러스터 NMI: {clustering_consistency['nmi']:.4f}")
        print(f"- 유추 QA 정확도: {analogy_performance['accuracy']:.4f}")
        
        return results

    def _calculate_retrieval_metrics(self, metadata: List[Dict], vectors: np.ndarray) -> Dict:
        """검색 성능 지표 계산 (Precision@k, Recall@k, MRR, MAP)"""
        # 도메인 특화 질문들 (더 다양한 케이스)
        domain_questions = [
            "임상시험 1상 안전성 평가",
            "의약품 심사 주요 항목",
            "가이드라인 핵심 내용",
            "민원 처리 절차",
            "신청서 유효성정보",
            "임상시험 2상 유효성 평가",
            "의약품 승인 기준",
            "평가 기준서",
            "이의제기 절차",
            "신청서 안전성정보"
        ]
        
        precision_at_1 = []
        precision_at_3 = []
        precision_at_5 = []
        recall_at_3 = []
        recall_at_5 = []
        mrr_scores = []
        map_scores = []
        
        for question in domain_questions:
            # 질문 임베딩
            question_embedding = self.model.encode([question])[0]
            
            # 유사도 계산 및 정렬
            similarities = []
            for i, doc_embedding in enumerate(vectors):
                sim = self._calculate_similarity_vec(question_embedding, doc_embedding)
                similarities.append((sim, i))
            
            similarities.sort(reverse=True)
            
            # Precision@k 계산
            top1_relevant = self._is_relevant(metadata[similarities[0][1]], question)
            precision_at_1.append(1.0 if top1_relevant else 0.0)
            
            top3_relevant = sum(1 for _, idx in similarities[:3] 
                              if self._is_relevant(metadata[idx], question))
            precision_at_3.append(top3_relevant / 3)
            
            top5_relevant = sum(1 for _, idx in similarities[:5] 
                              if self._is_relevant(metadata[idx], question))
            precision_at_5.append(top5_relevant / 5)
            
            # Recall@k 계산
            total_relevant = sum(1 for i in range(len(metadata)) 
                               if self._is_relevant(metadata[i], question))
            
            if total_relevant > 0:
                recall_at_3.append(top3_relevant / total_relevant)
                recall_at_5.append(top5_relevant / total_relevant)
            else:
                recall_at_3.append(0.0)
                recall_at_5.append(0.0)
            
            # MRR 계산
            mrr = 0
            for rank, (_, idx) in enumerate(similarities[:10]):
                if self._is_relevant(metadata[idx], question):
                    mrr = 1.0 / (rank + 1)
                    break
            mrr_scores.append(mrr)
            
            # MAP 계산 (Mean Average Precision)
            ap = 0
            relevant_count = 0
            for rank, (_, idx) in enumerate(similarities[:10]):
                if self._is_relevant(metadata[idx], question):
                    relevant_count += 1
                    ap += relevant_count / (rank + 1)
            
            if total_relevant > 0:
                map_scores.append(ap / total_relevant)
            else:
                map_scores.append(0.0)
        
        return {
            'precision_at_1': np.mean(precision_at_1),
            'precision_at_3': np.mean(precision_at_3),
            'precision_at_5': np.mean(precision_at_5),
            'recall_at_3': np.mean(recall_at_3),
            'recall_at_5': np.mean(recall_at_5),
            'mrr': np.mean(mrr_scores),
            'map': np.mean(map_scores),
            'total_queries': len(domain_questions)
        }

    def _calculate_qa_accuracy(self, metadata: List[Dict], vectors: np.ndarray) -> Dict:
        """QA 정확도 계산 (완전 동적 방식)"""
        
        # 1. 동적 QA 쌍 생성 (문서 내용 기반)
        qa_pairs = self._generate_dynamic_qa_pairs(metadata)
        
        if not qa_pairs:
            # QA 쌍이 생성되지 않으면 기본값 반환
            return {
                'semantic_cohesion': 0.0,
                'discriminative_power': 0.0,
                'keyword_matching': 0.0,
                'answer_inclusion': 0.0,
                'top_k_accuracy': 0.0,
                'exact_match': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'total_qa_pairs': 0
            }
        
        # 2. 다양한 평가 방식 적용
        results = {
            'semantic_cohesion': [],
            'discriminative_power': [],
            'keyword_matching': [],
            'answer_inclusion': [],
            'top_k_accuracy': [],
            'exact_match': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'semantic_similarity': [],
            'content_relevance': []
        }
        
        for question, expected_answer in qa_pairs:
            # 질문과 정답 임베딩
            question_embedding = self.model.encode([question])[0]
            expected_embedding = self.model.encode([expected_answer])[0]
            
            # Top-K 검색 (K=5)
            similarities = []
            for i, doc_embedding in enumerate(vectors):
                sim = self._calculate_similarity_vec(question_embedding, doc_embedding)
                similarities.append((sim, i))
            
            similarities.sort(reverse=True)
            
            # 1. Semantic Cohesion (질문-문서 유사도)
            best_doc_idx = similarities[0][1]
            best_doc_embedding = vectors[best_doc_idx]
            question_doc_sim = self._calculate_similarity_vec(question_embedding, best_doc_embedding)
            results['semantic_cohesion'].append(question_doc_sim)
            
            # 2. Discriminative Power (정답-문서 유사도)
            answer_doc_sim = self._calculate_similarity_vec(expected_embedding, best_doc_embedding)
            results['discriminative_power'].append(answer_doc_sim)
            
            # 3. 키워드 매칭 (정답 키워드가 문서에 포함되는지)
            keyword_score = self._calculate_keyword_matching(expected_answer, metadata[best_doc_idx])
            results['keyword_matching'].append(keyword_score)
            
            # 4. 정답 포함 여부 (정답이 문서 내용에 포함되는지)
            answer_inclusion = self._check_answer_inclusion(expected_answer, metadata[best_doc_idx])
            results['answer_inclusion'].append(answer_inclusion)
            
            # 5. Top-K 정확도 (Top-5 내에 정답이 있는지)
            top_k_accuracy = self._calculate_top_k_accuracy(expected_answer, similarities[:5], metadata)
            results['top_k_accuracy'].append(top_k_accuracy)
            
            # 6. Exact Match (정확한 일치)
            exact_match = 1.0 if answer_inclusion > 0.8 else 0.0
            results['exact_match'].append(exact_match)
            
            # 7. Content Relevance (내용 관련성)
            content_relevance = self._calculate_content_relevance(question, expected_answer, metadata[best_doc_idx])
            results['content_relevance'].append(content_relevance)
            
            # 8. Precision/Recall/F1 계산 (개선된 방식)
            precision = (keyword_score + answer_inclusion + content_relevance) / 3
            recall = top_k_accuracy
            f1 = self._calculate_f1_score_from_pr(precision, recall)
            
            results['precision'].append(precision)
            results['recall'].append(recall)
            results['f1_score'].append(f1)
            
            # 9. Semantic Similarity (전체적인 의미적 유사도)
            semantic_sim = (question_doc_sim + answer_doc_sim) / 2
            results['semantic_similarity'].append(semantic_sim)
        
        # 평균 계산
        avg_results = {}
        for key in results:
            avg_results[key] = np.mean(results[key])
        
        return {
            'semantic_cohesion': avg_results['semantic_cohesion'],
            'discriminative_power': avg_results['discriminative_power'],
            'keyword_matching': avg_results['keyword_matching'],
            'answer_inclusion': avg_results['answer_inclusion'],
            'top_k_accuracy': avg_results['top_k_accuracy'],
            'exact_match': avg_results['exact_match'],
            'precision': avg_results['precision'],
            'recall': avg_results['recall'],
            'f1_score': avg_results['f1_score'],
            'semantic_similarity': avg_results['semantic_similarity'],
            'content_relevance': avg_results['content_relevance'],
            'total_qa_pairs': len(qa_pairs)
        }
    
    def _calculate_content_relevance(self, question: str, expected_answer: str, doc: Dict) -> float:
        """질문과 정답의 내용 관련성 계산"""
        doc_content = f"{doc.get('content', '')} {doc.get('section_title', '')}"
        
        # 질문 키워드 추출
        question_keywords = self._extract_keywords_improved(question)
        answer_keywords = self._extract_keywords_improved(expected_answer)
        
        # 문서에서 관련 키워드 찾기
        doc_lower = doc_content.lower()
        
        question_relevance = 0.0
        answer_relevance = 0.0
        
        if question_keywords:
            question_matches = sum(1 for kw in question_keywords if kw.lower() in doc_lower)
            question_relevance = question_matches / len(question_keywords)
        
        if answer_keywords:
            answer_matches = sum(1 for kw in answer_keywords if kw.lower() in doc_lower)
            answer_relevance = answer_matches / len(answer_keywords)
        
        # 질문과 정답 관련성의 평균
        return (question_relevance + answer_relevance) / 2
    
    def _generate_dynamic_qa_pairs(self, metadata: List[Dict]) -> List[Tuple[str, str]]:
        """문서 내용 기반 완전 동적 QA 쌍 생성"""
        qa_pairs = []
        
        for doc in metadata:
            content = doc.get('content', '')
            section = doc.get('section_title', '')
            keywords = doc.get('keywords', [])
            
            if not content or len(content) < 30:  # 더 짧은 내용도 허용
                continue
            
            # 1. 키워드 기반 질문 생성 (더 많은 패턴)
            for keyword in keywords[:8]:  # 상위 8개 키워드로 증가
                if len(keyword) > 1:  # 더 짧은 키워드도 허용
                    # 다양한 질문 패턴 생성
                    questions = [
                        f"{keyword}에 대해 설명해주세요",
                        f"{keyword}의 주요 특징은?",
                        f"{keyword}와 관련된 내용은?",
                        f"{keyword}에서 중요한 점은?",
                        f"{keyword}의 정의는?",
                        f"{keyword}에 대한 정보를 알려주세요",
                        f"{keyword}의 역할은?",
                        f"{keyword}의 특징은?"
                    ]
                    
                    # 키워드가 포함된 문장 찾기
                    sentences = self._extract_sentences_with_keyword(content, keyword)
                    if sentences:
                        answer = sentences[0][:120]  # 답변 길이 증가
                        for question in questions:
                            qa_pairs.append((question, answer))
            
            # 2. 섹션 제목 기반 질문 생성 (더 많은 패턴)
            if section and len(section) > 2:
                section_questions = [
                    f"{section}에 대한 정보를 알려주세요",
                    f"{section}의 주요 내용은?",
                    f"{section}에서 다루는 내용은?",
                    f"{section}에 대해 설명해주세요",
                    f"{section}의 특징은?",
                    f"{section}의 정의는?"
                ]
                
                # 섹션 내용에서 답변 추출
                section_answer = self._extract_section_answer(content, section)
                if section_answer:
                    for question in section_questions:
                        qa_pairs.append((question, section_answer))
            
            # 3. 내용 기반 자동 질문 생성
            content_qa = self._generate_content_based_qa(content, keywords)
            qa_pairs.extend(content_qa)
            
            # 4. 문장 기반 질문 생성 (새로 추가)
            sentences = self._extract_sentences_from_content(content)
            for sentence in sentences[:5]:  # 상위 5개 문장
                if len(sentence) > 20:
                    question = f"다음 내용에 대해 설명해주세요: {sentence[:50]}..."
                    qa_pairs.append((question, sentence[:150]))
        
        # 중복 제거 및 최대 50개로 증가
        unique_qa_pairs = []
        seen_questions = set()
        
        for question, answer in qa_pairs:
            # 질문을 정규화하여 중복 체크
            normalized_question = question.lower().strip()
            if normalized_question not in seen_questions and len(answer) > 5:
                unique_qa_pairs.append((question, answer))
                seen_questions.add(normalized_question)
                
                if len(unique_qa_pairs) >= 50:  # 최대 개수 증가
                    break
        
        return unique_qa_pairs
    
    def _extract_sentences_from_content(self, content: str) -> List[str]:
        """내용에서 문장들 추출"""
        import re
        # 문장 단위로 분리 (더 정교한 패턴)
        sentence_pattern = r'[^.!?]*[.!?]'
        sentences = re.findall(sentence_pattern, content)
        
        # 의미있는 문장만 필터링
        meaningful_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10 and not sentence.isdigit():
                meaningful_sentences.append(sentence)
        
        return meaningful_sentences
    
    def _extract_sentences_with_keyword(self, content: str, keyword: str) -> List[str]:
        """키워드가 포함된 문장들 추출"""
        sentences = []
        content_lower = content.lower()
        keyword_lower = keyword.lower()
        
        # 문장 단위로 분리
        import re
        sentence_pattern = r'[^.!?]*[.!?]'
        all_sentences = re.findall(sentence_pattern, content)
        
        for sentence in all_sentences:
            if keyword_lower in sentence.lower():
                sentences.append(sentence.strip())
        
        return sentences
    
    def _extract_section_answer(self, content: str, section: str) -> str:
        """섹션 제목에 해당하는 답변 추출"""
        # 섹션 제목과 관련된 내용 찾기
        sentences = self._extract_sentences_with_keyword(content, section)
        if sentences:
            return sentences[0][:150]  # 첫 번째 문장을 답변으로
        
        # 섹션 제목의 키워드로 검색
        section_keywords = self._extract_keywords_improved(section)
        for keyword in section_keywords:
            sentences = self._extract_sentences_with_keyword(content, keyword)
            if sentences:
                return sentences[0][:150]
        
        return content[:150]  # 기본값으로 내용의 앞부분 사용
    
    def _generate_content_based_qa(self, content: str, keywords: List[str]) -> List[Tuple[str, str]]:
        """내용 기반 자동 QA 생성"""
        qa_pairs = []
        
        # 주요 키워드 조합으로 질문 생성
        if len(keywords) >= 2:
            for i in range(len(keywords) - 1):
                keyword1 = keywords[i]
                keyword2 = keywords[i + 1]
                
                if len(keyword1) > 1 and len(keyword2) > 1:  # 더 짧은 키워드도 허용
                    question = f"{keyword1}와 {keyword2}의 관계는?"
                    
                    # 두 키워드가 모두 포함된 문장 찾기
                    sentences = self._extract_sentences_with_keywords(content, [keyword1, keyword2])
                    if sentences:
                        answer = sentences[0][:120]  # 답변 길이 증가
                        qa_pairs.append((question, answer))
        
        # 내용의 주요 개념으로 질문 생성
        important_concepts = self._extract_important_concepts(content)
        for concept in important_concepts[:5]:  # 상위 5개 개념으로 증가
            question = f"{concept}에 대해 설명해주세요"
            sentences = self._extract_sentences_with_keyword(content, concept)
            if sentences:
                answer = sentences[0][:120]
                qa_pairs.append((question, answer))
        
        # 추가 질문 패턴 생성
        for keyword in keywords[:5]:
            if len(keyword) > 1:
                additional_questions = [
                    f"{keyword}의 중요성은?",
                    f"{keyword}의 역할과 기능은?",
                    f"{keyword}에 대한 상세한 설명을 해주세요"
                ]
                
                sentences = self._extract_sentences_with_keyword(content, keyword)
                if sentences:
                    answer = sentences[0][:120]
                    for question in additional_questions:
                        qa_pairs.append((question, answer))
        
        return qa_pairs
    
    def _extract_sentences_with_keywords(self, content: str, keywords: List[str]) -> List[str]:
        """여러 키워드가 모두 포함된 문장들 추출"""
        sentences = []
        content_lower = content.lower()
        
        # 문장 단위로 분리
        import re
        sentence_pattern = r'[^.!?]*[.!?]'
        all_sentences = re.findall(sentence_pattern, content)
        
        for sentence in all_sentences:
            sentence_lower = sentence.lower()
            if all(keyword.lower() in sentence_lower for keyword in keywords):
                sentences.append(sentence.strip())
        
        return sentences
    
    def _extract_important_concepts(self, content: str) -> List[str]:
        """내용에서 중요한 개념 추출"""
        # 키워드 추출
        keywords = self._extract_keywords_improved(content)
        
        # 빈도 기반으로 중요도 계산
        word_freq = {}
        for keyword in keywords:
            if keyword in word_freq:
                word_freq[keyword] += 1
            else:
                word_freq[keyword] = 1
        
        # 빈도순으로 정렬
        sorted_concepts = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # 상위 8개로 증가하고, 빈도가 1 이상인 것만 선택
        important_concepts = []
        for concept, freq in sorted_concepts[:8]:
            if freq >= 1 and len(concept) > 1:  # 빈도 1 이상이고 길이 2 이상
                important_concepts.append(concept)
        
        return important_concepts
    

    
    def _calculate_keyword_matching(self, answer: str, doc: Dict) -> float:
        """정답 키워드가 문서에 포함되는 정도 계산"""
        answer_keywords = self._extract_keywords_improved(answer)
        doc_text = f"{doc.get('content', '')} {doc.get('section_title', '')} {' '.join(doc.get('keywords', []))}"
        
        if not answer_keywords:
            return 0.0
        
        matched_keywords = 0
        for keyword in answer_keywords:
            if keyword in doc_text:
                matched_keywords += 1
        
        return matched_keywords / len(answer_keywords)
    
    def _check_answer_inclusion(self, answer: str, doc: Dict) -> float:
        """정답이 문서 내용에 포함되는 정도 확인"""
        doc_text = f"{doc.get('content', '')} {doc.get('section_title', '')}"
        answer_lower = answer.lower()
        doc_lower = doc_text.lower()
        
        # 부분 문자열 매칭
        if answer_lower in doc_lower:
            return 1.0
        
        # 키워드 기반 매칭
        answer_keywords = self._extract_keywords_improved(answer)
        if not answer_keywords:
            return 0.0
        
        matched_keywords = 0
        for keyword in answer_keywords:
            if keyword in doc_lower:
                matched_keywords += 1
        
        return matched_keywords / len(answer_keywords)
    
    def _calculate_top_k_accuracy(self, expected_answer: str, top_k_similarities: List[Tuple[float, int]], metadata: List[Dict]) -> float:
        """Top-K 검색 결과에서 정답 포함 여부 확인"""
        answer_keywords = self._extract_keywords_improved(expected_answer)
        
        if not answer_keywords:
            return 0.0
        
        for _, doc_idx in top_k_similarities:
            doc = metadata[doc_idx]
            doc_text = f"{doc.get('content', '')} {doc.get('section_title', '')}"
            
            # 키워드 매칭 확인
            matched_keywords = 0
            for keyword in answer_keywords:
                if keyword in doc_text.lower():
                    matched_keywords += 1
            
            if matched_keywords / len(answer_keywords) > 0.5:  # 50% 이상 매칭
                return 1.0
        
        return 0.0

    def _evaluate_sample_efficiency(self, metadata: List[Dict], vectors: np.ndarray) -> Dict:
        """샘플 효율성 평가 (소수 N 샘플 실험)"""
        sample_sizes = [10, 20, 50, 100]  # 테스트할 샘플 크기들
        performance_trend = {}
        
        for sample_size in sample_sizes:
            if sample_size >= len(metadata):
                continue
                
            # 랜덤 샘플링
            indices = np.random.choice(len(metadata), sample_size, replace=False)
            sample_metadata = [metadata[i] for i in indices]
            sample_vectors = vectors[indices]
            
            # 샘플에서 성능 측정
            sample_retrieval = self._calculate_retrieval_metrics(sample_metadata, sample_vectors)
            sample_qa = self._calculate_qa_accuracy(sample_metadata, sample_vectors)
            
            performance_trend[sample_size] = {
                'precision_at_1': sample_retrieval['precision_at_1'],
                'precision_at_3': sample_retrieval['precision_at_3'],
                'mrr': sample_retrieval['mrr'],
                'exact_match': sample_qa['exact_match'],
                'f1_score': sample_qa['f1_score']
            }
        
        return {
            'performance_trend': performance_trend,
            'efficiency_score': self._calculate_efficiency_score(performance_trend)
        }

    def _calculate_semantic_correlation(self, metadata: List[Dict], vectors: np.ndarray) -> Dict:
        """의미 유사도 상관관계 계산 (Spearman)"""
        from scipy.stats import spearmanr
        
        # 도메인 전문가 점수 시뮬레이션 (의약품 도메인 특화)
        expert_scores = []
        embedding_similarities = []
        
        # 의약품 도메인 문장 쌍들
        sentence_pairs = [
            ("임상시험 1상 안전성 평가", "임상시험 2상 유효성 평가"),
            ("의약품 심사 주요 항목", "의약품 승인 기준"),
            ("가이드라인 핵심 내용", "평가 기준서"),
            ("민원 처리 절차", "이의제기 절차"),
            ("신청서 유효성정보", "신청서 안전성정보")
        ]
        
        for sent1, sent2 in sentence_pairs:
            # 전문가 점수 시뮬레이션 (의약품 도메인 지식 기반)
            expert_score = self._simulate_expert_score(sent1, sent2)
            expert_scores.append(expert_score)
            
            # 임베딩 유사도 계산
            emb1 = self.model.encode([sent1])[0]
            emb2 = self.model.encode([sent2])[0]
            embedding_sim = self._calculate_similarity_vec(emb1, emb2)
            embedding_similarities.append(embedding_sim)
        
        # Spearman 상관계수 계산
        if len(expert_scores) > 1:
            correlation, p_value = spearmanr(expert_scores, embedding_similarities)
        else:
            correlation, p_value = 0, 1
        
        return {
            'spearman_correlation': correlation,
            'p_value': p_value,
            'expert_scores': expert_scores,
            'embedding_similarities': embedding_similarities
        }

    def _evaluate_clustering_consistency(self, metadata: List[Dict], vectors: np.ndarray) -> Dict:
        """클러스터링 일관성 평가 (동적 라벨링)"""
        from sklearn.cluster import KMeans
        from sklearn.metrics import normalized_mutual_info_score
        
        # 동적 라벨 생성 (문서 내용 기반)
        true_labels = self._generate_dynamic_labels(metadata)
        
        # K-means 클러스터링 (의미 기반 자동 클러스터 수 결정)
        n_clusters = len(set(true_labels))  # 실제 의미적 라벨 개수만큼 클러스터 설정
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        predicted_labels = kmeans.fit_predict(vectors)
        
        # Purity 계산
        purity = self._calculate_purity(true_labels, predicted_labels)
        
        # NMI 계산
        nmi = normalized_mutual_info_score(true_labels, predicted_labels)
        
        # 클러스터별 라벨 분포 분석
        cluster_label_distribution = self._analyze_cluster_label_distribution(true_labels, predicted_labels, n_clusters)
        
        return {
            'purity': purity,
            'nmi': nmi,
            'n_clusters': n_clusters,
            'cluster_distribution': cluster_label_distribution
        }

    def _generate_dynamic_labels(self, metadata: List[Dict]) -> List[int]:
        """동적 라벨 생성 (문서 내용 기반)"""
        from collections import Counter
        import re
        
        # 1단계: 문서에서 주요 키워드 추출
        all_keywords = []
        for item in metadata:
            content = item.get('content', '').lower()
            keywords = item.get('keywords', [])
            all_keywords.extend(keywords)
            # 추가 키워드 추출 (간단한 형태소 분석)
            words = re.findall(r'\b\w+\b', content)
            all_keywords.extend([w for w in words if len(w) > 2])
        
        # 2단계: 키워드 빈도 분석으로 주요 도메인 파악
        keyword_freq = Counter(all_keywords)
        top_keywords = [kw for kw, freq in keyword_freq.most_common(20) if freq > 1]
        
        # 3단계: 도메인별 키워드 패턴 정의 (동적)
        domain_patterns = self._identify_domain_patterns(top_keywords, metadata)
        
        # 4단계: 각 문서에 동적 라벨 할당
        true_labels = []
        for item in metadata:
            content = item.get('content', '').lower()
            section = item.get('section_title', '').lower()
            keywords = item.get('keywords', [])
            
            # 가장 적합한 도메인 찾기
            best_domain = self._find_best_domain(content, section, keywords, domain_patterns)
            true_labels.append(best_domain)
        
        return true_labels

    def _identify_domain_patterns(self, top_keywords: List[str], metadata: List[Dict]) -> Dict:
        """도메인 패턴 자동 식별"""
        patterns = {}
        domain_id = 0
        
        # 키워드 기반 도메인 그룹화
        keyword_groups = self._group_keywords_by_similarity(top_keywords)
        
        for group in keyword_groups:
            if len(group) >= 2:  # 최소 2개 이상의 관련 키워드가 있는 그룹만
                patterns[domain_id] = {
                    'keywords': group,
                    'name': self._generate_domain_name(group),
                    'priority': len(group)  # 키워드 수가 많을수록 우선순위 높음
                }
                domain_id += 1
        
        # 기본 도메인 추가 (키워드가 적은 경우)
        if not patterns:
            patterns[0] = {
                'keywords': ['일반'],
                'name': '일반',
                'priority': 1
            }
        
        return patterns

    def _group_keywords_by_similarity(self, keywords: List[str]) -> List[List[str]]:
        """의미적 유사도 기반 키워드 그룹화"""
        groups = []
        used_keywords = set()
        
        for i, kw1 in enumerate(keywords):
            if kw1 in used_keywords:
                continue
                
            group = [kw1]
            used_keywords.add(kw1)
            
            # 유사한 키워드 찾기
            for j, kw2 in enumerate(keywords[i+1:], i+1):
                if kw2 in used_keywords:
                    continue
                    
                # 간단한 유사도 체크 (접두사, 접미사, 포함 관계)
                if self._are_keywords_similar(kw1, kw2):
                    group.append(kw2)
                    used_keywords.add(kw2)
            
            if len(group) >= 1:
                groups.append(group)
        
        return groups

    def _are_keywords_similar(self, kw1: str, kw2: str) -> bool:
        """키워드 유사도 체크"""
        # 접두사/접미사 공통성
        if kw1.startswith(kw2[:3]) or kw2.startswith(kw1[:3]):
            return True
        
        # 포함 관계
        if kw1 in kw2 or kw2 in kw1:
            return True
        
        # 길이 차이가 1 이하인 경우
        if abs(len(kw1) - len(kw2)) <= 1:
            return True
        
        return False

    def _generate_domain_name(self, keywords: List[str]) -> str:
        """도메인 이름 자동 생성"""
        if not keywords:
            return "기타"
        
        # 가장 긴 키워드를 도메인 이름으로 사용
        longest_kw = max(keywords, key=len)
        return longest_kw

    def _find_best_domain(self, content: str, section: str, keywords: List[str], 
                          domain_patterns: Dict) -> int:
        """가장 적합한 도메인 찾기"""
        best_domain = 0
        best_score = 0
        
        for domain_id, pattern in domain_patterns.items():
            score = 0
            pattern_keywords = pattern['keywords']
            
            # 키워드 매칭 점수
            for kw in keywords:
                if any(pk in kw.lower() for pk in pattern_keywords):
                    score += 2
            
            # 내용 매칭 점수
            for pk in pattern_keywords:
                if pk in content or pk in section:
                    score += 1
            
            # 우선순위 보너스
            score += pattern['priority']
            
            if score > best_score:
                best_score = score
                best_domain = domain_id
        
        return best_domain

    def _calculate_purity(self, true_labels: List[int], predicted_labels: List[int]) -> float:
        """클러스터링 Purity 계산"""
        from collections import defaultdict
        
        cluster_assignments = defaultdict(list)
        for i, pred_label in enumerate(predicted_labels):
            cluster_assignments[pred_label].append(true_labels[i])
        
        total_purity = 0
        for cluster in cluster_assignments.values():
            if cluster:
                most_common = max(set(cluster), key=cluster.count)
                purity = cluster.count(most_common) / len(cluster)
                total_purity += purity * len(cluster)
        
        return total_purity / len(true_labels) if true_labels else 0

    def _analyze_cluster_label_distribution(self, true_labels: List[int], predicted_labels: List[int], n_clusters: int) -> Dict:
        """클러스터별 라벨 분포 분석"""
        from collections import Counter, defaultdict
        
        # 클러스터별 라벨 분포
        cluster_distribution = defaultdict(list)
        for i, pred_label in enumerate(predicted_labels):
            cluster_distribution[pred_label].append(true_labels[i])
        
        # 상세 분석 결과
        analysis = {
            'cluster_details': {},
            'label_names': {
                0: '임상시험',
                1: '심사', 
                2: '가이드라인',
                3: '민원',
                4: '신청서',
                5: '기타'
            },
            'total_samples': len(true_labels),
            'cluster_sizes': {},
            'dominant_labels': {},
            'mixing_degree': {}
        }
        
        for cluster_id in range(n_clusters):
            if cluster_id in cluster_distribution:
                labels = cluster_distribution[cluster_id]
                label_counts = Counter(labels)
                total_in_cluster = len(labels)
                
                # 가장 많은 라벨과 비율
                if label_counts:
                    dominant_label, dominant_count = label_counts.most_common(1)[0]
                    dominant_ratio = dominant_count / total_in_cluster
                    
                    # 혼합 정도 (1 - 가장 많은 라벨 비율)
                    mixing_degree = 1 - dominant_ratio
                    
                    analysis['cluster_details'][cluster_id] = {
                        'size': total_in_cluster,
                        'label_distribution': dict(label_counts),
                        'dominant_label': dominant_label,
                        'dominant_label_name': analysis['label_names'].get(dominant_label, f'라벨{dominant_label}'),
                        'dominant_ratio': dominant_ratio,
                        'mixing_degree': mixing_degree,
                        'purity': dominant_ratio
                    }
                    
                    analysis['cluster_sizes'][cluster_id] = total_in_cluster
                    analysis['dominant_labels'][cluster_id] = dominant_label
                    analysis['mixing_degree'][cluster_id] = mixing_degree
        
        # 전체 혼합도 계산
        total_mixing = sum(analysis['mixing_degree'].values()) / len(analysis['mixing_degree']) if analysis['mixing_degree'] else 0
        analysis['overall_mixing_degree'] = total_mixing
        
        return analysis

    def _evaluate_analogy_performance(self, metadata: List[Dict], vectors: np.ndarray) -> Dict:
        """유추 QA 성능 평가 (도메인 내부 연산)"""
        # 의약품 도메인 유추 문제들
        analogy_problems = [
            {
                'a': '신청서',
                'b': '유효성정보', 
                'c': '안전성정보',
                'question': '신청서 - 유효성정보 + 안전성정보 ≈ ?'
            },
            {
                'a': '임상시험1상',
                'b': '안전성',
                'c': '유효성',
                'question': '임상시험1상 - 안전성 + 유효성 ≈ ?'
            },
            {
                'a': '심사부',
                'b': '평가',
                'c': '승인',
                'question': '심사부 - 평가 + 승인 ≈ ?'
            }
        ]
        
        correct_answers = 0
        total_problems = len(analogy_problems)
        
        for problem in analogy_problems:
            # 벡터 연산: a - b + c
            a_vec = self._get_concept_vector(problem['a'], metadata, vectors)
            b_vec = self._get_concept_vector(problem['b'], metadata, vectors)
            c_vec = self._get_concept_vector(problem['c'], metadata, vectors)
            
            if a_vec is not None and b_vec is not None and c_vec is not None:
                analogy_vector = a_vec - b_vec + c_vec
                
                # 가장 유사한 개념 찾기
                similarities = []
                for i, doc_embedding in enumerate(vectors):
                    sim = self._calculate_similarity_vec(analogy_vector, doc_embedding)
                    similarities.append((sim, i))
                
                similarities.sort(reverse=True)
                best_match = metadata[similarities[0][1]]
                
                # 정답 여부 판단 (간단한 키워드 매칭)
                if self._is_analogy_correct(best_match, problem):
                    correct_answers += 1
        
        accuracy = correct_answers / total_problems if total_problems > 0 else 0
        
        return {
            'accuracy': accuracy,
            'correct_answers': correct_answers,
            'total_problems': total_problems
        }

    # 헬퍼 메서드들
    def _is_relevant(self, doc: Dict, question: str) -> bool:
        """문서가 질문과 관련있는지 판단"""
        content = doc.get('content', '').lower()
        section = doc.get('section_title', '').lower()
        
        # 간단한 키워드 매칭
        question_keywords = question.lower().split()
        doc_text = content + ' ' + section
        
        return any(keyword in doc_text for keyword in question_keywords)

    def _extract_keywords(self, text: str) -> List[str]:
        """텍스트에서 키워드 추출 (한국어 특화)"""
        import re
        
        # 의약품 도메인 특화 키워드 패턴
        domain_keywords = [
            '임상시험', '안전성', '유효성', '심사', '승인', '가이드라인', '민원', '신청서',
            'PK', 'PD', 'EBA', 'AFB', 'M.tuberculosis', '폐결핵', '결핵', '치료제',
            '1상', '2상', '3상', '4상', '단계', '평가', '검토', '절차', '기준',
            '약물', '약제', '의약품', '개발', '연구', '시험', '분석', '통계',
            '환자', '집단', '대상', '등록', '관찰', '추적', '결과', '효과',
            '부작용', '독성', '약물동태', '약물역학', '생체이용률', '반감기'
        ]
        
        # 텍스트에서 도메인 키워드 찾기
        found_keywords = []
        text_lower = text.lower()
        
        for keyword in domain_keywords:
            if keyword.lower() in text_lower:
                found_keywords.append(keyword)
        
        # 일반적인 한글 키워드 추출 (2글자 이상)
        korean_keywords = re.findall(r'[가-힣]{2,}', text)
        found_keywords.extend(korean_keywords[:10])  # 상위 10개만
        
        # 영문 키워드 추출 (3글자 이상)
        english_keywords = re.findall(r'[a-zA-Z]{3,}', text.lower())
        found_keywords.extend(english_keywords[:5])  # 상위 5개만
        
        return list(set(found_keywords))  # 중복 제거
    
    def _extract_keywords_improved(self, text: str) -> List[str]:
        """개선된 키워드 추출 (메타데이터 기반)"""
        import re
        
        # 메타데이터에서 키워드 추출 (이미 있는 키워드 활용)
        # 이 함수는 개별 텍스트에 대해 호출되므로, 
        # 메타데이터의 키워드와 텍스트 내용을 결합하여 사용
        
        # 텍스트에서 한글 키워드 추출
        korean_keywords = re.findall(r'[가-힣]{2,}', text)
        # 의미있는 단어만 필터링
        meaningful_korean = [word for word in korean_keywords if len(word) >= 2 and word not in ['그리고', '또는', '하지만', '그러나', '따라서']]
        
        # 영문 키워드 추출 (3글자 이상)
        english_keywords = re.findall(r'[a-zA-Z]{3,}', text.lower())
        meaningful_english = [word for word in english_keywords if word not in ['the', 'and', 'for', 'are', 'was', 'were', 'this', 'that', 'with', 'from']]
        
        # 모든 키워드 결합
        all_keywords = meaningful_korean + meaningful_english
        
        # 중복 제거 시 안전한 방법 사용
        unique_keywords = []
        seen_keywords = set()
        for keyword in all_keywords:
            if keyword not in seen_keywords:
                unique_keywords.append(keyword)
                seen_keywords.add(keyword)
        
        return unique_keywords[:15]  # 상위 15개 반환
    
    def _extract_domain_keywords_with_llm(self, text: str) -> List[str]:
        """도메인 키워드 추출 (사용하지 않음)"""
        # 이 함수는 더 이상 사용하지 않음
        return []
    
    def _parse_llm_keywords(self, response: str) -> List[str]:
        """LLM 키워드 파싱 (사용하지 않음)"""
        # 이 함수는 더 이상 사용하지 않음
        return []
    
    def _get_fallback_keywords(self, text: str) -> List[str]:
        """기본 키워드 추출"""
        import re
        from collections import Counter
        
        # 한글 단어 추출
        korean_words = re.findall(r'[가-힣]{2,}', text)
        word_freq = Counter(korean_words)
        
        # 빈도순으로 정렬하여 상위 키워드 반환
        top_keywords = [word for word, freq in word_freq.most_common(10) if freq > 1]
        
        return top_keywords

    def _calculate_precision(self, predicted: List[str], expected: List[str]) -> float:
        """Precision 계산"""
        if not predicted:
            return 0.0
        correct = len(set(predicted) & set(expected))
        return correct / len(predicted)

    def _calculate_recall(self, predicted: List[str], expected: List[str]) -> float:
        """Recall 계산"""
        if not expected:
            return 0.0
        correct = len(set(predicted) & set(expected))
        return correct / len(expected)
    
    def _calculate_f1_score(self, predicted: List[str], expected: List[str]) -> float:
        """F1-score 계산"""
        precision = self._calculate_precision(predicted, expected)
        recall = self._calculate_recall(predicted, expected)
        
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    def _calculate_f1_score_from_pr(self, precision: float, recall: float) -> float:
        """Precision과 Recall로부터 F1-score 계산"""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    def _calculate_accuracy(self, predicted: List[str], expected: List[str]) -> float:
        """Accuracy 계산"""
        if not predicted and not expected:
            return 1.0
        if not predicted or not expected:
            return 0.0
        
        correct = len(set(predicted) & set(expected))
        total = len(set(predicted) | set(expected))
        return correct / total if total > 0 else 0.0

    def _simulate_expert_score(self, sent1: str, sent2: str) -> float:
        """도메인 전문가 점수 시뮬레이션"""
        # 의약품 도메인 지식 기반 점수
        common_terms = set(sent1.lower().split()) & set(sent2.lower().split())
        if len(common_terms) > 0:
            return 0.8  # 높은 유사도
        elif any(term in sent1.lower() for term in ['임상시험', '심사', '가이드라인']) and \
             any(term in sent2.lower() for term in ['임상시험', '심사', '가이드라인']):
            return 0.6  # 중간 유사도
        else:
            return 0.2  # 낮은 유사도

    def _calculate_purity(self, true_labels: List[int], predicted_labels: List[int]) -> float:
        """클러스터링 Purity 계산"""
        from collections import defaultdict
        
        cluster_assignments = defaultdict(list)
        for i, pred_label in enumerate(predicted_labels):
            cluster_assignments[pred_label].append(true_labels[i])
        
        total_purity = 0
        for cluster in cluster_assignments.values():
            if cluster:
                most_common = max(set(cluster), key=cluster.count)
                purity = cluster.count(most_common) / len(cluster)
                total_purity += purity * len(cluster)
        
        return total_purity / len(true_labels) if true_labels else 0

    def _get_concept_vector(self, concept: str, metadata: List[Dict], vectors: np.ndarray) -> np.ndarray:
        """개념의 평균 벡터 계산"""
        concept_docs = []
        for i, item in enumerate(metadata):
            content = item.get('content', '').lower()
            section = item.get('section_title', '').lower()
            if concept.lower() in content or concept.lower() in section:
                concept_docs.append(vectors[i])
        
        if concept_docs:
            return np.mean(concept_docs, axis=0)
        return None

    def _is_analogy_correct(self, best_match: Dict, problem: Dict) -> bool:
        """유추 문제의 정답 여부 판단"""
        content = best_match.get('content', '').lower()
        section = best_match.get('section_title', '').lower()
        doc_text = content + ' ' + section
        
        # 간단한 키워드 매칭으로 정답 판단
        expected_concepts = [problem['a'], problem['b'], problem['c']]
        return any(concept.lower() in doc_text for concept in expected_concepts)

    def _calculate_efficiency_score(self, performance_trend: Dict) -> float:
        """샘플 효율성 점수 계산"""
        if not performance_trend:
            return 0.0
        
        # 성능 향상률 계산
        scores = []
        sample_sizes = sorted(performance_trend.keys())
        
        for i in range(1, len(sample_sizes)):
            prev_size = sample_sizes[i-1]
            curr_size = sample_sizes[i]
            
            prev_score = performance_trend[prev_size]['precision_at_1']
            curr_score = performance_trend[curr_size]['precision_at_1']
            
            if prev_score > 0:
                improvement = (curr_score - prev_score) / prev_score
                scores.append(improvement)
        
        return np.mean(scores) if scores else 0.0
    
    def _get_default_questions(self) -> List[str]:
        """기본 질문 목록 반환"""
        return [
            # 1. 임상시험 관련 질문들
            "임상시험 1상에서 안전성 평가는 어떻게 진행되나요?",
            "임상시험 2상에서 유효성 평가 기준은 무엇인가요?",
            "임상시험 3상에서 승인을 위한 주요 지표는 무엇인가요?",
            "임상시험 중단 기준에는 어떤 것들이 있나요?",
            
            # 2. 의약품 심사 관련 질문들
            "의약품 심사부에서 심사하는 주요 항목은 무엇인가요?",
            "종양약품과의 심사 프로세스는 어떻게 되나요?",
            "의약품 승인 시 필요한 서류는 무엇인가요?",
            "심사 과정에서 추가 자료 요청 기준은 무엇인가요?",
            
            # 3. 가이드라인 관련 질문들
            "폐결핵 치료제 임상시험 가이드라인의 핵심 내용은 무엇인가요?",
            "평가 가이드라인에서 중점적으로 다루는 부분은 무엇인가요?",
            "가이드라인 준수 여부는 어떻게 확인하나요?",
            "가이드라인 위반 시 조치사항은 무엇인가요?",
            
            # 4. 민원/안내 관련 질문들
            "민원인 안내서에는 어떤 정보가 포함되어 있나요?",
            "민원 접수 후 처리 절차는 어떻게 되나요?",
            "민원 처리 기간은 보통 얼마나 걸리나요?",
            "민원 결과에 대한 이의제기 방법은 무엇인가요?",
            
            # 5. 도메인 전문 용어 관련 질문들
            "신청서와 유효성정보의 차이점은 무엇인가요?",
            "안전성정보와 유효성정보의 구분 기준은 무엇인가요?",
            "승인절차 유형별 차이점은 무엇인가요?",
            "임상시험 단계별 주요 평가 항목은 무엇인가요?"
        ]
    
    def _evaluate_questions(self, questions: List[str], metadata: List[Dict], vectors: np.ndarray) -> Dict:
        """질문 목록으로 평가 수행"""

        results = {
            'questions': [],
            'avg_top1_similarity': 0,
            'avg_top3_similarity': 0,
            'avg_top5_similarity': 0,
            'retrieval_diversity': 0,
            'total_questions': len(questions)
        }

        total_top1_sim = 0
        total_top3_sim = 0
        total_top5_sim = 0
        all_retrieved_indices = set()

        for question in questions:
            # 질문을 임베딩
            question_embedding = self.model.encode([question])[0]
            
            # 모든 문서와의 유사도 계산
            similarities = []
            for i, doc_embedding in enumerate(vectors):
                sim = self._calculate_similarity_vec(question_embedding, doc_embedding)
                similarities.append((sim, i))
            
            # 유사도 순으로 정렬
            similarities.sort(reverse=True)
            
            # Top-1, Top-3, Top-5 유사도 계산
            top1_sim = similarities[0][0] if similarities else 0
            top3_sim = np.mean([sim for sim, _ in similarities[:3]]) if len(similarities) >= 3 else 0
            top5_sim = np.mean([sim for sim, _ in similarities[:5]]) if len(similarities) >= 5 else 0
            
            # 검색된 문서 인덱스 수집
            retrieved_indices = [idx for _, idx in similarities[:5]]
            all_retrieved_indices.update(retrieved_indices)
            
            # 결과 저장
            question_result = {
                'question': question,
                'top1_similarity': top1_sim,
                'top3_similarity': top3_sim,
                'top5_similarity': top5_sim,
                'top5_indices': retrieved_indices
            }
            results['questions'].append(question_result)
            
            total_top1_sim += top1_sim
            total_top3_sim += top3_sim
            total_top5_sim += top5_sim

        # 평균 계산
        num_questions = len(questions)
        results['avg_top1_similarity'] = total_top1_sim / num_questions if num_questions > 0 else 0
        results['avg_top3_similarity'] = total_top3_sim / num_questions if num_questions > 0 else 0
        results['avg_top5_similarity'] = total_top5_sim / num_questions if num_questions > 0 else 0
        
        # 검색 다양성 (고유한 문서 수 / 전체 문서 수)
        results['retrieval_diversity'] = len(all_retrieved_indices) / len(vectors) if vectors.shape[0] > 0 else 0

        print(f"[INFO] 질문-답변 검색 품질 평가 완료")
        print(f"- 평균 Top-1 유사도: {results['avg_top1_similarity']:.4f}")
        print(f"- 평균 Top-3 유사도: {results['avg_top3_similarity']:.4f}")
        print(f"- 평균 Top-5 유사도: {results['avg_top5_similarity']:.4f}")
        print(f"- 검색 다양성: {results['retrieval_diversity']:.2f}")

        return results

    def generate_quality_report(self, experiment_dir: str, output_file: str = "embedding_quality_report.json"):
        """품질 평가 보고서 생성 (상세 설명 포함)"""
        print("=" * 60)
        print("임베딩 품질 평가 시작")
        print("=" * 60)
        
        total_steps = 8  # 전체 단계 수
        current_step = 0

        try:
            # 데이터 로드
            current_step += 1
            print(f"[STEP {current_step}/{total_steps}] 데이터 로드 중... ({current_step/total_steps*100:.1f}%)")
            vectors, metadata, json_data = self.load_experiment_data(experiment_dir)

            # 1. 벡터 분포 분석
            vector_stats = self.analyze_vector_distribution(vectors)

            # 2. 의미적 유사도 평가
            semantic_results = self.evaluate_semantic_similarity(metadata, vectors)

            # 3. 콘텐츠 품질 평가
            content_results = self.evaluate_content_quality(metadata)

            # 4. 질문-답변 검색 품질 평가 (QA 쌍 파일이 있으면 사용)
            qa_pairs_file = "qa_pairs.json" if os.path.exists("qa_pairs.json") else None
            qa_results = self.evaluate_qa_retrieval(metadata, vectors, qa_pairs_file)

            # 5. 외재적 평가 지표
            extrinsic_results = self.evaluate_extrinsic_metrics(metadata, vectors)

            # 6. 내재적 평가 지표
            intrinsic_results = self.evaluate_intrinsic_metrics(metadata, vectors)

            # 7. 종합 점수 계산 (새로운 지표들 포함)
            overall_score = self._calculate_overall_score_enhanced(
                vector_stats, semantic_results, content_results, qa_results, 
                extrinsic_results, intrinsic_results
            )

            # 8. 상세한 평가 설명 생성
            evaluation_explanations = self._generate_evaluation_explanations_enhanced(
                vector_stats, semantic_results, content_results, qa_results, 
                extrinsic_results, intrinsic_results, overall_score
            )

            # 9. 보고서 생성
            # numpy 배열을 리스트로 변환
            vector_stats_serializable = {
                'total_vectors': int(vector_stats['total_vectors']),
                'vector_dimension': int(vector_stats['vector_dimension']),
                'mean_values': vector_stats['mean_values'].tolist(),
                'std_values': vector_stats['std_values'].tolist(),
                'min_values': vector_stats['min_values'].tolist(),
                'max_values': vector_stats['max_values'].tolist(),
                'vector_norms': vector_stats['vector_norms'].tolist(),
                'is_normalized': bool(vector_stats['is_normalized']),
                'norm_mean': float(vector_stats['norm_mean']),
                'norm_std': float(vector_stats['norm_std'])
            }
            
            report = {
                'experiment_info': {
                    'directory': experiment_dir,
                    'model_name': self.model_name,
                    'evaluation_timestamp': str(np.datetime64('now'))
                },
                'vector_analysis': vector_stats_serializable,
                'semantic_evaluation': semantic_results,
                'content_quality': content_results,
                'qa_retrieval_evaluation': qa_results,
                'extrinsic_evaluation': extrinsic_results,
                'intrinsic_evaluation': intrinsic_results,
                'overall_score': float(overall_score),
                'evaluation_explanations': evaluation_explanations
            }

            # 보고서 저장
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

            print("=" * 60)
            print("임베딩 품질 평가 완료!")
            print(f"- 보고서 저장: {output_file}")
            print(f"- 종합 점수: {overall_score:.2f}/100")
            print("=" * 60)

            return True

        except Exception as e:
            print(f"[ERROR] 품질 평가 실패: {e}")
            print(f"[ERROR] 상세 오류: {sys.exc_info()}")
            return False

    def _calculate_overall_score(self, vector_stats: Dict, semantic_results: Dict, content_results: Dict, qa_results: Dict) -> float:
        """종합 점수 계산 (의약품 도메인 특화)"""
        score = 0
        
        # 벡터 품질 (25점)
        if vector_stats['is_normalized']:
            score += 12
        if vector_stats['norm_std'] < 1e-6:
            score += 13
        
        # 의미적 유사도 (30점) - 내재적 평가 강화
        separation_score = semantic_results['separation_score']
        if separation_score > 0.3:
            score += 15
        elif separation_score > 0.2:
            score += 12
        elif separation_score > 0.1:
            score += 8
        elif separation_score > 0:
            score += 4
        
        intra_sim = semantic_results['intra_similarity_mean']
        if intra_sim > 0.8:
            score += 10
        elif intra_sim > 0.7:
            score += 8
        elif intra_sim > 0.6:
            score += 5
        
        # 도메인 특화 평가 (15점)
        domain_sim = semantic_results.get('domain_similarity_mean', 0)
        if domain_sim > 0.7:
            score += 15
        elif domain_sim > 0.6:
            score += 12
        elif domain_sim > 0.5:
            score += 8
        
        # 콘텐츠 품질 (20점)
        content_score = 0
        content_score += content_results['section_coverage'] * 8
        content_score += content_results['keyword_coverage'] * 8
        content_score += content_results['confidence_coverage'] * 4
        score += content_score
        
        # 질문-답변 검색 품질 (10점) - 외재적 평가
        qa_score = 0
        if qa_results['avg_top1_similarity'] > 0.7:
            qa_score += 6
        elif qa_results['avg_top1_similarity'] > 0.6:
            qa_score += 4
        elif qa_results['avg_top1_similarity'] > 0.5:
            qa_score += 2
        
        if qa_results['retrieval_diversity'] > 0.3:
            qa_score += 4
        elif qa_results['retrieval_diversity'] > 0.2:
            qa_score += 3
        elif qa_results['retrieval_diversity'] > 0.1:
            qa_score += 2
        
        score += qa_score
        
        return min(score, 100)

    def _calculate_overall_score_enhanced(self, vector_stats: Dict, semantic_results: Dict, 
                                        content_results: Dict, qa_results: Dict,
                                        extrinsic_results: Dict, intrinsic_results: Dict) -> float:
        """향상된 종합 점수 계산 (새로운 평가 지표들 포함)"""
        score = 0
        
        # 1. 벡터 품질 (15점)
        if vector_stats['is_normalized']:
            score += 8
        if vector_stats['norm_std'] < 1e-6:
            score += 7
        
        # 2. 의미적 유사도 (20점)
        separation_score = semantic_results['separation_score']
        if separation_score > 0.3:
            score += 10
        elif separation_score > 0.2:
            score += 8
        elif separation_score > 0.1:
            score += 5
        
        intra_sim = semantic_results['intra_similarity_mean']
        if intra_sim > 0.8:
            score += 10
        elif intra_sim > 0.7:
            score += 8
        elif intra_sim > 0.6:
            score += 5
        
        # 3. 외재적 평가 (35점) - 핵심 지표
        extrinsic_score = 0
        
        # 검색 성능 (15점)
        retrieval = extrinsic_results['retrieval_metrics']
        if retrieval['precision_at_1'] > 0.7:
            extrinsic_score += 8
        elif retrieval['precision_at_1'] > 0.5:
            extrinsic_score += 5
        elif retrieval['precision_at_1'] > 0.3:
            extrinsic_score += 3
        
        if retrieval['mrr'] > 0.6:
            extrinsic_score += 7
        elif retrieval['mrr'] > 0.4:
            extrinsic_score += 4
        elif retrieval['mrr'] > 0.2:
            extrinsic_score += 2
        
        # QA 정확도 (10점) - 개선된 방식
        qa_accuracy = extrinsic_results['qa_accuracy']
        
        # 다양한 지표 종합 평가
        qa_score = 0
        
        # Semantic Cohesion (3점)
        if qa_accuracy['semantic_cohesion'] > 0.7:
            qa_score += 3
        elif qa_accuracy['semantic_cohesion'] > 0.5:
            qa_score += 2
        elif qa_accuracy['semantic_cohesion'] > 0.3:
            qa_score += 1
        
        # Content Relevance (3점)
        if qa_accuracy['content_relevance'] > 0.6:
            qa_score += 3
        elif qa_accuracy['content_relevance'] > 0.4:
            qa_score += 2
        elif qa_accuracy['content_relevance'] > 0.2:
            qa_score += 1
        
        # Top-K Accuracy (2점)
        if qa_accuracy['top_k_accuracy'] > 0.7:
            qa_score += 2
        elif qa_accuracy['top_k_accuracy'] > 0.5:
            qa_score += 1
        
        # Keyword Matching (2점)
        if qa_accuracy['keyword_matching'] > 0.6:
            qa_score += 2
        elif qa_accuracy['keyword_matching'] > 0.4:
            qa_score += 1
        
        extrinsic_score += qa_score
        
        # 샘플 효율성 (10점)
        efficiency = extrinsic_results['sample_efficiency']
        if efficiency['efficiency_score'] > 0.1:
            extrinsic_score += 10
        elif efficiency['efficiency_score'] > 0.05:
            extrinsic_score += 7
        elif efficiency['efficiency_score'] > 0:
            extrinsic_score += 4
        
        score += extrinsic_score
        
        # 4. 내재적 평가 (20점)
        intrinsic_score = 0
        
        # 의미 유사도 상관관계 (8점)
        semantic_corr = intrinsic_results['semantic_correlation']
        if semantic_corr['spearman_correlation'] > 0.7:
            intrinsic_score += 8
        elif semantic_corr['spearman_correlation'] > 0.5:
            intrinsic_score += 6
        elif semantic_corr['spearman_correlation'] > 0.3:
            intrinsic_score += 4
        
        # 클러스터링 일관성 (7점)
        clustering = intrinsic_results['clustering_consistency']
        if clustering['purity'] > 0.8:
            intrinsic_score += 4
        elif clustering['purity'] > 0.6:
            intrinsic_score += 3
        elif clustering['purity'] > 0.4:
            intrinsic_score += 2
        
        if clustering['nmi'] > 0.6:
            intrinsic_score += 3
        elif clustering['nmi'] > 0.4:
            intrinsic_score += 2
        elif clustering['nmi'] > 0.2:
            intrinsic_score += 1
        
        # 유추 QA 성능 (5점)
        analogy = intrinsic_results['analogy_performance']
        if analogy['accuracy'] > 0.7:
            intrinsic_score += 5
        elif analogy['accuracy'] > 0.5:
            intrinsic_score += 3
        elif analogy['accuracy'] > 0.3:
            intrinsic_score += 2
        
        score += intrinsic_score
        
        # 5. 콘텐츠 품질 (10점)
        content_score = 0
        content_score += content_results['section_coverage'] * 4
        content_score += content_results['keyword_coverage'] * 4
        content_score += content_results['confidence_coverage'] * 2
        score += content_score
        
        return min(score, 100)

    def _generate_evaluation_explanations_enhanced(self, vector_stats: Dict, semantic_results: Dict,
                                                 content_results: Dict, qa_results: Dict,
                                                 extrinsic_results: Dict, intrinsic_results: Dict,
                                                 overall_score: float) -> Dict:
        """향상된 평가 설명 생성 (새로운 지표들 포함)"""
        
        explanations = {
            'overall_assessment': {
                'score': overall_score,
                'level': self._get_score_level(overall_score),
                'description': self._get_overall_description_enhanced(overall_score, extrinsic_results, intrinsic_results)
            },
            'intrinsic_evaluation': {
                'vector_quality': {
                    'score': self._calculate_vector_quality_score(vector_stats),
                    'description': self._get_vector_quality_description(vector_stats),
                    'recommendations': self._get_vector_quality_recommendations(vector_stats)
                },
                'semantic_similarity': {
                    'score': self._calculate_semantic_score(semantic_results),
                    'description': self._get_semantic_description(semantic_results),
                    'recommendations': self._get_semantic_recommendations(semantic_results)
                },
                'semantic_correlation': {
                    'score': self._calculate_semantic_correlation_score(intrinsic_results['semantic_correlation']),
                    'description': self._get_semantic_correlation_description(intrinsic_results['semantic_correlation']),
                    'recommendations': self._get_semantic_correlation_recommendations(intrinsic_results['semantic_correlation'])
                },
                'clustering_consistency': {
                    'score': self._calculate_clustering_score(intrinsic_results['clustering_consistency']),
                    'description': self._get_clustering_description(intrinsic_results['clustering_consistency']),
                    'recommendations': self._get_clustering_recommendations(intrinsic_results['clustering_consistency'])
                },
                'analogy_performance': {
                    'score': self._calculate_analogy_score(intrinsic_results['analogy_performance']),
                    'description': self._get_analogy_description(intrinsic_results['analogy_performance']),
                    'recommendations': self._get_analogy_recommendations(intrinsic_results['analogy_performance'])
                }
            },
            'extrinsic_evaluation': {
                'content_quality': {
                    'score': self._calculate_content_score(content_results),
                    'description': self._get_content_description(content_results),
                    'recommendations': self._get_content_recommendations(content_results)
                },
                'qa_retrieval': {
                    'score': self._calculate_qa_score(qa_results),
                    'description': self._get_qa_description(qa_results),
                    'recommendations': self._get_qa_recommendations(qa_results)
                },
                'retrieval_performance': {
                    'score': self._calculate_retrieval_score(extrinsic_results),
                    'description': self._get_retrieval_description(extrinsic_results),
                    'recommendations': self._get_retrieval_recommendations(extrinsic_results)
                },
                'qa_accuracy': {
                    'score': self._calculate_qa_accuracy_score(extrinsic_results),
                    'description': self._get_qa_accuracy_description(extrinsic_results),
                    'recommendations': self._get_qa_accuracy_recommendations(extrinsic_results)
                },
                'sample_efficiency': {
                    'score': self._calculate_efficiency_score_enhanced(extrinsic_results),
                    'description': self._get_efficiency_description(extrinsic_results),
                    'recommendations': self._get_efficiency_recommendations(extrinsic_results)
                }
            }
        }
        
        return explanations

    def _get_score_level(self, score: float) -> str:
        """점수 레벨 반환"""
        if score >= 90:
            return "우수"
        elif score >= 80:
            return "양호"
        elif score >= 70:
            return "보통"
        elif score >= 60:
            return "미흡"
        else:
            return "불량"

    def _get_overall_description_enhanced(self, score: float, extrinsic_results: Dict, intrinsic_results: Dict) -> str:
        """향상된 전체 평가 설명"""
        if score >= 90:
            return "임베딩 품질이 매우 우수합니다. 의약품 도메인에서 실제 서비스에 사용하기에 적합한 수준이며, 외재적/내재적 평가 모두에서 높은 성능을 보입니다."
        elif score >= 80:
            return "임베딩 품질이 양호합니다. 대부분의 용도에 적합하며, 일부 개선을 통해 더 나은 성능을 얻을 수 있습니다."
        elif score >= 70:
            return "임베딩 품질이 보통입니다. 기본적인 검색 기능에는 적합하지만, 고도화된 서비스에는 추가 개선이 필요합니다."
        elif score >= 60:
            return "임베딩 품질이 미흡합니다. 상당한 개선이 필요하며, 현재 상태로는 제한적인 용도에만 사용 가능합니다."
        else:
            return "임베딩 품질이 불량합니다. 대폭적인 개선이 필요하며, 현재 상태로는 사용을 권장하지 않습니다."

    # 새로운 평가 지표들의 설명 메서드들
    def _calculate_semantic_correlation_score(self, semantic_correlation: Dict) -> float:
        """의미 유사도 상관관계 점수"""
        try:
            corr = semantic_correlation['spearman_correlation']
            if corr > 0.7:
                return 100
            elif corr > 0.5:
                return 80
            elif corr > 0.3:
                return 60
            else:
                return 40
        except (KeyError, TypeError):
            return 0

    def _get_semantic_correlation_description(self, semantic_correlation: Dict) -> str:
        """의미 유사도 상관관계 설명"""
        try:
            corr = semantic_correlation['spearman_correlation']
            if corr > 0.7:
                return "의미 유사도 상관관계가 매우 높습니다. 임베딩 모델이 도메인 전문가의 판단과 일치합니다."
            elif corr > 0.5:
                return "의미 유사도 상관관계가 양호합니다. 대부분의 경우 전문가 판단과 일치합니다."
            elif corr > 0.3:
                return "의미 유사도 상관관계가 보통입니다. 일부 개선이 필요합니다."
            else:
                return "의미 유사도 상관관계가 낮습니다. 임베딩 모델 개선이 필요합니다."
        except (KeyError, TypeError):
            return "의미 유사도 상관관계 데이터를 찾을 수 없습니다."

    def _get_semantic_correlation_recommendations(self, semantic_correlation: Dict) -> List[str]:
        """의미 유사도 상관관계 개선 권장사항"""
        recommendations = []
        try:
            corr = semantic_correlation['spearman_correlation']
            
            if corr < 0.5:
                recommendations.append("더 정교한 임베딩 모델을 사용하세요.")
                recommendations.append("도메인 특화 텍스트 전처리를 적용하세요.")
        except (KeyError, TypeError):
            recommendations.append("의미 유사도 상관관계 데이터를 확인하세요.")
        
        return recommendations

    def _calculate_clustering_score(self, clustering_consistency: Dict) -> float:
        """클러스터링 점수"""
        try:
            score = 0
            
            if clustering_consistency['purity'] > 0.8:
                score += 50
            elif clustering_consistency['purity'] > 0.6:
                score += 40
            elif clustering_consistency['purity'] > 0.4:
                score += 30
            
            if clustering_consistency['nmi'] > 0.6:
                score += 50
            elif clustering_consistency['nmi'] > 0.4:
                score += 40
            elif clustering_consistency['nmi'] > 0.2:
                score += 30
            
            return min(score, 100)
        except (KeyError, TypeError):
            return 0

    def _get_clustering_description(self, clustering_consistency: Dict) -> str:
        """클러스터링 설명"""
        try:
            purity = clustering_consistency['purity']
            nmi = clustering_consistency['nmi']
            
            if purity > 0.8 and nmi > 0.6:
                return "클러스터링 일관성이 매우 우수합니다. 문서들이 의미적으로 잘 그룹화되어 있습니다."
            elif purity > 0.6 and nmi > 0.4:
                return "클러스터링 일관성이 양호합니다. 대부분의 문서가 적절히 그룹화되어 있습니다."
            elif purity > 0.4 and nmi > 0.2:
                return "클러스터링 일관성이 보통입니다. 기본적인 그룹화는 되지만 개선이 필요합니다."
            else:
                return "클러스터링 일관성이 낮습니다. 문서 그룹화가 명확하지 않습니다."
        except (KeyError, TypeError):
            return "클러스터링 데이터를 찾을 수 없습니다."

    def _get_clustering_recommendations(self, clustering_consistency: Dict) -> List[str]:
        """클러스터링 개선 권장사항"""
        recommendations = []
        try:
            if clustering_consistency['purity'] < 0.6:
                recommendations.append("더 정교한 클러스터링 알고리즘을 사용하세요.")
            if clustering_consistency['nmi'] < 0.4:
                recommendations.append("문서 분할 방식을 개선하세요.")
        except (KeyError, TypeError):
            recommendations.append("클러스터링 데이터를 확인하세요.")
        
        return recommendations

    def _calculate_analogy_score(self, analogy_performance: Dict) -> float:
        """유추 QA 점수"""
        try:
            accuracy = analogy_performance['accuracy']
            return accuracy * 100
        except (KeyError, TypeError):
            return 0

    def _get_analogy_description(self, analogy_performance: Dict) -> str:
        """유추 QA 설명"""
        try:
            accuracy = analogy_performance['accuracy']
            
            if accuracy > 0.7:
                return "유추 QA 성능이 매우 우수합니다. 도메인 내부 연산이 정확하게 수행됩니다."
            elif accuracy > 0.5:
                return "유추 QA 성능이 양호합니다. 대부분의 유추 문제를 해결할 수 있습니다."
            elif accuracy > 0.3:
                return "유추 QA 성능이 보통입니다. 기본적인 유추는 가능하지만 개선이 필요합니다."
            else:
                return "유추 QA 성능이 낮습니다. 도메인 내부 연산이 어렵습니다."
        except (KeyError, TypeError):
            return "유추 QA 데이터를 찾을 수 없습니다."

    def _get_analogy_recommendations(self, analogy_performance: Dict) -> List[str]:
        """유추 QA 개선 권장사항"""
        recommendations = []
        try:
            accuracy = analogy_performance['accuracy']
            
            if accuracy < 0.5:
                recommendations.append("더 정교한 임베딩 모델을 사용하세요.")
                recommendations.append("도메인 특화 벡터 연산을 개선하세요.")
        except (KeyError, TypeError):
            recommendations.append("유추 QA 데이터를 확인하세요.")
        
        return recommendations

    def _calculate_retrieval_score(self, extrinsic_results: Dict) -> float:
        """검색 성능 점수"""
        retrieval = extrinsic_results['retrieval_metrics']
        score = 0
        
        if retrieval['precision_at_1'] > 0.7:
            score += 50
        elif retrieval['precision_at_1'] > 0.5:
            score += 40
        elif retrieval['precision_at_1'] > 0.3:
            score += 30
        
        if retrieval['mrr'] > 0.6:
            score += 50
        elif retrieval['mrr'] > 0.4:
            score += 40
        elif retrieval['mrr'] > 0.2:
            score += 30
        
        return min(score, 100)

    def _get_retrieval_description(self, extrinsic_results: Dict) -> str:
        """검색 성능 설명"""
        retrieval = extrinsic_results['retrieval_metrics']
        precision = retrieval['precision_at_1']
        mrr = retrieval['mrr']
        
        if precision > 0.7 and mrr > 0.6:
            return "검색 성능이 매우 우수합니다. 사용자 질문에 대해 정확하고 관련성 높은 결과를 제공합니다."
        elif precision > 0.5 and mrr > 0.4:
            return "검색 성능이 양호합니다. 대부분의 질문에 대해 적절한 결과를 제공합니다."
        elif precision > 0.3 and mrr > 0.2:
            return "검색 성능이 보통입니다. 기본적인 검색은 가능하지만 정확도 개선이 필요합니다."
        else:
            return "검색 성능이 낮습니다. 사용자 질문에 대한 적절한 결과를 찾기 어렵습니다."

    def _get_retrieval_recommendations(self, extrinsic_results: Dict) -> List[str]:
        """검색 성능 개선 권장사항"""
        recommendations = []
        retrieval = extrinsic_results['retrieval_metrics']
        
        if retrieval['precision_at_1'] < 0.5:
            recommendations.append("임베딩 모델을 더 정교한 것으로 교체하세요.")
        if retrieval['mrr'] < 0.4:
            recommendations.append("검색 알고리즘을 개선하거나 재순위화를 적용하세요.")
        
        return recommendations

    def _calculate_qa_accuracy_score(self, extrinsic_results: Dict) -> float:
        """QA 정확도 점수"""
        qa_accuracy = extrinsic_results['qa_accuracy']
        score = 0
        
        if qa_accuracy['exact_match'] > 0.6:
            score += 50
        elif qa_accuracy['exact_match'] > 0.4:
            score += 40
        elif qa_accuracy['exact_match'] > 0.2:
            score += 30
        
        if qa_accuracy['f1_score'] > 0.6:
            score += 50
        elif qa_accuracy['f1_score'] > 0.4:
            score += 40
        elif qa_accuracy['f1_score'] > 0.2:
            score += 30
        
        return min(score, 100)

    def _get_qa_accuracy_description(self, extrinsic_results: Dict) -> str:
        """QA 정확도 설명"""
        qa_accuracy = extrinsic_results['qa_accuracy']
        exact_match = qa_accuracy['exact_match']
        f1 = qa_accuracy['f1_score']
        
        if exact_match > 0.6 and f1 > 0.6:
            return "QA 정확도가 매우 우수합니다. 질문에 대한 정확한 답변을 제공합니다."
        elif exact_match > 0.4 and f1 > 0.4:
            return "QA 정확도가 양호합니다. 대부분의 질문에 대해 적절한 답변을 제공합니다."
        elif exact_match > 0.2 and f1 > 0.2:
            return "QA 정확도가 보통입니다. 기본적인 답변은 가능하지만 정확도 개선이 필요합니다."
        else:
            return "QA 정확도가 낮습니다. 질문에 대한 적절한 답변을 제공하기 어렵습니다."

    def _get_qa_accuracy_recommendations(self, extrinsic_results: Dict) -> List[str]:
        """QA 정확도 개선 권장사항"""
        recommendations = []
        qa_accuracy = extrinsic_results['qa_accuracy']
        
        if qa_accuracy['exact_match'] < 0.4:
            recommendations.append("더 정교한 답변 생성 모델을 사용하세요.")
        if qa_accuracy['f1_score'] < 0.4:
            recommendations.append("답변 품질 평가 및 개선 시스템을 도입하세요.")
        
        return recommendations

    def _calculate_efficiency_score_enhanced(self, extrinsic_results: Dict) -> float:
        """향상된 샘플 효율성 점수"""
        efficiency = extrinsic_results['sample_efficiency']
        score = efficiency['efficiency_score'] * 100
        return min(score, 100)

    def _get_efficiency_description(self, extrinsic_results: Dict) -> str:
        """샘플 효율성 설명"""
        efficiency = extrinsic_results['sample_efficiency']
        score = efficiency['efficiency_score']
        
        if score > 0.1:
            return "샘플 효율성이 매우 우수합니다. 소량의 데이터로도 높은 성능을 달성합니다."
        elif score > 0.05:
            return "샘플 효율성이 양호합니다. 적절한 양의 데이터로 충분한 성능을 달성합니다."
        elif score > 0:
            return "샘플 효율성이 보통입니다. 더 많은 데이터가 필요할 수 있습니다."
        else:
            return "샘플 효율성이 낮습니다. 대량의 데이터가 필요합니다."

    def _get_efficiency_recommendations(self, extrinsic_results: Dict) -> List[str]:
        """샘플 효율성 개선 권장사항"""
        recommendations = []
        efficiency = extrinsic_results['sample_efficiency']
        
        if efficiency['efficiency_score'] < 0.05:
            recommendations.append("데이터 품질을 개선하세요.")
            recommendations.append("더 효율적인 학습 방법을 적용하세요.")
        
        return recommendations

    def _calculate_vector_quality_score(self, vector_stats: Dict) -> float:
        """벡터 품질 점수 계산"""
        score = 0
        if vector_stats['is_normalized']:
            score += 50
        if vector_stats['norm_std'] < 1e-6:
            score += 50
        return score

    def _get_vector_quality_description(self, vector_stats: Dict) -> str:
        """벡터 품질 설명"""
        if vector_stats['is_normalized'] and vector_stats['norm_std'] < 1e-6:
            return "벡터가 완벽하게 정규화되어 있어 코사인 유사도 계산에 최적화되어 있습니다."
        elif vector_stats['is_normalized']:
            return "벡터가 정규화되어 있지만, 일부 불규칙성이 있어 추가 정규화가 도움될 수 있습니다."
        else:
            return "벡터가 정규화되지 않아 유사도 계산 시 정확도가 떨어질 수 있습니다."

    def _get_vector_quality_recommendations(self, vector_stats: Dict) -> List[str]:
        """벡터 품질 개선 권장사항"""
        recommendations = []
        if not vector_stats['is_normalized']:
            recommendations.append("벡터 정규화를 적용하여 코사인 유사도 계산을 개선하세요.")
        if vector_stats['norm_std'] > 1e-6:
            recommendations.append("정규화 과정에서 더 엄격한 기준을 적용하세요.")
        return recommendations

    def _calculate_semantic_score(self, semantic_results: Dict) -> float:
        """의미적 유사도 점수 계산"""
        score = 0
        separation = semantic_results['separation_score']
        if separation > 0.3:
            score += 50
        elif separation > 0.2:
            score += 40
        elif separation > 0.1:
            score += 30
        elif separation > 0:
            score += 20
        
        intra_sim = semantic_results['intra_similarity_mean']
        if intra_sim > 0.8:
            score += 50
        elif intra_sim > 0.7:
            score += 40
        elif intra_sim > 0.6:
            score += 30
        else:
            score += 20
        
        return min(score, 100)

    def _get_semantic_description(self, semantic_results: Dict) -> str:
        """의미적 유사도 설명"""
        separation = semantic_results['separation_score']
        intra_sim = semantic_results['intra_similarity_mean']
        
        if separation > 0.3 and intra_sim > 0.8:
            return "의미적 클러스터링이 매우 우수합니다. 관련 문서들이 잘 그룹화되어 있고, 다른 그룹과 명확히 구분됩니다."
        elif separation > 0.2 and intra_sim > 0.7:
            return "의미적 클러스터링이 양호합니다. 대부분의 관련 문서들이 적절히 그룹화되어 있습니다."
        elif separation > 0.1 and intra_sim > 0.6:
            return "의미적 클러스터링이 보통입니다. 기본적인 그룹화는 되지만 개선의 여지가 있습니다."
        else:
            return "의미적 클러스터링이 미흡합니다. 문서 간 구분이 명확하지 않아 검색 성능에 영향을 줄 수 있습니다."

    def _get_semantic_recommendations(self, semantic_results: Dict) -> List[str]:
        """의미적 유사도 개선 권장사항"""
        recommendations = []
        separation = semantic_results['separation_score']
        intra_sim = semantic_results['intra_similarity_mean']
        
        if separation < 0.2:
            recommendations.append("더 정교한 텍스트 전처리를 통해 문서 간 구분을 명확히 하세요.")
        if intra_sim < 0.7:
            recommendations.append("관련 문서들을 더 유사한 형태로 구성하거나 임베딩 모델을 개선하세요.")
        
        return recommendations

    def _calculate_content_score(self, content_results: Dict) -> float:
        """콘텐츠 품질 점수 계산"""
        score = 0
        score += content_results['section_coverage'] * 40
        score += content_results['keyword_coverage'] * 40
        score += content_results['confidence_coverage'] * 20
        return score

    def _get_content_description(self, content_results: Dict) -> str:
        """콘텐츠 품질 설명"""
        coverage = (content_results['section_coverage'] + content_results['keyword_coverage']) / 2
        
        if coverage > 0.95:
            return "콘텐츠 메타데이터가 매우 완벽하게 구성되어 있습니다."
        elif coverage > 0.9:
            return "콘텐츠 메타데이터가 양호하게 구성되어 있습니다."
        elif coverage > 0.8:
            return "콘텐츠 메타데이터가 보통 수준으로 구성되어 있습니다."
        else:
            return "콘텐츠 메타데이터가 부족하여 검색 품질에 영향을 줄 수 있습니다."

    def _get_content_recommendations(self, content_results: Dict) -> List[str]:
        """콘텐츠 품질 개선 권장사항"""
        recommendations = []
        
        if content_results['section_coverage'] < 0.9:
            recommendations.append("섹션 제목이 없는 문서들을 식별하고 추가하세요.")
        if content_results['keyword_coverage'] < 0.9:
            recommendations.append("키워드가 없는 문서들을 식별하고 관련 키워드를 추가하세요.")
        if content_results['confidence_coverage'] < 0.9:
            recommendations.append("신뢰도 정보가 없는 문서들을 식별하고 신뢰도를 계산하세요.")
        
        return recommendations

    def _calculate_qa_score(self, qa_results: Dict) -> float:
        """QA 검색 점수 계산"""
        score = 0
        top1_sim = qa_results['avg_top1_similarity']
        diversity = qa_results['retrieval_diversity']
        
        if top1_sim > 0.7:
            score += 60
        elif top1_sim > 0.6:
            score += 40
        elif top1_sim > 0.5:
            score += 20
        
        if diversity > 0.3:
            score += 40
        elif diversity > 0.2:
            score += 30
        elif diversity > 0.1:
            score += 20
        
        return min(score, 100)

    def _get_qa_description(self, qa_results: Dict) -> str:
        """QA 검색 설명"""
        top1_sim = qa_results['avg_top1_similarity']
        diversity = qa_results['retrieval_diversity']
        
        if top1_sim > 0.7 and diversity > 0.2:
            return "질문-답변 검색 성능이 매우 우수합니다. 사용자 질문에 대해 정확하고 다양한 관련 문서를 찾을 수 있습니다."
        elif top1_sim > 0.6 and diversity > 0.15:
            return "질문-답변 검색 성능이 양호합니다. 대부분의 질문에 대해 적절한 답변을 제공할 수 있습니다."
        elif top1_sim > 0.5:
            return "질문-답변 검색 성능이 보통입니다. 기본적인 검색은 가능하지만 정확도 개선이 필요합니다."
        else:
            return "질문-답변 검색 성능이 미흡합니다. 사용자 질문에 대한 적절한 답변을 찾기 어려울 수 있습니다."

    def _get_qa_recommendations(self, qa_results: Dict) -> List[str]:
        """QA 검색 개선 권장사항"""
        recommendations = []
        top1_sim = qa_results['avg_top1_similarity']
        diversity = qa_results['retrieval_diversity']
        
        if top1_sim < 0.6:
            recommendations.append("임베딩 모델을 더 정교한 것으로 교체하거나 텍스트 전처리를 개선하세요.")
        if diversity < 0.2:
            recommendations.append("질문 다양성을 높이거나 문서 분할 방식을 개선하세요.")
        
        return recommendations

    def _calculate_retrieval_metrics_score(self, retrieval_metrics: Dict) -> float:
        """검색 성능 지표 점수 계산 (Precision@k, Recall@k, MRR, MAP)"""
        score = 0
        precision_at_1 = retrieval_metrics['precision_at_1']
        precision_at_3 = retrieval_metrics['precision_at_3']
        precision_at_5 = retrieval_metrics['precision_at_5']
        recall_at_3 = retrieval_metrics['recall_at_3']
        recall_at_5 = retrieval_metrics['recall_at_5']
        mrr = retrieval_metrics['mrr']
        map_score = retrieval_metrics['map']

        # Precision@1 점수 (30점)
        if precision_at_1 > 0.9:
            score += 30
        elif precision_at_1 > 0.8:
            score += 25
        elif precision_at_1 > 0.7:
            score += 20
        elif precision_at_1 > 0.6:
            score += 15
        elif precision_at_1 > 0.5:
            score += 10

        # Precision@3 점수 (20점)
        if precision_at_3 > 0.8:
            score += 20
        elif precision_at_3 > 0.7:
            score += 15
        elif precision_at_3 > 0.6:
            score += 10
        elif precision_at_3 > 0.5:
            score += 5

        # Recall@3 점수 (15점)
        if recall_at_3 > 0.7:
            score += 15
        elif recall_at_3 > 0.6:
            score += 12
        elif recall_at_3 > 0.5:
            score += 8
        elif recall_at_3 > 0.4:
            score += 5

        # MRR 점수 (20점)
        if mrr > 0.9:
            score += 20
        elif mrr > 0.8:
            score += 15
        elif mrr > 0.7:
            score += 10
        elif mrr > 0.6:
            score += 5

        # MAP 점수 (15점)
        if map_score > 0.8:
            score += 15
        elif map_score > 0.7:
            score += 12
        elif map_score > 0.6:
            score += 8
        elif map_score > 0.5:
            score += 5

        return min(score, 100)

    def _get_retrieval_metrics_description(self, retrieval_metrics: Dict) -> str:
        """검색 성능 지표 설명"""
        precision_at_1 = retrieval_metrics['precision_at_1']
        precision_at_3 = retrieval_metrics['precision_at_3']
        mrr = retrieval_metrics['mrr']

        if precision_at_1 > 0.9:
            return "검색 성능이 매우 우수합니다. 상위 1개 문서에서 정확하게 관련 문서를 찾을 수 있습니다."
        elif precision_at_1 > 0.8:
            return "검색 성능이 양호합니다. 상위 1개 문서에서 대부분의 관련 문서를 찾을 수 있습니다."
        elif precision_at_1 > 0.7:
            return "검색 성능이 보통입니다. 상위 1개 문서에서 일부 관련 문서를 찾을 수 있습니다."
        elif precision_at_1 > 0.6:
            return "검색 성능이 미흡합니다. 상위 1개 문서에서 관련 문서를 찾기 어려울 수 있습니다."
        else:
            return "검색 성능이 매우 미흡합니다. 상위 1개 문서에서 관련 문서를 찾을 수 없습니다."

    def _get_retrieval_metrics_recommendations(self, retrieval_metrics: Dict) -> List[str]:
        """검색 성능 지표 개선 권장사항"""
        recommendations = []
        precision_at_1 = retrieval_metrics['precision_at_1']
        precision_at_3 = retrieval_metrics['precision_at_3']
        mrr = retrieval_metrics['mrr']

        if precision_at_1 < 0.8:
            recommendations.append("임베딩 모델을 더 정교한 것으로 교체하거나 텍스트 전처리를 개선하세요.")
        if precision_at_3 < 0.7:
            recommendations.append("문서 분할 방식을 개선하거나 더 많은 문서를 포함하도록 데이터를 확보하세요.")
        if mrr < 0.7:
            recommendations.append("문서 임베딩 방식을 개선하거나 더 많은 문서를 포함하도록 데이터를 확보하세요.")

        return recommendations

    def _calculate_qa_accuracy_score(self, extrinsic_results: Dict) -> float:
        """QA 정확도 점수"""
        qa_accuracy = extrinsic_results['qa_accuracy']
        score = 0
        
        if qa_accuracy['exact_match'] > 0.6:
            score += 50
        elif qa_accuracy['exact_match'] > 0.4:
            score += 40
        elif qa_accuracy['exact_match'] > 0.2:
            score += 30
        
        if qa_accuracy['f1_score'] > 0.6:
            score += 50
        elif qa_accuracy['f1_score'] > 0.4:
            score += 40
        elif qa_accuracy['f1_score'] > 0.2:
            score += 30
        
        return min(score, 100)

    def _get_qa_accuracy_description(self, extrinsic_results: Dict) -> str:
        """QA 정확도 설명"""
        qa_accuracy = extrinsic_results['qa_accuracy']
        exact_match = qa_accuracy['exact_match']
        f1 = qa_accuracy['f1_score']
        
        if exact_match > 0.6 and f1 > 0.6:
            return "QA 정확도가 매우 우수합니다. 질문에 대한 정확한 답변을 제공합니다."
        elif exact_match > 0.4 and f1 > 0.4:
            return "QA 정확도가 양호합니다. 대부분의 질문에 대해 적절한 답변을 제공합니다."
        elif exact_match > 0.2 and f1 > 0.2:
            return "QA 정확도가 보통입니다. 기본적인 답변은 가능하지만 정확도 개선이 필요합니다."
        else:
            return "QA 정확도가 낮습니다. 질문에 대한 적절한 답변을 제공하기 어렵습니다."

    def _get_qa_accuracy_recommendations(self, extrinsic_results: Dict) -> List[str]:
        """QA 정확도 개선 권장사항"""
        recommendations = []
        qa_accuracy = extrinsic_results['qa_accuracy']
        
        if qa_accuracy['exact_match'] < 0.4:
            recommendations.append("더 정교한 답변 생성 모델을 사용하세요.")
        if qa_accuracy['f1_score'] < 0.4:
            recommendations.append("답변 품질 평가 및 개선 시스템을 도입하세요.")
        
        return recommendations

    def _calculate_sample_efficiency_score(self, sample_efficiency: Dict) -> float:
        """샘플 효율성 점수 계산"""
        score = 0
        efficiency_score = sample_efficiency['efficiency_score']

        if efficiency_score > 0.5:
            score += 50
        elif efficiency_score > 0.3:
            score += 40
        elif efficiency_score > 0.1:
            score += 30
        elif efficiency_score > 0:
            score += 20

        return min(score, 100)

    def _get_sample_efficiency_description(self, sample_efficiency: Dict) -> str:
        """샘플 효율성 설명"""
        efficiency_score = sample_efficiency['efficiency_score']

        if efficiency_score > 0.5:
            return "샘플 효율성이 매우 우수합니다. 데이터 양에 비해 빠르게 성능 향상을 보입니다."
        elif efficiency_score > 0.3:
            return "샘플 효율성이 양호합니다. 데이터 양에 비해 적절한 성능 향상을 보입니다."
        elif efficiency_score > 0.1:
            return "샘플 효율성이 보통입니다. 데이터 양에 비해 성능 향상이 제한적입니다."
        else:
            return "샘플 효율성이 미흡합니다. 데이터 양에 비해 성능 향상이 거의 없습니다."

    def _get_sample_efficiency_recommendations(self, sample_efficiency: Dict) -> List[str]:
        """샘플 효율성 개선 권장사항"""
        recommendations = []
        efficiency_score = sample_efficiency['efficiency_score']

        if efficiency_score < 0.3:
            recommendations.append("더 많은 데이터를 확보하거나 문서 분할 방식을 개선하세요.")
        if efficiency_score < 0.1:
            recommendations.append("데이터 양을 확보하거나 임베딩 모델을 개선하세요.")

        return recommendations

    def _calculate_semantic_correlation_score(self, semantic_correlation: Dict) -> float:
        """의미 유사도 상관관계 점수 계산"""
        score = 0
        spearman_correlation = semantic_correlation['spearman_correlation']

        if spearman_correlation > 0.9:
            score += 50
        elif spearman_correlation > 0.8:
            score += 40
        elif spearman_correlation > 0.7:
            score += 30
        elif spearman_correlation > 0.6:
            score += 20
        elif spearman_correlation > 0.5:
            score += 10

        return min(score, 100)

    def _get_semantic_correlation_description(self, semantic_correlation: Dict) -> str:
        """의미 유사도 상관관계 설명"""
        spearman_correlation = semantic_correlation['spearman_correlation']

        if spearman_correlation > 0.9:
            return "의미 유사도와 전문가 점수 간 매우 강한 양의 상관관계가 있습니다."
        elif spearman_correlation > 0.8:
            return "의미 유사도와 전문가 점수 간 강한 양의 상관관계가 있습니다."
        elif spearman_correlation > 0.7:
            return "의미 유사도와 전문가 점수 간 높은 양의 상관관계가 있습니다."
        elif spearman_correlation > 0.6:
            return "의미 유사도와 전문가 점수 간 중간 양의 상관관계가 있습니다."
        elif spearman_correlation > 0.5:
            return "의미 유사도와 전문가 점수 간 약한 양의 상관관계가 있습니다."
        else:
            return "의미 유사도와 전문가 점수 간 상관관계가 없습니다."

    def _get_semantic_correlation_recommendations(self, semantic_correlation: Dict) -> List[str]:
        """의미 유사도 상관관계 개선 권장사항"""
        recommendations = []
        spearman_correlation = semantic_correlation['spearman_correlation']

        if spearman_correlation < 0.7:
            recommendations.append("더 많은 의미 유사도 문장 쌍을 포함하거나 임베딩 모델을 개선하세요.")
        if spearman_correlation < 0.6:
            recommendations.append("의미 유사도 계산 방식을 개선하거나 더 정교한 전문가 점수를 사용하세요.")

        return recommendations

    def _calculate_clustering_consistency_score(self, clustering_consistency: Dict) -> float:
        """클러스터링 일관성 점수 계산"""
        score = 0
        purity = clustering_consistency['purity']
        nmi = clustering_consistency['nmi']

        if purity > 0.9:
            score += 50
        elif purity > 0.8:
            score += 40
        elif purity > 0.7:
            score += 30
        elif purity > 0.6:
            score += 20
        elif purity > 0.5:
            score += 10

        if nmi > 0.9:
            score += 50
        elif nmi > 0.8:
            score += 40
        elif nmi > 0.7:
            score += 30
        elif nmi > 0.6:
            score += 20
        elif nmi > 0.5:
            score += 10

        return min(score, 100)

    def _get_clustering_consistency_description(self, clustering_consistency: Dict) -> str:
        """클러스터링 일관성 설명"""
        purity = clustering_consistency['purity']
        nmi = clustering_consistency['nmi']

        if purity > 0.9 and nmi > 0.9:
            return "클러스터링 일관성이 매우 우수합니다. 문서 간 구분이 명확하고 안정적입니다."
        elif purity > 0.8 and nmi > 0.8:
            return "클러스터링 일관성이 양호합니다. 대부분의 문서가 적절히 그룹화되어 있습니다."
        elif purity > 0.7 and nmi > 0.7:
            return "클러스터링 일관성이 보통입니다. 기본적인 그룹화는 되지만 개선의 여지가 있습니다."
        elif purity > 0.6 and nmi > 0.6:
            return "클러스터링 일관성이 미흡합니다. 문서 간 구분이 명확하지 않아 검색 성능에 영향을 줄 수 있습니다."
        else:
            return "클러스터링 일관성이 매우 미흡합니다. 문서 간 구분이 명확하지 않습니다."

    def _get_clustering_consistency_recommendations(self, clustering_consistency: Dict) -> List[str]:
        """클러스터링 일관성 개선 권장사항"""
        recommendations = []
        purity = clustering_consistency['purity']
        nmi = clustering_consistency['nmi']

        if purity < 0.7:
            recommendations.append("더 정교한 텍스트 전처리를 통해 문서 간 구분을 명확히 하세요.")
        if nmi < 0.7:
            recommendations.append("클러스터링 방식을 개선하거나 더 많은 문서를 포함하도록 데이터를 확보하세요.")

        return recommendations

    def _calculate_analogy_performance_score(self, analogy_performance: Dict) -> float:
        """유추 QA 성능 점수 계산"""
        score = 0
        accuracy = analogy_performance['accuracy']

        if accuracy > 0.9:
            score += 50
        elif accuracy > 0.8:
            score += 40
        elif accuracy > 0.7:
            score += 30
        elif accuracy > 0.6:
            score += 20
        elif accuracy > 0.5:
            score += 10

        return min(score, 100)

    def _get_analogy_performance_description(self, analogy_performance: Dict) -> str:
        """유추 QA 성능 설명"""
        accuracy = analogy_performance['accuracy']

        if accuracy > 0.9:
            return "유추 QA 성능이 매우 우수합니다. 의미 유사도 연산을 통해 정확한 유추를 수행할 수 있습니다."
        elif accuracy > 0.8:
            return "유추 QA 성능이 양호합니다. 의미 유사도 연산을 통해 대부분의 유추를 정확하게 수행할 수 있습니다."
        elif accuracy > 0.7:
            return "유추 QA 성능이 보통입니다. 의미 유사도 연산을 통해 일부 유추를 정확하게 수행할 수 있습니다."
        elif accuracy > 0.6:
            return "유추 QA 성능이 미흡합니다. 의미 유사도 연산을 통해 유추를 정확하게 수행하기 어려울 수 있습니다."
        else:
            return "유추 QA 성능이 매우 미흡합니다. 의미 유사도 연산을 통해 유추를 수행할 수 없습니다."

    def _get_analogy_performance_recommendations(self, analogy_performance: Dict) -> List[str]:
        """유추 QA 성능 개선 권장사항"""
        recommendations = []
        accuracy = analogy_performance['accuracy']

        if accuracy < 0.7:
            recommendations.append("임베딩 모델을 더 정교한 것으로 교체하거나 텍스트 전처리를 개선하세요.")
        if accuracy < 0.6:
            recommendations.append("유추 문제 집합을 더 다양하게 구성하거나 문서 분할 방식을 개선하세요.")

        return recommendations

def main():
    if len(sys.argv) < 2:
        print("사용법: python embedding_quality_evaluator.py <실험디렉토리> [출력파일]")
        sys.exit(1)
    
    experiment_dir = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "embedding_quality_report.json"
    
    if not os.path.exists(experiment_dir):
        print(f"[ERROR] 실험 디렉토리를 찾을 수 없습니다: {experiment_dir}")
        sys.exit(1)
    
    # 평가기 실행
    evaluator = EmbeddingQualityEvaluator()
    success = evaluator.generate_quality_report(experiment_dir, output_file)
    
    if success:
        print("임베딩 품질 평가가 성공적으로 완료되었습니다!")
    else:
        print("임베딩 품질 평가에 실패했습니다.")
        sys.exit(1)

if __name__ == "__main__":
    main() 