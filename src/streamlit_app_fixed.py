import streamlit as st
import os
import subprocess
import json
import pickle

def get_venv_python(venv_name):
    """가상환경의 Python 경로 반환"""
    venv_path = f"environments/{venv_name}/bin/python"
    if os.path.exists(venv_path):
        return venv_path
    else:
        st.error(f"가상환경을 찾을 수 없습니다: {venv_path}")
        return None

def run_safe_subprocess(cmd, description):
    """안전한 subprocess 실행"""
    try:
        with st.spinner(f"{description} 실행 중..."):
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            
        if result.returncode == 0:
            st.success(f"✅ {description} 완료!")
            if result.stdout:
                st.text("출력:")
                st.code(result.stdout)
        else:
            st.error(f"❌ {description} 실패!")
            if result.stderr:
                st.error("오류:")
                st.code(result.stderr)
    except Exception as e:
        st.error(f"❌ {description} 실행 중 오류 발생: {e}")

def load_json_file(filename):
    """JSON 파일 로드"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"JSON 파일 로드 실패: {e}")
        return None

def main():
    # 세션 상태 초기화
    if 'data_deleted' not in st.session_state:
        st.session_state.data_deleted = False
    if 'cache_deleted' not in st.session_state:
        st.session_state.cache_deleted = False
    if 'qa_generation_triggered' not in st.session_state:
        st.session_state.qa_generation_triggered = False
    
    st.set_page_config(
        page_title="PDF 텍스트 추출 & 임베딩 시스템",
        page_icon="",
        layout="wide"
    )
    
    st.title("PDF 텍스트 추출 & 임베딩 품질 평가 시스템")
    
    # 탭 생성
    tab1, tab2, tab3 = st.tabs(["파일 업로드 & 추출", "결과 확인", "시스템 정보"])
    
    with tab1:
        st.header("파일 업로드 및 데이터 관리")
        
        # 기존 데이터 삭제 버튼
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("기존 데이터 삭제", key="delete_data_btn"):
                files_to_delete = [
                    "text_only_chunks.json",
                    "final_text_only_data.json", 
                    "pymupdf_table_areas.json",
                    "data/llm_enhanced_sections.json",
                    "embeddings.faiss",
                    "embeddings_metadata.pkl",
                    "embedding_quality_report.json",
                    "embedding_quality_report_detailed.json",
                    "qa_pairs.json"
                ]
                
                deleted_count = 0
                for file in files_to_delete:
                    if os.path.exists(file):
                        try:
                            os.remove(file)
                            deleted_count += 1
                        except Exception as e:
                            st.error(f"파일 삭제 실패: {file} - {e}")
                
                if deleted_count > 0:
                    st.success(f"{deleted_count}개 파일 삭제 완료!")
                    # 세션 상태 업데이트로 안전한 리로드
                    st.session_state.data_deleted = True
                    st.rerun()
                else:
                    st.info("삭제할 파일이 없습니다.")
        
        with col2:
            if st.button("캐시 및 임시 파일 삭제", key="delete_cache_btn"):
                cache_files = [f for f in os.listdir('.') if f.endswith('.tmp') or f.endswith('.cache')]
                deleted_count = 0
                
                for file in cache_files:
                    try:
                        os.remove(file)
                        deleted_count += 1
                    except Exception as e:
                        st.error(f"파일 삭제 실패: {file} - {e}")
                
                if deleted_count > 0:
                    st.success(f"{deleted_count}개 캐시 파일 삭제 완료!")
                    # 세션 상태 업데이트로 안전한 리로드
                    st.session_state.cache_deleted = True
                    st.rerun()
                else:
                    st.info("삭제할 캐시 파일이 없습니다.")
        
        # 파일 업로드
        uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type=['pdf'])
        
        if uploaded_file is not None:
            # 파일 저장
            with open("input.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success(f"{uploaded_file.name} 업로드 완료!")
            
            # 추출 옵션
            st.subheader("추출 옵션")
            
            if st.button("하이브리드 시스템 (텍스트 전용)", type="primary", key="hybrid_system_btn"):
                venv_python = get_venv_python("venv_text_new")
                cmd = [venv_python, "src/hybrid_extraction_system.py", "input.pdf"]
                run_safe_subprocess(cmd, "하이브리드 시스템")
    
    with tab2:
        st.header("추출 결과 확인")
        
        # 결과 파일 목록 (핵심 파일만)
        result_files = [
            ("final_text_only_data.json", "최종 텍스트 데이터 (JSON)"),
            ("embeddings.faiss", "FAISS 임베딩 인덱스"),
            ("embeddings_metadata.pkl", "임베딩 메타데이터"),
            ("embedding_quality_report.json", "임베딩 품질 평가 보고서")
        ]
        
        for filename, description in result_files:
            if os.path.exists(filename):
                st.write(f"**{description}** (사용 가능)")
                
                # 파일 크기 표시
                file_size = os.path.getsize(filename)
                st.write(f"파일 크기: {file_size:,} bytes")
                
                # 다운로드 버튼
                with open(filename, 'rb') as f:
                    if filename.endswith('.json'):
                        # JSON 파일은 텍스트로 다운로드
                        data = f.read().decode('utf-8')
                        mime_type = "application/json"
                    else:
                        # 바이너리 파일은 바이너리로 다운로드
                        data = f.read()
                        mime_type = "application/octet-stream"
                    
                    st.download_button(
                        label=f"{description} 다운로드",
                        data=data,
                        file_name=filename,
                        mime=mime_type
                    )
            else:
                st.write(f"**{description}** (파일 없음)")
        
        # 통계 정보
        st.subheader("통계 정보")
        
        # 최종 데이터 통계
        if os.path.exists("final_text_only_data.json"):
            data = load_json_file("final_text_only_data.json")
            if data:
                st.write(f"**최종 텍스트 데이터**: {len(data)}개 항목")
        
        # 임베딩 파일 통계
        if os.path.exists("embeddings.faiss"):
            try:
                import faiss
                index = faiss.read_index("embeddings.faiss")
                st.write(f"**FAISS 인덱스**: {index.ntotal}개 벡터, {index.d}차원")
            except Exception as e:
                st.write(f"**FAISS 인덱스**: 로드 실패 - {e}")
        
        if os.path.exists("embeddings_metadata.pkl"):
            try:
                import pickle
                with open("embeddings_metadata.pkl", 'rb') as f:
                    metadata = pickle.load(f)
                st.write(f"**임베딩 메타데이터**: {len(metadata)}개 항목")
            except Exception as e:
                st.write(f"**임베딩 메타데이터**: 로드 실패 - {e}")
        
        # 임베딩 품질 평가 섹션
        st.subheader("임베딩 품질 평가")
        st.markdown("임베딩이 생성된 후 품질을 평가할 수 있습니다.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("자동 QA 생성")
            
            if st.button("LLM으로 질문-답변 쌍 생성", type="primary", key="qa_generation_btn"):
                st.session_state.qa_generation_triggered = True
            
            # QA 생성 프로세스
            if st.session_state.qa_generation_triggered:
                # 컨테이너를 사용하여 안전한 DOM 조작
                with st.container():
                if os.path.exists("embeddings_metadata.pkl"):
                    # QA 생성 단계
                    with st.spinner("LLM을 사용하여 질문-답변 쌍을 생성 중..."):
                        try:
                            cmd = [get_venv_python("venv_rag_new"), "src/qa_pair_generator.py", "embeddings_metadata.pkl", "qa_pairs.json"]
                                st.write(f"실행 명령어: {' '.join(cmd)}")
                                
                                # 실시간 출력을 위해 Popen 사용
                                import subprocess
                                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
                                
                                # 실시간으로 출력 읽기
                                output_lines = []
                                while True:
                                    output = process.stdout.readline()
                                    if output == '' and process.poll() is not None:
                                        break
                                    if output:
                                        output_lines.append(output.strip())
                                        st.write(f"실행 중: {output.strip()}")
                                
                                # 프로세스 완료 대기
                                returncode = process.wait()
                                
                                st.write(f"반환 코드: {returncode}")
                                
                                if returncode == 0:
                                    st.success("질문-답변 쌍 생성 완료!")
                                
                                # 생성된 QA 쌍 로드 및 표시
                                if os.path.exists("qa_pairs.json"):
                                    qa_data = load_json_file("qa_pairs.json")
                                        if qa_data and isinstance(qa_data, dict) and "questions" in qa_data:
                                            # 새로운 JSON 구조에서 questions 배열 추출
                                            questions = qa_data["questions"]
                                            total_questions = qa_data.get("total_questions", len(questions))
                                            st.write(f"**생성된 QA 쌍**: {total_questions}개")
                                        elif isinstance(qa_data, list):
                                            # 기존 구조 (직접 배열)
                                            questions = qa_data
                                            st.write(f"**생성된 QA 쌍**: {len(questions)}개")
                                        else:
                                            st.error("QA 데이터 형식이 올바르지 않습니다.")
                                            return
                                        
                                        # 동적으로 chunk 수 계산
                                        import pickle
                                        try:
                                            with open("embeddings_metadata.pkl", 'rb') as f:
                                                metadata = pickle.load(f)
                                            chunk_count = len(metadata)
                                            expected_count = chunk_count * 3  # chunk당 3개씩
                                        except:
                                            chunk_count = 0
                                            expected_count = 0
                                        
                                        # 실제 QA 쌍 수 계산
                                        actual_count = len(questions) if 'questions' in locals() else 0
                                        
                                        if actual_count < expected_count:
                                            st.warning(f"예상보다 적은 QA 쌍이 생성되었습니다. (예상: {expected_count}개, 실제: {actual_count}개)")
                                            st.info(f"문서 분석: {chunk_count}개 chunk에서 chunk당 3개씩 QA 쌍 생성 예정")
                                        else:
                                            st.success(f"모든 chunk에서 QA 쌍이 성공적으로 생성되었습니다!")
                                            st.info(f"문서 분석: {chunk_count}개 chunk에서 총 {actual_count}개 QA 쌍 생성 완료")
                                        
                                        # 샘플 QA 쌍 표시
                                        if actual_count > 0:
                                            st.write("**샘플 QA 쌍:**")
                                            for i, qa in enumerate(questions[:5]):  # 5개까지 표시
                                                st.write(f"**Q{i+1}**: {qa.get('question', 'N/A')}")
                                                st.write(f"**A{i+1}**: {qa.get('answer', 'N/A')[:100]}...")
                                                st.write("---")
                                
                                # 자동으로 품질 평가 실행
                                with st.spinner("QA 기반 품질 평가를 실행 중..."):
                                    try:
                                                cmd = [get_venv_python("venv_rag_new"), "src/embedding_quality_evaluator.py", ".", "embedding_quality_report_detailed.json"]
                                        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
                                        
                                        if result.returncode == 0:
                                                    st.success("평가가 완료되었습니다!")
                                            st.session_state.qa_generation_triggered = False
                                            
                                            # 평가 결과 표시
                                            if os.path.exists("embedding_quality_report_detailed.json"):
                                                report = load_json_file("embedding_quality_report_detailed.json")
                                                if report:
                                                    st.write("**평가 결과 요약:**")
                                                            
                                                            # 내재적 평가 점수 계산
                                                            intrinsic_data = report.get('intrinsic_evaluation', {})
                                                            intrinsic_score = 0
                                                            if intrinsic_data:
                                                                semantic_corr = intrinsic_data.get('semantic_correlation', {})
                                                                clustering = intrinsic_data.get('clustering_consistency', {})
                                                                analogy = intrinsic_data.get('analogy_performance', {})
                                                                
                                                                if semantic_corr:
                                                                    corr = semantic_corr.get('spearman_correlation', 0)
                                                                    if corr > 0.6:
                                                                        intrinsic_score += 15
                                                                    elif corr > 0.4:
                                                                        intrinsic_score += corr * 20
                                                                    else:
                                                                        intrinsic_score += corr * 15
                                                                if clustering:
                                                                    purity = clustering.get('purity', 0)
                                                                    if purity > 0.8:
                                                                        intrinsic_score += 20
                                                                    elif purity > 0.6:
                                                                        intrinsic_score += purity * 25
                                                                    else:
                                                                        intrinsic_score += purity * 20
                                                                if analogy:
                                                                    accuracy = analogy.get('accuracy', 0)
                                                                    intrinsic_score += accuracy * 5
                                                            
                                                            # 외재적 평가 점수 계산
                                                            extrinsic_data = report.get('extrinsic_evaluation', {})
                                                            extrinsic_score = 0
                                                            if extrinsic_data:
                                                                retrieval = extrinsic_data.get('retrieval_metrics', {})
                                                                qa_accuracy = extrinsic_data.get('qa_accuracy', {})
                                                                
                                                                if retrieval:
                                                                    precision = retrieval.get('precision_at_1', 0)
                                                                    if precision > 0.7:
                                                                        extrinsic_score += 35
                                                                    elif precision > 0.5:
                                                                        extrinsic_score += precision * 45
                                                                    else:
                                                                        extrinsic_score += precision * 35
                                                                    
                                                                    mrr = retrieval.get('mrr', 0)
                                                                    if mrr > 0.7:
                                                                        extrinsic_score += 25
                                                                    elif mrr > 0.5:
                                                                        extrinsic_score += mrr * 35
                                                                    else:
                                                                        extrinsic_score += mrr * 25
                                                                if qa_accuracy:
                                                                    f1 = qa_accuracy.get('f1_score', 0)
                                                                    if f1 > 0.6:
                                                                        extrinsic_score += 40
                                                                    elif f1 > 0.4:
                                                                        extrinsic_score += f1 * 60
                                                                    else:
                                                                        extrinsic_score += f1 * 40
                                                            
                                                            # 벡터 품질 점수 계산 및 표시
                                                            vector_analysis = report.get('vector_analysis', {})
                                                            vector_score = 0
                                                            if vector_analysis:
                                                                st.write("**벡터 품질 정보:**")
                                                                total_vectors = vector_analysis.get('total_vectors', 0)
                                                                vector_dimension = vector_analysis.get('vector_dimension', 0)
                                                                is_normalized = vector_analysis.get('is_normalized', False)
                                                                norm_mean = vector_analysis.get('norm_mean', 0)
                                                                
                                                                st.write(f"  • 총 벡터 수: {total_vectors}")
                                                                st.write(f"  • 벡터 차원: {vector_dimension}")
                                                                st.write(f"  • 정규화 상태: {'정규화됨' if is_normalized else '정규화되지 않음'}")
                                                                st.write(f"  • 평균 노름: {norm_mean:.6f}")
                                                                
                                                                # 벡터 품질 점수 계산 (20점 만점)
                                                                if total_vectors >= 20:
                                                                    vector_score += 5
                                                                elif total_vectors >= 10:
                                                                    vector_score += 3
                                                                else:
                                                                    vector_score += total_vectors * 0.25
                                                                
                                                                if vector_dimension >= 384:
                                                                    vector_score += 5
                                                                elif vector_dimension >= 256:
                                                                    vector_score += 3
                                                                else:
                                                                    vector_score += vector_dimension * 0.02
                                                                
                                                                if is_normalized:
                                                                    vector_score += 5
                                                                else:
                                                                    vector_score += 2
                                                                
                                                                if 0.95 <= norm_mean <= 1.05:
                                                                    vector_score += 5
                                                                elif 0.9 <= norm_mean <= 1.1:
                                                                    vector_score += 3
                                                                else:
                                                                    vector_score += 2
                                                                
                                                                st.write(f"**벡터 품질 점수: {vector_score:.1f}/20점**")
                                                            
                                                            # 종합 점수 계산 (벡터 품질 + 내재적 + 외재적)
                                                            # 외재적 평가 점수 재계산 (모든 세부 항목 포함)
                                                            extrinsic_total = 0
                                                            if extrinsic_data:
                                                                retrieval = extrinsic_data.get('retrieval_metrics', {})
                                                                qa_accuracy = extrinsic_data.get('qa_accuracy', {})
                                                                
                                                                if retrieval:
                                                                    precision = retrieval.get('precision_at_1', 0)
                                                                    mrr = retrieval.get('mrr', 0)
                                                                    map_score = retrieval.get('map', 0)
                                                                    
                                                                    # 검색 정확도 (35점)
                                                                    if precision > 0.7:
                                                                        extrinsic_total += 35
                                                                    elif precision > 0.5:
                                                                        extrinsic_total += precision * 45
                                                                    else:
                                                                        extrinsic_total += precision * 35
                                                                    
                                                                    # MRR (25점)
                                                                    if mrr > 0.7:
                                                                        extrinsic_total += 25
                                                                    elif mrr > 0.5:
                                                                        extrinsic_total += mrr * 35
                                                                    else:
                                                                        extrinsic_total += mrr * 25
                                                                    
                                                                    # MAP (15점)
                                                                    if map_score > 0.6:
                                                                        extrinsic_total += 15
                                                                    elif map_score > 0.4:
                                                                        extrinsic_total += map_score * 25
                                                                    else:
                                                                        extrinsic_total += map_score * 20
                                                                
                                                                if qa_accuracy:
                                                                    f1 = qa_accuracy.get('f1_score', 0)
                                                                    precision_qa = qa_accuracy.get('precision', 0)
                                                                    recall_qa = qa_accuracy.get('recall', 0)
                                                                    
                                                                    # QA F1점수 (25점)
                                                                    if f1 > 0.6:
                                                                        extrinsic_total += 25
                                                                    elif f1 > 0.4:
                                                                        extrinsic_total += f1 * 40
                                                                    else:
                                                                        extrinsic_total += f1 * 25
                                                                    
                                                                    # QA 정밀도 (10점)
                                                                    if precision_qa > 0.6:
                                                                        extrinsic_total += 10
                                                                    elif precision_qa > 0.4:
                                                                        extrinsic_total += precision_qa * 15
                                                                    else:
                                                                        extrinsic_total += precision_qa * 10
                                                                    
                                                                    # QA 재현율 (5점)
                                                                    if recall_qa > 0.7:
                                                                        extrinsic_total += 5
                                                                    elif recall_qa > 0.5:
                                                                        extrinsic_total += recall_qa * 8
                                                                    else:
                                                                        extrinsic_total += recall_qa * 5
                                                            
                                                            # 종합 점수 계산 (벡터 품질 20점 + 내재적 40점 + 외재적 40점)
                                                            overall_score = vector_score + intrinsic_score + extrinsic_total
                                                            st.write(f"종합 점수: {overall_score:.1f}/100점")
                                                            st.write(f"벡터 품질: {vector_score:.1f}점")
                                                            st.write(f"내재적 평가: {intrinsic_score:.1f}점")
                                                            st.write(f"외재적 평가: {extrinsic_total:.1f}점")
                                                            
                                                            # 내재적 평가 점수 표시
                                                            st.write(f"내재적 평가: {intrinsic_score:.1f}")
                                                            
                                                            # 내재적 평가 세부 항목 표시
                                                            st.write("**내재적 평가 세부 항목:**")
                                                            intrinsic_details = []
                                                            if semantic_corr:
                                                                corr = semantic_corr.get('spearman_correlation', 0)
                                                                if corr > 0.6:
                                                                    score = 15
                                                                    intrinsic_details.append(f"  • 의미적 상관관계: {corr:.3f} (우수) = {score}점")
                                                                elif corr > 0.4:
                                                                    score = corr * 20
                                                                    intrinsic_details.append(f"  • 의미적 상관관계: {corr:.3f} × 20 = {score:.1f}점")
                                                                else:
                                                                    score = corr * 15
                                                                    intrinsic_details.append(f"  • 의미적 상관관계: {corr:.3f} × 15 = {score:.1f}점")
                                                            if clustering:
                                                                purity = clustering.get('purity', 0)
                                                                nmi = clustering.get('nmi', 0)
                                                                if purity > 0.8:
                                                                    score = 20
                                                                    intrinsic_details.append(f"  • 클러스터링 일관성: Purity={purity:.3f} (우수) = {score}점")
                                                                elif purity > 0.6:
                                                                    score = purity * 25
                                                                    intrinsic_details.append(f"  • 클러스터링 일관성: Purity={purity:.3f} × 25 = {score:.1f}점")
                                                                else:
                                                                    score = purity * 20
                                                                    intrinsic_details.append(f"  • 클러스터링 일관성: Purity={purity:.3f} × 20 = {score:.1f}점")
                                                            if analogy:
                                                                accuracy = analogy.get('accuracy', 0)
                                                                score = accuracy * 5
                                                                intrinsic_details.append(f"  • 유추 성능: {accuracy:.3f} × 5 = {score:.1f}점")
                                                            
                                                            for detail in intrinsic_details:
                                                                st.write(detail)
                                                            
                                                            # 외재적 평가 점수 표시
                                                            st.write(f"외재적 평가: {extrinsic_score:.1f}")
                                                            
                                                            # 외재적 평가 세부 항목 표시
                                                            st.write("**외재적 평가 세부 항목:**")
                                                            extrinsic_details = []
                                                            if extrinsic_data:
                                                                retrieval = extrinsic_data.get('retrieval_metrics', {})
                                                                qa_accuracy = extrinsic_data.get('qa_accuracy', {})
                                                                if retrieval:
                                                                    precision = retrieval.get('precision_at_1', 0)
                                                                    mrr = retrieval.get('mrr', 0)
                                                                    map_score = retrieval.get('map', 0)
                                                                    st.write(f"  • 검색 정확도 (P@1): {precision:.3f}")
                                                                    st.write(f"  • MRR (Mean Reciprocal Rank): {mrr:.3f}")
                                                                    st.write(f"  • MAP (Mean Average Precision): {map_score:.3f}")
                                                                    
                                                                    # 검색 점수 계산 (각각 개별 점수)
                                                                    if precision > 0.7:
                                                                        precision_score = 35
                                                                        extrinsic_details.append(f"  • 검색 정확도: {precision:.3f} (우수) = {precision_score}점")
                                                                    elif precision > 0.5:
                                                                        precision_score = precision * 45
                                                                        extrinsic_details.append(f"  • 검색 정확도: {precision:.3f} × 45 = {precision_score:.1f}점")
                                                                    else:
                                                                        precision_score = precision * 35
                                                                        extrinsic_details.append(f"  • 검색 정확도: {precision:.3f} × 35 = {precision_score:.1f}점")
                                                                    
                                                                    if mrr > 0.7:
                                                                        mrr_score = 25
                                                                        extrinsic_details.append(f"  • MRR: {mrr:.3f} (우수) = {mrr_score}점")
                                                                    elif mrr > 0.5:
                                                                        mrr_score = mrr * 35
                                                                        extrinsic_details.append(f"  • MRR: {mrr:.3f} × 35 = {mrr_score:.1f}점")
                                                                    else:
                                                                        mrr_score = mrr * 25
                                                                        extrinsic_details.append(f"  • MRR: {mrr:.3f} × 25 = {mrr_score:.1f}점")
                                                                    
                                                                    # MAP 점수 계산 (15점 만점)
                                                                    if map_score > 0.6:
                                                                        map_score_calc = 15
                                                                        extrinsic_details.append(f"  • MAP: {map_score:.3f} (우수) = {map_score_calc}점")
                                                                    elif map_score > 0.4:
                                                                        map_score_calc = map_score * 25
                                                                        extrinsic_details.append(f"  • MAP: {map_score:.3f} × 25 = {map_score_calc:.1f}점")
                                                                    else:
                                                                        map_score_calc = map_score * 20
                                                                        extrinsic_details.append(f"  • MAP: {map_score:.3f} × 20 = {map_score_calc:.1f}점")
                                                                if qa_accuracy:
                                                                    f1 = qa_accuracy.get('f1_score', 0)
                                                                    precision_qa = qa_accuracy.get('precision', 0)
                                                                    recall_qa = qa_accuracy.get('recall', 0)
                                                                    st.write(f"  • QA F1점수: {f1:.3f}")
                                                                    st.write(f"  • QA 정밀도: {precision_qa:.3f}")
                                                                    st.write(f"  • QA 재현율: {recall_qa:.3f}")
                                                                    
                                                                    # QA 점수 계산 (각각 개별 점수)
                                                                    if f1 > 0.6:
                                                                        f1_score = 25
                                                                        extrinsic_details.append(f"  • QA F1점수: {f1:.3f} (우수) = {f1_score}점")
                                                                    elif f1 > 0.4:
                                                                        f1_score = f1 * 40
                                                                        extrinsic_details.append(f"  • QA F1점수: {f1:.3f} × 40 = {f1_score:.1f}점")
                                                                    else:
                                                                        f1_score = f1 * 25
                                                                        extrinsic_details.append(f"  • QA F1점수: {f1:.3f} × 25 = {f1_score:.1f}점")
                                                                    
                                                                    # QA 정밀도 점수 계산 (10점 만점)
                                                                    if precision_qa > 0.6:
                                                                        precision_qa_score = 10
                                                                        extrinsic_details.append(f"  • QA 정밀도: {precision_qa:.3f} (우수) = {precision_qa_score}점")
                                                                    elif precision_qa > 0.4:
                                                                        precision_qa_score = precision_qa * 15
                                                                        extrinsic_details.append(f"  • QA 정밀도: {precision_qa:.3f} × 15 = {precision_qa_score:.1f}점")
                                                                    else:
                                                                        precision_qa_score = precision_qa * 10
                                                                        extrinsic_details.append(f"  • QA 정밀도: {precision_qa:.3f} × 10 = {precision_qa_score:.1f}점")
                                                                    
                                                                    # QA 재현율 점수 계산 (5점 만점)
                                                                    if recall_qa > 0.7:
                                                                        recall_qa_score = 5
                                                                        extrinsic_details.append(f"  • QA 재현율: {recall_qa:.3f} (우수) = {recall_qa_score}점")
                                                                    elif recall_qa > 0.5:
                                                                        recall_qa_score = recall_qa * 8
                                                                        extrinsic_details.append(f"  • QA 재현율: {recall_qa:.3f} × 8 = {recall_qa_score:.1f}점")
                                                                    else:
                                                                        recall_qa_score = recall_qa * 5
                                                                        extrinsic_details.append(f"  • QA 재현율: {recall_qa:.3f} × 5 = {recall_qa_score:.1f}점")
                                                                
                                                                for detail in extrinsic_details:
                                                                    st.write(detail)
                                                                
                                                                # 점수 해석 추가
                                                                if extrinsic_score > 80:
                                                                    st.success("외재적 평가: 매우 우수 (실용적 성능이 뛰어남)")
                                                                elif extrinsic_score > 60:
                                                                    st.info("외재적 평가: 양호 (실용적 성능이 좋음)")
                                                                elif extrinsic_score > 40:
                                                                    st.warning("외재적 평가: 보통 (실용적 성능이 개선 필요)")
                                                                else:
                                                                    st.error("외재적 평가: 미흡 (실용적 성능이 크게 개선 필요)")
                                                            else:
                                                                st.write("외재적 평가: N/A")
                                                            
                                                            # 평가 기준 설명
                                                            st.write("**📊 상세 평가 기준 설명:**")
                                                            st.write("")
                                                            st.write("**🔧 벡터 품질 평가 (20점 만점):**")
                                                            st.write("벡터 품질은 임베딩의 기본적인 기술적 특성을 평가합니다.")
                                                            st.write("")
                                                            st.write("  • **총 벡터 수 (5점)**: 문서에서 추출된 벡터의 총 개수")
                                                            st.write("    - 20개 이상: 5점 (충분한 데이터로 의미 있는 패턴 학습 가능)")
                                                            st.write("    - 10개 이상: 3점 (적당한 데이터로 기본적인 패턴 학습 가능)")
                                                            st.write("    - 10개 미만: 개수×0.25점 (데이터 부족으로 패턴 학습 어려움)")
                                                            st.write("")
                                                            st.write("  • **벡터 차원 (5점)**: 각 벡터의 차원 수 (정보량과 복잡도)")
                                                            st.write("    - 384 이상: 5점 (고차원으로 풍부한 의미 정보 포함)")
                                                            st.write("    - 256 이상: 3점 (중간 차원으로 적절한 의미 정보)")
                                                            st.write("    - 256 미만: 차원×0.02점 (저차원으로 제한된 정보)")
                                                            st.write("")
                                                            st.write("  • **정규화 상태 (5점)**: 벡터의 정규화 여부")
                                                            st.write("    - 정규화됨: 5점 (일관된 스케일로 안정적인 유사도 계산)")
                                                            st.write("    - 정규화 안됨: 2점 (스케일 불일치로 유사도 계산 부정확)")
                                                            st.write("")
                                                            st.write("  • **평균 노름 (5점)**: 벡터들의 평균 길이")
                                                            st.write("    - 0.95~1.05: 5점 (이상적인 정규화 상태)")
                                                            st.write("    - 0.9~1.1: 3점 (적절한 정규화 상태)")
                                                            st.write("    - 그 외: 2점 (정규화 상태 불량)")
                                                            st.write("")
                                                            st.write("**🧠 내재적 평가 (40점 만점):**")
                                                            st.write("내재적 평가는 벡터 자체의 의미적 품질과 구조적 특성을 평가합니다.")
                                                            st.write("")
                                                            st.write("  • **의미적 상관관계 (15점)**: 벡터 간 의미적 유사도가 인간 판단과 일치하는 정도")
                                                            st.write("    - 0.6 이상: 15점 (매우 높은 의미적 일치도)")
                                                            st.write("    - 0.4 이상: corr×20점 (높은 의미적 일치도)")
                                                            st.write("    - 0.4 미만: corr×15점 (낮은 의미적 일치도)")
                                                            st.write("    - *의미: 유사한 내용의 벡터가 실제로 가까운 위치에 있는지 확인*")
                                                            st.write("")
                                                            st.write("  • **클러스터링 일관성 (20점)**: 유사한 내용이 같은 그룹으로 묶이는 정도")
                                                            st.write("    - 0.8 이상: 20점 (매우 우수한 클러스터링 성능)")
                                                            st.write("    - 0.6 이상: purity×25점 (우수한 클러스터링 성능)")
                                                            st.write("    - 0.6 미만: purity×20점 (보통의 클러스터링 성능)")
                                                            st.write("    - *의미: 의미적으로 유사한 내용들이 실제로 같은 그룹으로 분류되는지 확인*")
                                                            st.write("")
                                                            st.write("  • **유추 성능 (5점)**: 의미적 유사도를 통한 유추 문제 해결 능력")
                                                            st.write("    - accuracy×5점 (유추 문제의 정확도)")
                                                            st.write("    - *의미: 'A:B = C:?' 형태의 유추 문제를 얼마나 잘 해결하는지 확인*")
                                                            st.write("")
                                                            st.write("**🎯 외재적 평가 (40점 만점):**")
                                                            st.write("외재적 평가는 실제 사용 시나리오에서의 성능을 평가합니다.")
                                                            st.write("")
                                                            st.write("  • **검색 정확도 (15점)**: 질문에 대한 정확한 답변 검색 성능")
                                                            st.write("    - 0.7 이상: 15점 (매우 높은 검색 정확도)")
                                                            st.write("    - 0.5 이상: precision×20점 (높은 검색 정확도)")
                                                            st.write("    - 0.5 미만: precision×15점 (보통의 검색 정확도)")
                                                            st.write("    - *의미: 질문과 가장 관련성 높은 답변이 1순위로 검색되는 비율*")
                                                            st.write("")
                                                            st.write("  • **MRR (10점)**: 정답이 상위 순위에 나타나는 정도")
                                                            st.write("    - 0.7 이상: 10점 (매우 높은 순위 정확도)")
                                                            st.write("    - 0.5 이상: mrr×15점 (높은 순위 정확도)")
                                                            st.write("    - 0.5 미만: mrr×10점 (보통의 순위 정확도)")
                                                            st.write("    - *의미: 정답이 평균적으로 몇 번째 순위에 나타나는지 (낮을수록 좋음)*")
                                                            st.write("")
                                                            st.write("  • **MAP (5점)**: 평균 정밀도의 평균")
                                                            st.write("    - 0.6 이상: 5점 (매우 높은 평균 정밀도)")
                                                            st.write("    - 0.4 이상: map×8점 (높은 평균 정밀도)")
                                                            st.write("    - 0.4 미만: map×5점 (보통의 평균 정밀도)")
                                                            st.write("    - *의미: 여러 질문에 대해 정답이 상위 순위에 나타나는 평균적 성능*")
                                                            st.write("")
                                                            st.write("  • **QA F1점수 (5점)**: 질문-답변 쌍의 정확성과 완성도")
                                                            st.write("    - 0.6 이상: 5점 (매우 높은 QA 정확도)")
                                                            st.write("    - 0.4 이상: f1×8점 (높은 QA 정확도)")
                                                            st.write("    - 0.4 미만: f1×5점 (보통의 QA 정확도)")
                                                            st.write("    - *의미: 생성된 답변이 정답과 얼마나 일치하는지 (정밀도와 재현율의 조화평균)*")
                                                            st.write("")
                                                            st.write("  • **QA 정밀도 (3점)**: 생성된 답변 중 정답인 비율")
                                                            st.write("    - 0.6 이상: 3점 (매우 높은 정밀도)")
                                                            st.write("    - 0.4 이상: precision×5점 (높은 정밀도)")
                                                            st.write("    - 0.4 미만: precision×3점 (보통의 정밀도)")
                                                            st.write("    - *의미: 생성된 답변이 얼마나 정확한지 (틀린 답변을 얼마나 적게 생성하는지)*")
                                                            st.write("")
                                                            st.write("  • **QA 재현율 (2점)**: 정답 중에서 생성된 비율")
                                                            st.write("    - 0.7 이상: 2점 (매우 높은 재현율)")
                                                            st.write("    - 0.5 이상: recall×3점 (높은 재현율)")
                                                            st.write("    - 0.5 미만: recall×2점 (보통의 재현율)")
                                                            st.write("    - *의미: 정답을 얼마나 많이 찾아내는지 (빠뜨리는 답변을 얼마나 적게 하는지)*")
                                                            st.write("")
                                                            st.write("**📈 종합 점수 해석:**")
                                                            st.write("  • 90점 이상: 매우 우수 (프로덕션 환경에서 사용 가능)")
                                                            st.write("  • 80-89점: 우수 (대부분의 사용 사례에 적합)")
                                                            st.write("  • 70-79점: 양호 (일반적인 사용에 적합)")
                                                            st.write("  • 60-69점: 보통 (개선이 권장됨)")
                                                            st.write("  • 60점 미만: 미흡 (대폭 개선 필요)")
                                        else:
                                            st.error("품질 평가 실행 실패!")
                                            if result.stderr:
                                                st.error(f"오류: {result.stderr}")
                                    except Exception as e:
                                        st.error(f"품질 평가 중 오류 발생: {e}")
                            else:
                                st.error("QA 쌍 생성 실패!")
                                if result.stderr:
                                    st.error(f"오류: {result.stderr}")
                        except Exception as e:
                            st.error(f"QA 생성 중 오류 발생: {e}")
                else:
                    st.error("임베딩 메타데이터 파일이 없습니다. 먼저 임베딩을 생성해주세요.")
        
        with col2:
            st.subheader("품질 평가 정보")
            st.info("""
            **QA 생성 버튼을 클릭하면:**
            1. LLM으로 동적 질문-답변 쌍 생성
            2. 자동으로 QA 기반 품질 평가 실행
            3. 상세한 평가 보고서 생성
            
            **평가 항목:**
            - 내재적 평가 (벡터 품질, 의미적 유사도)
            - 외재적 평가 (콘텐츠 품질, QA 검색 성능)
            - 종합 점수 및 개선 권장사항
            
            **🆕 완전 동적 시스템:**
            - 문서 내용 기반 자동 QA 생성
            - 어떤 문서든 자동 적응 가능
            - 하드코딩 없는 유연한 평가
            """)
    
    with tab3:
        st.header("시스템 정보")
        
        # 가상환경 상태 확인
        st.subheader("가상환경 상태")
        venv_status = {}
        
        for venv_name in ["venv_web_new", "venv_text_new", "venv_rag_new"]:
            venv_path = f"environments/{venv_name}"
            if os.path.exists(venv_path):
                venv_status[venv_name] = "활성"
            else:
                venv_status[venv_name] = "비활성"
        
        for venv_name, status in venv_status.items():
            st.write(f"**{venv_name}**: {status}")
        
        # 파일 시스템 상태
        st.subheader("파일 시스템 상태")
        important_files = [
            "input.pdf",
            "final_text_only_data.json",
            "embeddings.faiss",
            "embeddings_metadata.pkl",
            "embedding_quality_report.json",
            "qa_pairs.json"
        ]
        
        for file in important_files:
            if os.path.exists(file):
                size = os.path.getsize(file)
                st.write(f"**{file}**: 사용 가능 ({size:,} bytes)")
            else:
                st.write(f"**{file}**: 없음")

if __name__ == "__main__":
    main() 