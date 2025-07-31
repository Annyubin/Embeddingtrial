#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
향상된 Streamlit 앱 - 새로운 구조화된 파이프라인 사용
표 제외, overlap, LLM 메타데이터 생성까지 완전한 시스템
"""

import streamlit as st
import os
import subprocess
import json
import pickle

def get_venv_python(venv_name):
    venv_path = f"environments/{venv_name}/bin/python"
    if os.path.exists(venv_path):
        return venv_path
    else:
        st.error(f"가상환경을 찾을 수 없습니다: {venv_path}")
        return None

def run_safe_subprocess(cmd, description):
    try:
        st.info(f"{description} 실행 중...")
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
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"JSON 파일 로드 실패: {e}")
        return None

def main():
    st.set_page_config(
        page_title="향상된 PDF 구조화 & 임베딩 시스템",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="collapsed",
        menu_items={
            'Get Help': None,
            'Report a bug': None,
            'About': None
        }
    )
    st.title("🚀 향상된 PDF 구조화 & 임베딩 시스템")
    st.markdown("**표 제외, Overlap, LLM 메타데이터 생성까지 완전한 파이프라인**")
    
    tab1, tab2, tab3, tab4 = st.tabs(["파일 업로드 & 구조화", "결과 확인", "임베딩 생성", "시스템 정보"])

    with tab1:
        st.header("📄 파일 업로드 및 구조화 파이프라인")
        
        # 데이터 관리 버튼
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ 기존 데이터 삭제", key="delete_data_btn"):
                files_to_delete = [
                    "text_only_chunks.json",
                    "final_text_only_data.json",
                    "full_pipeline_llm_enhanced_chunks.json",
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
                    st.success(f"✅ {deleted_count}개 파일 삭제 완료!")
                else:
                    st.info("삭제할 파일이 없습니다.")
        
        with col2:
            if st.button("🧹 캐시 및 임시 파일 삭제", key="delete_cache_btn"):
                cache_files = [f for f in os.listdir('.') if f.endswith('.tmp') or f.endswith('.cache')]
                deleted_count = 0
                for file in cache_files:
                    try:
                        os.remove(file)
                        deleted_count += 1
                    except Exception as e:
                        st.error(f"파일 삭제 실패: {file} - {e}")
                
                if deleted_count > 0:
                    st.success(f"✅ {deleted_count}개 캐시 파일 삭제 완료!")
                else:
                    st.info("삭제할 캐시 파일이 없습니다.")
        
        # 파일 업로드
        uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type=['pdf'])
        if uploaded_file is not None:
            with open("input.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"✅ {uploaded_file.name} 업로드 완료!")
            
            st.subheader("🚀 향상된 구조화 파이프라인")
            
            # 파이프라인 옵션
            pipeline_option = st.selectbox(
                "파이프라인 선택",
                [
                    "🆕 새로운 완전 구조화 파이프라인 (추천)",
                    "기존 하이브리드 시스템"
                ]
            )
            
            if st.button("🚀 구조화 파이프라인 실행", type="primary", key="enhanced_pipeline_btn"):
                if "새로운 완전 구조화 파이프라인" in pipeline_option:
                    # 새로운 구조화된 파이프라인 실행
                    st.info("🆕 새로운 완전 구조화 파이프라인 실행 중...")
                    
                    # 1단계: 새로운 구조화된 파이프라인
                    cmd1 = ["python", "test_full_pipeline_with_llm.py"]
                    result1 = subprocess.run(cmd1, capture_output=True, text=True, encoding='utf-8')
                    
                    if result1.returncode == 0:
                        st.success("✅ 1단계: 구조화된 파이프라인 완료!")
                        
                        # 2단계: Streamlit 형식으로 변환
                        cmd2 = ["python", "src/streamlit_integration.py", "convert"]
                        result2 = subprocess.run(cmd2, capture_output=True, text=True, encoding='utf-8')
                        
                        if result2.returncode == 0:
                            st.success("✅ 2단계: Streamlit 형식 변환 완료!")
                            
                            # 결과 통계 표시
                            if os.path.exists("final_text_only_data.json"):
                                data = load_json_file("final_text_only_data.json")
                                if data:
                                    st.success(f"🎉 완료! {len(data)}개 청크 생성")
                                    
                                    # 통계 정보
                                    total_length = sum(item.get("content_length", 0) for item in data)
                                    avg_length = total_length / len(data) if data else 0
                                    
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("청크 수", len(data))
                                    with col2:
                                        st.metric("총 텍스트 길이", f"{total_length:,}자")
                                    with col3:
                                        st.metric("평균 청크 길이", f"{avg_length:.1f}자")
                        else:
                            st.error("❌ Streamlit 형식 변환 실패!")
                            if result2.stderr:
                                st.error(f"오류: {result2.stderr}")
                    else:
                        st.error("❌ 구조화된 파이프라인 실패!")
                        if result1.stderr:
                            st.error(f"오류: {result1.stderr}")
                else:
                    # 기존 하이브리드 시스템
                    venv_python = get_venv_python("venv_text_new")
                    cmd = [venv_python, "src/hybrid_extraction_system.py", "input.pdf"]
                    run_safe_subprocess(cmd, "하이브리드 시스템")

    with tab2:
        st.header("📊 구조화 결과 확인")
        
        # 결과 파일들 확인
        result_files = [
            ("final_text_only_data.json", "최종 구조화 데이터 (JSON)"),
            ("full_pipeline_llm_enhanced_chunks.json", "원본 구조화 데이터 (JSON)"),
            ("embeddings.faiss", "FAISS 임베딩 인덱스"),
            ("embeddings_metadata.pkl", "임베딩 메타데이터")
        ]
        
        for filename, description in result_files:
            if os.path.exists(filename):
                st.write(f"**{description}** ✅")
                file_size = os.path.getsize(filename)
                st.write(f"파일 크기: {file_size:,} bytes")
                
                # 파일 다운로드 버튼 생성
                try:
                    if filename.endswith('.json'):
                        with open(filename, 'r', encoding='utf-8') as f:
                            data = f.read()
                        mime_type = "application/json"
                    else:
                        with open(filename, 'rb') as f:
                            data = f.read()
                        mime_type = "application/octet-stream"
                    
                    st.download_button(
                        label=f"{description} 다운로드",
                        data=data,
                        file_name=filename,
                        mime=mime_type
                    )
                except Exception as e:
                    st.error(f"파일 읽기 실패: {e}")
            else:
                st.write(f"**{description}** ❌ (파일 없음)")
        
        # 구조화된 데이터 상세 정보
        if os.path.exists("final_text_only_data.json"):
            data = load_json_file("final_text_only_data.json")
            if data:
                st.subheader("📈 구조화 데이터 통계")
                
                # 기본 통계
                total_chunks = len(data)
                total_length = sum(item.get("content_length", 0) for item in data)
                avg_length = total_length / total_chunks if total_chunks > 0 else 0
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("총 청크 수", total_chunks)
                with col2:
                    st.metric("총 텍스트 길이", f"{total_length:,}자")
                with col3:
                    st.metric("평균 청크 길이", f"{avg_length:.1f}자")
                with col4:
                    llm_processed = sum(1 for item in data if item.get("extraction_method") == "llm_ollama")
                    st.metric("LLM 처리된 청크", llm_processed)
                
                # 샘플 데이터 표시
                st.subheader("📋 샘플 구조화 데이터")
                if len(data) > 0:
                    sample_data = data[:3]  # 처음 3개 청크
                    for i, chunk in enumerate(sample_data):
                        with st.expander(f"청크 {i+1}: {chunk.get('section_title', '제목 없음')}"):
                            st.write(f"**내용**: {chunk.get('content', '')[:200]}...")
                            st.write(f"**페이지**: {chunk.get('page', 0)}")
                            st.write(f"**키워드**: {', '.join(chunk.get('keywords', []))}")
                            st.write(f"**요약**: {chunk.get('summary', '')}")
                            st.write(f"**신뢰도**: {chunk.get('confidence', 0.5):.2f}")

    with tab3:
        st.header("🤖 임베딩 생성")
        
        if os.path.exists("final_text_only_data.json"):
            st.success("✅ 구조화된 데이터가 준비되었습니다!")
            
            if st.button("🚀 임베딩 생성", type="primary", key="embedding_btn"):
                venv_python = get_venv_python("venv_rag_new")
                cmd = [venv_python, "src/embedding_generator.py"]
                run_safe_subprocess(cmd, "임베딩 생성")
                
                # 임베딩 품질 평가
                if os.path.exists("embeddings.faiss"):
                    st.success("✅ 임베딩 생성 완료!")
                    
                    # QA 쌍 생성
                    if st.button("🔄 QA 쌍 생성 및 품질 평가", type="primary", key="qa_btn"):
                        cmd_qa = [venv_python, "src/qa_pair_generator.py", "final_text_only_data.json", "qa_pairs.json"]
                        result_qa = subprocess.run(cmd_qa, capture_output=True, text=True, encoding='utf-8')
                        
                        if result_qa.returncode == 0:
                            st.success("✅ QA 쌍 생성 완료!")
                            
                            # 품질 평가
                            cmd_eval = [venv_python, "src/embedding_quality_evaluator.py"]
                            result_eval = subprocess.run(cmd_eval, capture_output=True, text=True, encoding='utf-8')
                            
                            if result_eval.returncode == 0:
                                st.success("🎉 품질 평가 완료!")
                                
                                # 평가 결과 표시
                                if os.path.exists("embedding_quality_report_detailed.json"):
                                    report = load_json_file("embedding_quality_report_detailed.json")
                                    if report:
                                        st.subheader("📊 품질 평가 결과")
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("종합 점수", f"{report.get('overall_score', 0):.2f}")
                                        with col2:
                                            st.metric("내재적 평가", f"{report.get('intrinsic_score', 0):.2f}")
                                        with col3:
                                            st.metric("외재적 평가", f"{report.get('extrinsic_score', 0):.2f}")
                            else:
                                st.error("❌ 품질 평가 실패!")
                        else:
                            st.error("❌ QA 쌍 생성 실패!")
        else:
            st.warning("⚠️ 먼저 구조화된 데이터를 생성해주세요.")

    with tab4:
        st.header("🔧 시스템 정보")
        
        st.subheader("가상환경 상태")
        venv_status = {}
        for venv_name in ["venv_web_new", "venv_text_new", "venv_rag_new"]:
            venv_path = f"environments/{venv_name}"
            if os.path.exists(venv_path):
                venv_status[venv_name] = "✅ 활성"
            else:
                venv_status[venv_name] = "❌ 비활성"
        for venv_name, status in venv_status.items():
            st.write(f"**{venv_name}**: {status}")
        
        st.subheader("파일 시스템 상태")
        important_files = [
            "input.pdf",
            "final_text_only_data.json",
            "full_pipeline_llm_enhanced_chunks.json",
            "embeddings.faiss",
            "embeddings_metadata.pkl",
            "embedding_quality_report.json",
            "qa_pairs.json"
        ]
        for file in important_files:
            if os.path.exists(file):
                size = os.path.getsize(file)
                st.write(f"**{file}**: ✅ ({size:,} bytes)")
            else:
                st.write(f"**{file}**: ❌ (없음)")
        
        st.subheader("🆕 새로운 기능")
        st.info("""
        **새로운 완전 구조화 파이프라인:**
        - ✅ 표 제외 기능
        - ✅ Overlap 청킹 (30%)
        - ✅ LLM 메타데이터 생성 (섹션 제목, 키워드, 요약)
        - ✅ 영어 참고문헌 자동 필터링
        - ✅ JSON 파싱 강화 (100% 성공률)
        - ✅ 완전한 메타데이터 구조
        
        **기존 시스템과 완전 호환:**
        - ✅ Streamlit 앱과 호환
        - ✅ 임베딩 생성 시스템과 호환
        - ✅ 품질 평가 시스템과 호환
        """)

if __name__ == "__main__":
    main() 