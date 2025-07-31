#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit 앱 - 텍스트 전용 버전
표 추출 제외하고 텍스트만 추출하는 시스템
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
        page_title="PDF 텍스트 추출 & 임베딩 시스템",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="collapsed",
        menu_items={
            'Get Help': None,
            'Report a bug': None,
            'About': None
        }
    )
    st.title("📄 PDF 텍스트 추출 & 임베딩 품질 평가 시스템")
    tab1, tab2, tab3 = st.tabs(["파일 업로드 & 추출", "결과 확인", "시스템 정보"])

    with tab1:
        st.header("파일 업로드 및 데이터 관리")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ 기존 데이터 삭제", key="delete_data_btn"):
                if 'delete_triggered' not in st.session_state:
                    st.session_state.delete_triggered = False
                
                if not st.session_state.delete_triggered:
                    st.session_state.delete_triggered = True
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
                        st.success(f"✅ {deleted_count}개 파일 삭제 완료!")
                    else:
                        st.info("삭제할 파일이 없습니다.")
                    
                    # 상태 초기화
                    st.session_state.delete_triggered = False
        with col2:
            if st.button("🧹 캐시 및 임시 파일 삭제", key="delete_cache_btn"):
                if 'cache_delete_triggered' not in st.session_state:
                    st.session_state.cache_delete_triggered = False
                
                if not st.session_state.cache_delete_triggered:
                    st.session_state.cache_delete_triggered = True
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
                    
                    # 상태 초기화
                    st.session_state.cache_delete_triggered = False
        # 파일 업로드
        uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type=['pdf'])
        if uploaded_file is not None:
            with open("input.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"{uploaded_file.name} 업로드 완료!")
            st.subheader("추출 옵션")
            if st.button("하이브리드 시스템 (텍스트 전용)", type="primary", key="hybrid_system_btn"):
                venv_python = get_venv_python("venv_text_new")
                cmd = [venv_python, "src/hybrid_extraction_system.py", "input.pdf"]
                run_safe_subprocess(cmd, "하이브리드 시스템")

    with tab2:
        st.header("추출 결과 확인")
        result_files = [
            ("final_text_only_data.json", "최종 텍스트 데이터 (JSON)"),
            ("embeddings.faiss", "FAISS 임베딩 인덱스"),
            ("embeddings_metadata.pkl", "임베딩 메타데이터"),
            ("embedding_quality_report.json", "임베딩 품질 평가 보고서")
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
        st.subheader("통계 정보")
        if os.path.exists("final_text_only_data.json"):
            data = load_json_file("final_text_only_data.json")
            if data:
                st.write(f"**최종 텍스트 데이터**: {len(data)}개 항목")
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
        st.subheader("🤖 임베딩 품질 평가")
        st.markdown("임베딩이 생성된 후 품질을 평가할 수 있습니다.")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("자동 QA 생성")
            if 'qa_generation_triggered' not in st.session_state:
                st.session_state.qa_generation_triggered = False
            if st.button("🔄 LLM으로 질문-답변 쌍 생성", type="primary", key="qa_generation_btn"):
                if 'qa_generation_triggered' not in st.session_state:
                    st.session_state.qa_generation_triggered = False
                st.session_state.qa_generation_triggered = True
            
            if st.session_state.qa_generation_triggered:
                if os.path.exists("embeddings_metadata.pkl"):
                    # QA 쌍 생성
                    st.info("LLM을 사용하여 질문-답변 쌍을 생성 중...")
                    try:
                        cmd = [get_venv_python("venv_rag_new"), "src/qa_pair_generator.py", "final_text_only_data.json", "qa_pairs.json"]
                        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
                        if result.returncode == 0:
                            st.success("✅ 질문-답변 쌍 생성 완료!")
                            if os.path.exists("qa_pairs.json"):
                                qa_data = load_json_file("qa_pairs.json")
                                if qa_data:
                                    st.write(f"**생성된 QA 쌍**: {len(qa_data)}개")
                                    if len(qa_data) > 0:
                                        st.write("**샘플 QA 쌍:**")
                                        for i, qa in enumerate(qa_data[:3]):
                                            st.write(f"**Q{i+1}**: {qa.get('question', 'N/A')}")
                                            st.write(f"**A{i+1}**: {qa.get('answer', 'N/A')[:100]}...")
                                            st.write("---")
                            
                            # QA 기반 품질 평가
                            st.info("QA 기반 품질 평가를 실행 중...")
                            try:
                                cmd = [get_venv_python("venv_rag_new"), "src/embedding_quality_evaluator.py"]
                                result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
                                if result.returncode == 0:
                                    st.success("🎉 평가가 완료되었습니다!")
                                    st.session_state.qa_generation_triggered = False
                                    if os.path.exists("embedding_quality_report_detailed.json"):
                                        report = load_json_file("embedding_quality_report_detailed.json")
                                        if report:
                                            st.write("**평가 결과 요약:**")
                                            st.write(f"종합 점수: {report.get('overall_score', 'N/A')}")
                                            st.write(f"내재적 평가: {report.get('intrinsic_score', 'N/A')}")
                                            st.write(f"외재적 평가: {report.get('extrinsic_score', 'N/A')}")
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

if __name__ == "__main__":
    main() 