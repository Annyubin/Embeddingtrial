#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
안정적인 Streamlit 앱 - DOM 충돌 문제 해결
"""

import streamlit as st
import os
import subprocess
import json
import pickle
import time

def get_venv_python(venv_name):
    venv_path = f"environments/{venv_name}/bin/python"
    if os.path.exists(venv_path):
        return venv_path
    else:
        st.error(f"가상환경을 찾을 수 없습니다: {venv_path}")
        return None

def load_json_file(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"JSON 파일 로드 실패: {e}")
        return None

def safe_file_operation(operation_func, *args, **kwargs):
    """안전한 파일 작업 수행"""
    try:
        return operation_func(*args, **kwargs)
    except Exception as e:
        st.error(f"파일 작업 실패: {e}")
        return None

def main():
    st.set_page_config(
        page_title="PDF 텍스트 추출 시스템",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    st.title("📄 PDF 텍스트 추출 & 임베딩 시스템")
    
    # 세션 상태 초기화
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "upload"
    
    # 탭 선택
    tab_selection = st.sidebar.selectbox(
        "메뉴 선택",
        ["파일 업로드", "결과 확인", "시스템 정보"],
        index=0
    )
    
    if tab_selection == "파일 업로드":
        st.header("📤 파일 업로드 및 데이터 관리")
        
        # 파일 관리 버튼들
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ 기존 데이터 삭제", key="delete_btn"):
                with st.spinner("데이터 삭제 중..."):
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
                                st.error(f"파일 삭제 실패: {file}")
                    
                    if deleted_count > 0:
                        st.success(f"✅ {deleted_count}개 파일 삭제 완료!")
                    else:
                        st.info("삭제할 파일이 없습니다.")
        
        with col2:
            if st.button("🧹 캐시 파일 삭제", key="cache_btn"):
                with st.spinner("캐시 삭제 중..."):
                    cache_files = [f for f in os.listdir('.') if f.endswith('.tmp') or f.endswith('.cache')]
                    deleted_count = 0
                    for file in cache_files:
                        try:
                            os.remove(file)
                            deleted_count += 1
                        except Exception as e:
                            st.error(f"캐시 삭제 실패: {file}")
                    
                    if deleted_count > 0:
                        st.success(f"✅ {deleted_count}개 캐시 파일 삭제 완료!")
                    else:
                        st.info("삭제할 캐시 파일이 없습니다.")
        
        # 파일 업로드
        st.subheader("📁 PDF 파일 업로드")
        uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type=['pdf'], key="file_uploader")
        
        if uploaded_file is not None:
            # 파일 저장
            with open("input.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"✅ {uploaded_file.name} 업로드 완료!")
            
            # 추출 옵션
            st.subheader("🔧 추출 옵션")
            if st.button("🚀 하이브리드 시스템 실행", type="primary", key="hybrid_btn"):
                with st.spinner("하이브리드 시스템 실행 중..."):
                    venv_python = get_venv_python("venv_text_new")
                    if venv_python:
                        cmd = [venv_python, "src/hybrid_extraction_system.py", "input.pdf"]
                        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
                        
                        if result.returncode == 0:
                            st.success("✅ 하이브리드 시스템 실행 완료!")
                            if result.stdout:
                                st.text_area("실행 결과:", result.stdout, height=200)
                        else:
                            st.error("❌ 하이브리드 시스템 실행 실패!")
                            if result.stderr:
                                st.text_area("오류:", result.stderr, height=200)
    
    elif tab_selection == "결과 확인":
        st.header("📊 추출 결과 확인")
        
        # 파일 상태 확인
        st.subheader("📁 파일 상태")
        files_status = [
            ("final_text_only_data.json", "최종 텍스트 데이터"),
            ("embeddings.faiss", "FAISS 임베딩 인덱스"),
            ("embeddings_metadata.pkl", "임베딩 메타데이터"),
            ("embedding_quality_report.json", "임베딩 품질 평가")
        ]
        
        for filename, description in files_status:
            if os.path.exists(filename):
                size = os.path.getsize(filename)
                st.success(f"✅ {description}: {size:,} bytes")
                
                # 다운로드 버튼
                if filename.endswith('.json'):
                    try:
                        with open(filename, 'r', encoding='utf-8') as f:
                            data = f.read()
                        st.download_button(
                            label=f"{description} 다운로드",
                            data=data,
                            file_name=filename,
                            mime="application/json"
                        )
                    except Exception as e:
                        st.error(f"파일 읽기 실패: {e}")
                else:
                    try:
                        with open(filename, 'rb') as f:
                            data = f.read()
                        st.download_button(
                            label=f"{description} 다운로드",
                            data=data,
                            file_name=filename,
                            mime="application/octet-stream"
                        )
                    except Exception as e:
                        st.error(f"파일 읽기 실패: {e}")
            else:
                st.error(f"❌ {description}: 파일 없음")
        
        # 통계 정보
        st.subheader("📈 통계 정보")
        if os.path.exists("final_text_only_data.json"):
            data = load_json_file("final_text_only_data.json")
            if data:
                st.info(f"📄 최종 텍스트 데이터: {len(data)}개 항목")
        
        if os.path.exists("embeddings.faiss"):
            try:
                import faiss
                index = faiss.read_index("embeddings.faiss")
                st.info(f"🔍 FAISS 인덱스: {index.ntotal}개 벡터, {index.d}차원")
            except Exception as e:
                st.error(f"FAISS 인덱스 로드 실패: {e}")
        
        # QA 생성 및 평가
        st.subheader("🤖 QA 생성 및 품질 평가")
        
        if st.button("🔄 QA 쌍 생성 및 평가", type="primary", key="qa_eval_btn"):
            if os.path.exists("final_text_only_data.json"):
                with st.spinner("QA 쌍 생성 중..."):
                    venv_python = get_venv_python("venv_rag_new")
                    if venv_python:
                        # QA 쌍 생성
                        cmd = [venv_python, "src/qa_pair_generator.py", "final_text_only_data.json"]
                        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
                        
                        if result.returncode == 0:
                            st.success("✅ QA 쌍 생성 완료!")
                            
                            # 품질 평가
                            with st.spinner("품질 평가 실행 중..."):
                                cmd = [venv_python, "src/embedding_quality_evaluator.py"]
                                result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
                                
                                if result.returncode == 0:
                                    st.success("🎉 품질 평가 완료!")
                                    
                                    if os.path.exists("embedding_quality_report_detailed.json"):
                                        report = load_json_file("embedding_quality_report_detailed.json")
                                        if report:
                                            st.subheader("📊 평가 결과")
                                            col1, col2, col3 = st.columns(3)
                                            with col1:
                                                st.metric("종합 점수", report.get('overall_score', 'N/A'))
                                            with col2:
                                                st.metric("내재적 평가", report.get('intrinsic_score', 'N/A'))
                                            with col3:
                                                st.metric("외재적 평가", report.get('extrinsic_score', 'N/A'))
                                else:
                                    st.error("❌ 품질 평가 실패!")
                                    if result.stderr:
                                        st.text_area("오류:", result.stderr, height=100)
                        else:
                            st.error("❌ QA 쌍 생성 실패!")
                            if result.stderr:
                                st.text_area("오류:", result.stderr, height=100)
            else:
                st.error("❌ 텍스트 데이터 파일이 없습니다. 먼저 추출을 실행해주세요.")
    
    elif tab_selection == "시스템 정보":
        st.header("ℹ️ 시스템 정보")
        
        # 가상환경 상태
        st.subheader("🐍 가상환경 상태")
        venv_status = {}
        for venv_name in ["venv_web_new", "venv_text_new", "venv_rag_new"]:
            venv_path = f"environments/{venv_name}"
            if os.path.exists(venv_path):
                venv_status[venv_name] = "✅ 활성"
            else:
                venv_status[venv_name] = "❌ 비활성"
        
        for venv_name, status in venv_status.items():
            st.write(f"**{venv_name}**: {status}")
        
        # 파일 시스템 상태
        st.subheader("📁 파일 시스템 상태")
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
                st.success(f"✅ {file}: {size:,} bytes")
            else:
                st.error(f"❌ {file}: 없음")

if __name__ == "__main__":
    main() 