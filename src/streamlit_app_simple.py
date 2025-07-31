#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
간단한 Streamlit 앱 - 안정적인 버전
JavaScript 오류를 피하기 위해 단순화된 UI
"""

import streamlit as st
import os
import subprocess
import json

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
        page_title="PDF 구조화 시스템",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 PDF 구조화 & 임베딩 시스템")
    st.markdown("**표 제외, Overlap, LLM 메타데이터 생성**")
    
    # 이전 기록 삭제 버튼
    st.header("🗑️ 데이터 관리")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ 기존 데이터 삭제", key="delete_data_btn"):
            files_to_delete = [
                "text_only_chunks.json",
                "final_text_only_data.json",
                "full_pipeline_llm_enhanced_chunks.json",
                "structured_pipeline_chunks.json",
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
    st.header("📄 파일 업로드")
    uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type=['pdf'])
    
    if uploaded_file is not None:
        with open("input.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"✅ {uploaded_file.name} 업로드 완료!")
        
        # 파이프라인 실행
        if st.button("🚀 구조화 파이프라인 실행", type="primary"):
            with st.spinner("구조화 파이프라인 실행 중..."):
                # 1단계: 새로운 구조화된 파이프라인
                cmd1 = ["python", "test_full_pipeline_with_llm.py"]
                result1 = subprocess.run(cmd1, capture_output=True, text=True, encoding='utf-8')
                
                if result1.returncode == 0:
                    st.success("✅ 구조화 파이프라인 완료!")
                    
                    # 2단계: Streamlit 형식으로 변환
                    cmd2 = ["python", "src/streamlit_integration.py", "convert"]
                    result2 = subprocess.run(cmd2, capture_output=True, text=True, encoding='utf-8')
                    
                    if result2.returncode == 0:
                        st.success("✅ Streamlit 형식 변환 완료!")
                    else:
                        st.error("❌ Streamlit 형식 변환 실패!")
                else:
                    st.error("❌ 구조화 파이프라인 실패!")
                    if result1.stderr:
                        st.error(f"오류: {result1.stderr}")
    
    # 결과 확인
    st.header("📊 결과 확인")
    
    if os.path.exists("final_text_only_data.json"):
        data = load_json_file("final_text_only_data.json")
        if data:
            st.success(f"✅ 구조화된 데이터: {len(data)}개 청크")
            
            # 통계
            total_length = sum(item.get("content_length", 0) for item in data)
            avg_length = total_length / len(data) if data else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("청크 수", len(data))
            with col2:
                st.metric("총 텍스트 길이", f"{total_length:,}자")
            with col3:
                st.metric("평균 청크 길이", f"{avg_length:.1f}자")
            
            # 샘플 데이터
            if len(data) > 0:
                st.subheader("📋 샘플 데이터")
                sample = data[0]
                st.write(f"**섹션 제목**: {sample.get('section_title', '제목 없음')}")
                st.write(f"**내용**: {sample.get('content', '')[:200]}...")
                st.write(f"**키워드**: {', '.join(sample.get('keywords', []))}")
                st.write(f"**요약**: {sample.get('summary', '')}")
    else:
        st.info("구조화된 데이터가 없습니다. 먼저 파이프라인을 실행해주세요.")
    
    # 임베딩 생성
    st.header("🤖 임베딩 생성")
    
    if os.path.exists("final_text_only_data.json"):
        if st.button("🚀 임베딩 생성", type="primary"):
            with st.spinner("임베딩 생성 중..."):
                cmd = ["bash", "-c", "source environments/venv_rag_new/bin/activate && python src/embedding_generator.py final_text_only_data.json"]
                result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
                
                if result.returncode == 0:
                    st.success("✅ 임베딩 생성 완료!")
                    st.text(result.stdout)
                else:
                    st.error("❌ 임베딩 생성 실패!")
                    st.error(result.stderr)
    else:
        st.warning("먼저 구조화된 데이터를 생성해주세요.")
    
    # 품질 평가 (원래 코드와 동일)
    st.header("🤖 임베딩 품질 평가")
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
    
    # 파일 다운로드
    st.header("📁 파일 다운로드")
    
    files_to_download = [
        ("final_text_only_data.json", "구조화된 데이터"),
        ("full_pipeline_llm_enhanced_chunks.json", "원본 구조화 데이터"),
        ("embeddings.faiss", "FAISS 임베딩"),
        ("embeddings_metadata.pkl", "임베딩 메타데이터"),
        ("qa_pairs.json", "QA 쌍"),
        ("embedding_quality_report.json", "품질 평가 보고서"),
        ("embedding_quality_report_detailed.json", "상세 품질 평가 보고서")
    ]
    
    for filename, description in files_to_download:
        if os.path.exists(filename):
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
                    label=f"📥 {description} 다운로드",
                    data=data,
                    file_name=filename,
                    mime=mime_type
                )
            except Exception as e:
                st.error(f"파일 읽기 실패: {filename} - {e}")
        else:
            st.write(f"❌ {description}: 파일 없음")

if __name__ == "__main__":
    main() 