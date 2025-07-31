#!/bin/bash
echo "======================================"
echo "Linux용 가상환경만 생성 (기존 환경 유지)"
echo "======================================"

echo ""
echo "[1/3] Creating venv_text_linux..."
python3 -m venv environments/venv_text_linux
source environments/venv_text_linux/bin/activate
pip install -r requirements/requirements_text_new.txt
deactivate
echo "venv_text_linux setup complete!"

echo ""
echo "[2/3] Creating venv_table_linux..."
python3 -m venv environments/venv_table_linux
source environments/venv_table_linux/bin/activate
pip install -r requirements/requirements_table_new.txt
deactivate
echo "venv_table_linux setup complete!"

echo ""
echo "[3/3] Creating venv_rag_linux..."
python3 -m venv environments/venv_rag_linux
source environments/venv_rag_linux/bin/activate
pip install -r requirements/requirements_rag_new.txt
deactivate
echo "venv_rag_linux setup complete!"

echo ""
echo "======================================"
echo "🎉 Linux용 가상환경 생성 완료!"
echo "======================================"
echo ""
echo "기존 Windows 환경은 그대로 유지됩니다."
echo "Linux에서 실행할 때는 _linux 접미사가 붙은 환경을 사용하세요."
echo "" 