cp ./pyproject_transformers4.57.txt ./pyproject.toml
pip install uv
uv sync --python 3.10
souce .venv/bin/activate
uv pip install -e . --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://pypi.nvidia.com/
