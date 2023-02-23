#!sh
conda create -p bloom_demo python=3.8
source activate ./bloom_demo
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu111
# streamlit run main.py
