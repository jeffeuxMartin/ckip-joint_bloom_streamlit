# ------------------- LIBRARIES -------------------- #
import os, logging, torch, streamlit as st
from transformers import (
    AutoTokenizer, AutoModelForCausalLM)

# --------------------- HELPER --------------------- #
def C(text, color="yellow"):
    color_dict: dict = dict(
            red="\033[01;31m",
          green="\033[01;32m",
         yellow="\033[01;33m",
           blue="\033[01;34m",
        magenta="\033[01;35m",
           cyan="\033[01;36m",
    )
    color_dict[None] = "\033[0m"
    return (
        f"{color_dict.get(color, None)}"
        f"{text}{color_dict[None]}")

# ------------------ ENVIORNMENT ------------------- #
os.environ["HF_ENDPOINT"] = "https://huggingface.co"
device = ("cuda" 
    if torch.cuda.is_available() else "cpu")
logging.info(C("[INFO] "f"device = {device}"))

# ------------------ INITITALIZE ------------------- #
@st.cache_resource
def model_init():
    tokenizer = AutoTokenizer.from_pretrained(
        "ckip-joint/bloom-1b1-zh")
    model = AutoModelForCausalLM.from_pretrained(
        "ckip-joint/bloom-1b1-zh",
        # Ref.: Eric, Thanks!
        # torch_dtype="auto", 
        # device_map="auto",
    # Ref. for `half`: Chan-Jan, Thanks!
    ).eval().to(device)
    st.balloons()
    logging.info(C("[INFO] "f"Model init success!"))
    return tokenizer, model

tokenizer, model = model_init()

# ===================== INPUT ====================== #
# prompt = "\u554F\uFF1A\u53F0\u7063\u6700\u9AD8\u7684\u5EFA\u7BC9\u7269\u662F\uFF1F\u7B54\uFF1A"  #@param {type:"string"}
prompt = st.text_input("Prompt: ")

# =================== INFERENCE ==================== #
if prompt:
    with torch.no_grad():
        [texts_out] = model.generate(
            **tokenizer(
                prompt, return_tensors="pt"
            ).to(device))
    output_text = tokenizer.decode(texts_out)
    st.markdown(output_text)
    