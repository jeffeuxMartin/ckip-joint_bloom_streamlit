# ------------------- LIBRARIES -------------------- #
import os, torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM)

# ------------------ ENVIORNMENT ------------------- #
os.environ["HF_ENDPOINT"] = "https://huggingface.co"
device = ("cuda" 
    if torch.cuda.is_available() else "cpu")

# ------------------ INITITALIZE ------------------- #
tokenizer = AutoTokenizer.from_pretrained(
    "ckip-joint/bloom-1b1-zh")
model = AutoModelForCausalLM.from_pretrained(
    "ckip-joint/bloom-1b1-zh",
    # Ref.: Eric, Thanks!
    torch_dtype="auto", device_map="auto",
# Ref. for `half`: Chan-Jan, Thanks!
).eval().half().to(device)

# ===================== INPUT ====================== #
prompt = "\u554F\uFF1A\u53F0\u7063\u6700\u9AD8\u7684\u5EFA\u7BC9\u7269\u662F\uFF1F\u7B54\uFF1A"  #@param {type:"string"}

# =================== INFERENCE ==================== #
with torch.no_grad():
    [texts_out] = model.generate(
        **tokenizer(
            prompt, return_tensors="pt"
        ).to(device))
print(tokenizer.decode(texts_out))
