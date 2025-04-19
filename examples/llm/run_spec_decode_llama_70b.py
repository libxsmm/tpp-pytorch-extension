from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

# import pandas as pd
import numpy as np

# pd.set_option('display.max_colwidth', 500)

orig_checkpoint = "meta-llama/Llama-2-70b-hf"
draft_checkpoint = "meta-llama/Llama-2-7b-hf"

orig_model = AutoModelForCausalLM.from_pretrained(
    orig_checkpoint,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
orig_tokenizer = AutoTokenizer.from_pretrained(orig_checkpoint)

orig_model.generation_config.pad_token_id = orig_tokenizer.eos_token_id

draft_model = AutoModelForCausalLM.from_pretrained(
    draft_checkpoint,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
draft_tokenizer = AutoTokenizer.from_pretrained(draft_checkpoint, use_fast=False)

draft_model.generation_config.pad_token_id = draft_tokenizer.eos_token_id

from tpp_pytorch_extension.llm.fused_llama_infer import OptimizeModelForLlama

dtype = torch.bfloat16
OptimizeModelForLlama(orig_model, dtype=dtype)
OptimizeModelForLlama(draft_model, dtype=dtype)  # , weight_dtype=torch.bfloat8)

generation_config = {
    "do_sample": False,
    "temperature": None,
    "top_p": None,
    "max_new_tokens": 500,
}

prompt = "Story of David and Goliath: Once upon a time"

orig_inputs = orig_tokenizer(prompt, return_tensors="pt").to(orig_model.device)

# Warmup
# _ = orig_model.generate(**orig_inputs, assistant_model=draft_model, **generation_config)

start = time.time()
spec_outputs = orig_model.generate(
    **orig_inputs, assistant_model=draft_model, **generation_config
)
end = time.time()

spec_text = orig_tokenizer.decode(spec_outputs[0], skip_special_tokens=True)
print(f"Speculative Decoding time: {end-start} s")
print(f"Speculative Decoding output:\n{spec_text}")

# Warmup
# _ = orig_model.generate(**orig_inputs, **generation_config)

start = time.time()
orig_outputs = orig_model.generate(**orig_inputs, **generation_config)
end = time.time()

orig_text = orig_tokenizer.decode(orig_outputs[0], skip_special_tokens=True)
print(f"Original Model time: {end-start} s")
print(f"Original Model output:\n{orig_text}")
