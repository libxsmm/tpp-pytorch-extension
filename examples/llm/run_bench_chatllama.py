from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import os
import numpy as np
import datasets

dataset_chat_prompts = datasets.load_dataset("miracl/hagrid", split="train")

orig_checkpoint = "meta-llama/Llama-2-7b-chat-hf"
draft_checkpoint = "meta-llama/Llama-2-7b-chat-hf"
draft_checkpoint_helper = "Felladrin/Llama-68M-Chat-v1"
use_same_tokenizers = 1

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
draft_tokenizer = AutoTokenizer.from_pretrained(draft_checkpoint)
draft_model.generation_config.pad_token_id = draft_tokenizer.eos_token_id

draft_model_helper = AutoModelForCausalLM.from_pretrained(
    draft_checkpoint_helper,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
draft_tokenizer_helper = AutoTokenizer.from_pretrained(draft_checkpoint_helper)
draft_model_helper.generation_config.pad_token_id = draft_tokenizer_helper.eos_token_id

from tpp_pytorch_extension.llm.fused_llama_infer import OptimizeModelForLlama

dtype = torch.bfloat16
OptimizeModelForLlama(orig_model, dtype=dtype)
OptimizeModelForLlama(draft_model, dtype=dtype, weight_dtype="mxfp4")
OptimizeModelForLlama(draft_model_helper, dtype=dtype)

generation_config = {
    "do_sample": False,
    "temperature": None,
    "top_p": None,
    "top_k": None, 
    "max_new_tokens": 128,
    "num_assistant_tokens": 8   
}

for i in range(0, 100):
    prompt = dataset_chat_prompts[i]['query']
    orig_inputs = orig_tokenizer(prompt, return_tensors="pt").to(orig_model.device)

    start = time.time()
    if use_same_tokenizers :
        spec_outputs = orig_model.generate(**orig_inputs, assistant_model=draft_model, assistant_model_helper=draft_model_helper, tokenizer=orig_tokenizer,  **generation_config) 
    else :
        spec_outputs = orig_model.generate(**orig_inputs, assistant_model=draft_model, assistant_model_helper=draft_model_helper, tokenizer=orig_tokenizer, assistant_tokenizer_helper=draft_tokenizer_helper, **generation_config)
    end = time.time()

    start = time.time()
    if use_same_tokenizers :
        spec_outputs = orig_model.generate(**orig_inputs, assistant_model=draft_model, assistant_model_helper=draft_model_helper, tokenizer=orig_tokenizer,  **generation_config) 
    else :
        spec_outputs = orig_model.generate(**orig_inputs, assistant_model=draft_model, assistant_model_helper=draft_model_helper, tokenizer=orig_tokenizer, assistant_tokenizer_helper=draft_tokenizer_helper, **generation_config)
    end = time.time()
    print(f"[{i}] Speculative MULTI time: {end-start} s")

    start = time.time()
    if use_same_tokenizers :
        spec_outputs = orig_model.generate( **orig_inputs, assistant_model=draft_model_helper, tokenizer=orig_tokenizer,  **generation_config)
    else :
        spec_outputs = orig_model.generate( **orig_inputs, assistant_model=draft_model_helper, tokenizer=orig_tokenizer, assistant_tokenizer=draft_tokenizer_helper,  **generation_config)
    end = time.time()

    start = time.time()
    if use_same_tokenizers :
        spec_outputs = orig_model.generate( **orig_inputs, assistant_model=draft_model_helper, tokenizer=orig_tokenizer,  **generation_config)
    else :
        spec_outputs = orig_model.generate( **orig_inputs, assistant_model=draft_model_helper, tokenizer=orig_tokenizer, assistant_tokenizer=draft_tokenizer_helper,  **generation_config)
    end = time.time()
    print(f"[{i}] Speculative DRAFT time: {end-start} s")

    start = time.time()
    spec_outputs = orig_model.generate(**orig_inputs, assistant_model=draft_model, tokenizer=orig_tokenizer,  **generation_config)
    end = time.time()

    start = time.time()
    spec_outputs = orig_model.generate(**orig_inputs, assistant_model=draft_model, tokenizer=orig_tokenizer,  **generation_config)
    end = time.time()
    print(f"[{i}] Speculative MXFP4 time: {end-start} s")

    start = time.time()
    orig_outputs = orig_model.generate(**orig_inputs, tokenizer=orig_tokenizer,  **generation_config)
    end = time.time()
    print(f"[{i}] Original GREEDY time: {end-start} s")

