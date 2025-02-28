import torch
import time
import json
import pathlib
import argparse
import os
import inspect
from accelerate import init_empty_weights
from typing import Tuple

from transformers import (
    # pipeline,
    AutoModelForCausalLM,
    # AutoModel,
    GenerationMixin,
    LlamaForCausalLM,
    T5ForConditionalGeneration,
    AutoTokenizer,
    LlamaTokenizer,
    AutoConfig,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    import tpp_pytorch_extension as tpx
    from tpp_pytorch_extension.llm.llm_common import (
        jit_trace_model,
        optimize_for_first_token,
    )
except:
    pass

# supported models now
MODEL_CLASSES = {
    "gpt": (AutoModelForCausalLM, AutoTokenizer),
    "opt": (AutoModelForCausalLM, AutoTokenizer),
    "bloom": (AutoModelForCausalLM, AutoTokenizer),
    # "llama": (LlamaForCausalLM, LlamaTokenizer),
    "llama": (LlamaForCausalLM, AutoTokenizer),
    "t5": (T5ForConditionalGeneration, AutoTokenizer),
    "auto": (AutoModelForCausalLM, AutoTokenizer),
    # "chatglm": (AutoModel, AutoTokenizer),
}

# args
parser = argparse.ArgumentParser("Generation script", add_help=False)
parser.add_argument(
    "-m",
    "--model-id",
    type=str,
    default="EleutherAI/gpt-j-6B",
    help="the huggingface mdoel id",
)
parser.add_argument(
    "--device",
    type=str,
    choices=["cpu", "xpu", "cuda", "hpu"],
    default="cpu",
    help="cpu, xpu, hpu or cuda",
)
parser.add_argument(
    "--dtype",
    type=str,
    choices=["float32", "bfloat16", "float16"],
    default="bfloat16",
    help="bfloat16, float32 or float16",
)
parser.add_argument(
    "--weight-dtype",
    type=str,
    choices=["float32", "bfloat16", "bfloat8", "hfloat8", "mxfp4", "qint8", None],
    default=None,
    help="bfloat16, float32 or bfloat8 or hfloat8 or mxfp4 or qint8",
)
parser.add_argument(
    "--input-tokens",
    default="32",
    type=str,
    help="input tokens length if needed from prompt.json",
)
parser.add_argument(
    "--max-new-tokens", default=32, type=int, help="output max new tokens"
)
parser.add_argument(
    "--prompt", default=None, type=str, help="input prompt for self-defined if needed"
)
parser.add_argument("--greedy", action="store_true")
parser.add_argument("--ipex", action="store_true")
parser.add_argument("--use-tpp", action="store_true")
parser.add_argument("--tpp-linear-only", action="store_true")
parser.add_argument("--tpp-no-opt", action="store_true")
parser.add_argument("--jit", action="store_true")
parser.add_argument("--num-iter", default=10, type=int, help="num iter")
parser.add_argument("--num-warmup", default=3, type=int, help="num warmup")
parser.add_argument("--batch-size", default=1, type=int, help="batch size")
parser.add_argument("--token-latency", action="store_true", help="get token latency")
parser.add_argument("--profile", action="store_true")
parser.add_argument("--dist-backend", default="mpi", type=str)
parser.add_argument("--load-sharded-model", action="store_true")
parser.add_argument("--save-sharded-model", action="store_true")
args = parser.parse_args()
print(args)

my_rank = 0
my_size = 1

if os.environ.get("LIBXSMM_X86_AMX_GEMM_ENFORCE_Mx1_TILE_BLOCKING", None) is None:
    os.environ["LIBXSMM_X86_AMX_GEMM_ENFORCE_Mx1_TILE_BLOCKING"] = "1"
if os.environ.get("LIBXSMM_X86_AMX_GEMM_STREAMING_TILELOAD", None) is None:
    os.environ["LIBXSMM_X86_AMX_GEMM_STREAMING_TILELOAD"] = "1"


def dist_init():
    import os

    global my_rank
    global my_size
    if (
        int(os.environ.get("PMI_SIZE", "0")) > 1
        and int(os.environ.get("MULTI_INSTANCE", "0")) == 0
    ):
        if args.dist_backend == "ccl":
            try:
                import oneccl_bindings_for_pytorch
            except:
                print(
                    "CCL backend requested but import oneccl_bindings_for_pytorch failed"
                )
                raise
        elif args.dist_backend == "mpi":
            if not torch.distributed.is_mpi_available():
                try:
                    import torch_mpi
                except:
                    print(
                        "MPI backend requested but not available try installing torch_mpi module"
                    )
                    raise
        elif args.dist_backend == "gloo":
            if not torch.distributed.is_gloo_available():
                raise ValueError(
                    f"{args.dist_backend} backend requested but not supported"
                )
        else:
            raise ValueError(f"{args.dist_backend} backend requested but not supported")

        os.environ["RANK"] = os.environ.get("PMI_RANK", "0")
        os.environ["WORLD_SIZE"] = os.environ.get("PMI_SIZE", "1")
        torch.distributed.init_process_group(backend=args.dist_backend)
        my_rank = torch.distributed.get_rank()
        my_size = torch.distributed.get_world_size()
        print(f"My rank: {my_rank} size: {my_size}")


orig_print = print


def print_rank0(*args, **kwargs):
    if my_rank == 0:
        orig_print(*args, **kwargs)


print = print_rank0

# device
device = torch.device(args.device)

# import extension
if args.ipex:
    import intel_extension_for_pytorch as ipex

    try:
        ipex._C.disable_jit_linear_repack()
    except Exception:
        pass
if args.device == "hpu":
    import habana_frameworks.torch as ht

# dtype
tpp_dtype = getattr(torch, args.dtype)
# AMP doesn't work for Half on CPU so adding workaround
if args.use_tpp and tpp_dtype == torch.half:
    args.dtype = "float32"

amp_enabled = True if args.dtype != "float32" else False
amp_dtype = getattr(torch, args.dtype)

# load model
model_type = next(
    (x for x in MODEL_CLASSES.keys() if x in args.model_id.lower()), "auto"
)
print("model_type", model_type)
model_class = MODEL_CLASSES[model_type]
if args.use_tpp and args.load_sharded_model:
    config = AutoConfig.from_pretrained(args.model_id)
    config.return_dict = return_dict = not args.jit
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, torch_dtype=amp_dtype)
else:
    model = model_class[0].from_pretrained(
        args.model_id,
        low_cpu_mem_usage=True,
        return_dict=not args.jit,
        torch_dtype=amp_dtype,
    )
tokenizer = model_class[1].from_pretrained(args.model_id)
if not args.load_sharded_model:
    model = model.eval().to(device)
model = model.to(memory_format=torch.channels_last)

# to hpu graph
if args.device == "hpu":
    model = ht.hpu.wrap_in_hpu_graph(model)
# to ipex
if args.ipex:
    model = ipex.optimize(model.eval(), dtype=amp_dtype, inplace=True)

if args.use_tpp:
    dist_init()
    # weight_dtype = getattr(torch, args.weight_dtype) if args.weight_dtype else None
    weight_dtype = args.weight_dtype
    if args.tpp_no_opt:
        # use tpp only to print first and 2nd token latencies
        pass
    elif args.tpp_linear_only:
        from tpp_pytorch_extension.nn import OptimizeForLinear

        OptimizeForLinear(model)

    elif model.config.architectures[0] == "GPTJForCausalLM":
        from tpp_pytorch_extension.llm.fused_gptj_infer import OptimizeModelForGPTJ

        OptimizeModelForGPTJ(
            model, dtype=tpp_dtype, device=device, weight_dtype=weight_dtype
        )
    elif model.config.architectures[0] == "OPTForCausalLM":
        from tpp_pytorch_extension.llm.fused_opt_infer import OptimizeModelForOPT

        OptimizeModelForOPT(
            model, dtype=tpp_dtype, device=device, weight_dtype=weight_dtype
        )
    elif model.config.architectures[0] == "LLaMAForCausalLM":
        from tpp_pytorch_extension.llm.fused_llama_infer import OptimizeModelForLlama

        OptimizeModelForLlama(
            model, dtype=tpp_dtype, device=device, weight_dtype=weight_dtype
        )
    elif model.config.architectures[0] == "LlamaForCausalLM":
        from tpp_pytorch_extension.llm.fused_llama_infer import OptimizeModelForLlama

        OptimizeModelForLlama(
            model, dtype=tpp_dtype, device=device, weight_dtype=weight_dtype
        )
    elif model.config.architectures[0] == "Qwen2ForCausalLM":
        from tpp_pytorch_extension.llm.fused_qwen2_infer import OptimizeModelForQwen2

        OptimizeModelForQwen2(
            model, dtype=tpp_dtype, device=device, weight_dtype=weight_dtype
        )
    else:
        print(type(model.config.architectures))
        print(model.config.architectures)
        raise NotImplementedError("Model type not supported by TPP")

if my_size > 1:
    sharded_model_path = f"./sharded_model/r{my_rank}_{my_size}"
    model_file = f"{sharded_model_path}/model.pt"
    if args.load_sharded_model == True:
        model.load_state_dict(torch.load(model_file))
    elif args.save_sharded_model == True:
        os.makedirs(sharded_model_path, exist_ok=True)
        torch.save(model.state_dict(), model_file)
model = model.eval().to(device)

# for n, p in model.named_parameters():
#    print(f"{n}: {list(p.shape)}   {p.dtype} {type(p)}")

# input prompt
current_path = pathlib.Path(__file__).parent.resolve()
with open(str(current_path) + "/prompt.txt") as f:
    prompt = f.read()
    tokens_to_use = int(args.input_tokens)
    input_ids = (
        tokenizer(prompt, return_tensors="pt").input_ids[:, :tokens_to_use].contiguous()
    )
    prompt = tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]

"""
tokens_to_use = 0
with open(str(current_path) + "/prompt.json") as f:
    prompt_pool = json.load(f)
if args.prompt is not None:
    prompt = args.prompt
elif model_type == "auto":
    raise SystemExit(
        "[ERROR] model prompt is not supported, please use --prompt for this model: "
        + args.model_id
    )
elif args.input_tokens in prompt_pool[model_type]:
    prompt = prompt_pool[model_type][args.input_tokens]
else:
    tokens_to_use = int(args.input_tokens)
    prompt = None
    for key in prompt_pool[model_type].keys():
        if int(key) >= tokens_to_use:
            prompt = prompt_pool[model_type][key]
            break
    if not prompt:
        raise SystemExit("[ERROR] Plese use --prompt if want to use custom input.")

if tokens_to_use > 0:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids[:,:tokens_to_use].contiguous()
    prompt = tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]
'''
for k in prompt_pool[model_type].keys():
    v = prompt_pool[model_type][k]
    sz = tokenizer(v, return_tensors="pt").input_ids.size(dim=1)
    print(f"Prompt: {k}  sz = {sz}")

exit()
'''
"""

input_size = tokenizer(prompt, return_tensors="pt").input_ids.size(dim=1)
print("---- Prompt size:", input_size)
print("---- Prompt text:", prompt)

# generate args
generate_kwargs = dict(
    do_sample=False, temperature=0.9, num_beams=1 if args.greedy else 4
)
if args.use_tpp:
    cpp_profile = True
    if args.jit:
        model = jit_trace_model(
            model,
            tokenizer,
            generate_kwargs["num_beams"],
            enable_profile=cpp_profile,
            only_last_logit=True,
        )
    else:
        model = optimize_for_first_token(
            model,
            generate_kwargs["num_beams"],
            enable_profile=cpp_profile,
            only_last_logit=True,
            default_kv=(args.tpp_no_opt or args.tpp_linear_only),
        )

    # generate_kwargs["jit"] = True
if args.token_latency:
    if args.use_tpp:
        generate_kwargs["token_latency"] = True
    else:
        print("Warning: --token-latnecy is ignored when not using TPP (--use-tpp)")
        args.token_latency = False
# if args.use_tpp:
#    generate_kwargs["TP_number"] = my_size

# start
total_time = 0.0
num_iter = args.num_iter
num_warmup = args.num_warmup
prompt = [prompt] * args.batch_size
# prompt = [
#         "It is done, and submitted. You can play 'Survival of the Tastiest' on Android, and on the web. Playing on the web works fine if you like",
#         "Once upon a time, there existed a little girl, who liked to have adventures. She wanted to go to places and meet new people, and have fun.",
#         ]
tokenizer.pad_token = tokenizer.eos_token
if tokenizer.pad_token == "":
    tokenizer.pad_token = "</s>"
tokenizer.padding_side = "left"
total_list = []
record_shapes = True
output_past_key_values = False
if args.use_tpp and output_past_key_values == True:
    generate_kwargs["output_past_key_values"] = True
else:
    output_past_key_values = False


def trace_handler(prof):
    print(
        prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1),
        flush=True,
    )
    # prof.export_chrome_trace("my_trace.log" + str(prof.step_num) + ".json")
    try:
        prof.profiler.print_op_timings(prof.profiler, prefix="llm_time_" + str(my_rank))
    except:
        pass


with torch.inference_mode(), torch.no_grad(), torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU],
    schedule=torch.profiler.schedule(wait=0, warmup=9, active=1),
    record_shapes=record_shapes,
    on_trace_ready=trace_handler,
) as prof, torch.autocast(
    device_type=args.device,
    enabled=amp_enabled,
    dtype=amp_dtype if amp_enabled else None,
):
    for i in range(num_iter):
        tic = time.time()
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
        # inputs = tokenizer(prompt, return_tensors="pt", padding=False).to(device)
        # input_ids = inputs.input_ids.to(device)
        # print(type(inputs))

        output = model.generate(
            **inputs, max_new_tokens=args.max_new_tokens, **generate_kwargs
        )
        if output_past_key_values == True:
            output, pkv = output
        gen_ids = output[0] if args.token_latency else output
        gen_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        if args.device == "xpu":
            torch.xpu.synchronize()
        elif args.device == "cuda":
            torch.cuda.synchronize()
        elif args.device == "hpu":
            gen_ids.to("cpu")
        toc = time.time()
        if args.profile:
            prof.step()
        print(gen_text, len(gen_ids), flush=True)
        if i < num_warmup or not args.token_latency:
            print("Iteration: %d, Time: %.6f sec" % (i, toc - tic), flush=True)
        if i >= num_warmup:
            total_time += toc - tic
            if args.token_latency:
                total_list.append(output[1])
                first = output[1][0]
                rest = output[1][1:]
                sum_rest = sum(rest)
                print(
                    "Iteration: %d, Time: %.6f sec  first: %.3f s  sum next: %.3f s  avg next: %.4f s"
                    % (i, toc - tic, first, sum_rest, sum_rest / len(rest)),
                    flush=True,
                )


print("\n", "-" * 10, "Summary:", "-" * 10)
latency = total_time / (num_iter - num_warmup)
print("Inference latency: %.3f sec." % latency)
if args.token_latency:
    import numpy as np
    from itertools import chain

    first_latency = np.mean([x[0] for x in total_list])
    average_2n = list(chain(*[x[1:] for x in total_list]))
    average_2n.sort()
    average_2n_latency = np.mean(average_2n)
    p90_latency = average_2n[int(len(average_2n) * 0.9)]
    p99_latency = average_2n[int(len(average_2n) * 0.99)]
    print("First token average latency: %.3f sec." % first_latency)
    print("Average 2... latency: %.4f sec." % average_2n_latency)
    print("P90 2... latency: %.4f sec." % p90_latency)
    print("P99 2... latency: %.4f sec." % p99_latency)
