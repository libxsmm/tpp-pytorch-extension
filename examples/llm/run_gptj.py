from transformers import AutoModelForCausalLM, AutoTokenizer, GPTJConfig
import torch
import time
import json
import psutil
import pathlib
import argparse
import numpy as np
from itertools import chain
import transformers
import os
from accelerate import init_empty_weights

# import extend_profiler
try:
    import tpp_pytorch_extension as tpx
except:
    pass


# args
parser = argparse.ArgumentParser("GPT-J generation script", add_help=False)
parser.add_argument(
    "--device",
    type=str,
    choices=["cpu", "xpu", "cuda", "hpu"],
    help="cpu, xpu, hpu or cuda",
    default="cpu",
)
parser.add_argument(
    "--dtype",
    type=str,
    choices=["float32", "bfloat16", "float16"],
    help="bfloat16 or float32 or float16",
    default="bfloat16",
)
parser.add_argument(
    "--max-new-tokens", default=32, type=int, help="output max new tokens"
)
parser.add_argument("--greedy", action="store_true")
parser.add_argument("--ipex", action="store_true")
parser.add_argument("--use_tpp", action="store_true")
parser.add_argument("--jit", action="store_true")
parser.add_argument("--input-tokens", default="32", type=str)
parser.add_argument("--prompt", default=None, type=str)
parser.add_argument("--num-iter", default=100, type=int, help="num iter")
parser.add_argument("--num-warmup", default=10, type=int, help="num warmup")
parser.add_argument("--batch-size", default=1, type=int, help="batch size")
parser.add_argument("--print-memory", action="store_true")
parser.add_argument("--profile", action="store_true")
parser.add_argument("--dist-backend", default="mpi", type=str)
parser.add_argument("--load-sharded-model", action="store_true")
parser.add_argument("--save-sharded-model", action="store_true")
args = parser.parse_args()
print(args)
my_rank = 0
my_size = 1

def dist_init():
    import os
    global my_rank
    global my_size
    if int(os.environ.get("PMI_SIZE", "0")) > 1:
        if args.dist_backend == "ccl":
            try:
                import oneccl_bindings_for_pytorch
            except:
                print("CCL backend requested but import oneccl_bindings_for_pytorch failed")
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
        else:
            raise ValueError(f"{args.dist_backend} backend requested but not supported")

        os.environ["RANK"] = os.environ.get("PMI_RANK", "0")
        os.environ["WORLD_SIZE"] = os.environ.get("PMI_SIZE", "1")
        torch.distributed.init_process_group(backend=args.dist_backend)
        my_rank = torch.distributed.get_rank()
        my_size = torch.distributed.get_world_size()
        print(f"My rank: {my_rank} size: {my_size}")


def get_memory_usage(name, args):
    if args.print_memory:
        if args.device == "cuda":
            memory_allocated = round(torch.cuda.memory_reserved() / 1024**3, 3)
        elif args.device == "hpu":
            memory_allocated = round(ht.hpu.memory_allocated(0) / 1024**3, 3)
        elif args.device == "xpu":
            memory_allocated = round(torch.xpu.memory_reserved() / 1024**3, 3)
        else:
            memory_allocated = round(psutil.Process().memory_info().rss / 1024**3, 3)

        print(name, "memory used total:", memory_allocated, "GB")
    else:
        return


# device
device = torch.device(args.device)
# import extension
if args.ipex:
    import intel_extension_for_pytorch as ipex

    try:
        ipex._C.disable_jit_linear_repack()
    except:
        pass
if args.device == "hpu":
    import habana_frameworks.torch as ht
    import habana_frameworks.torch.core as htcore

# generate args
if args.greedy:
    generate_kwargs = dict(do_sample=False, temperature=0.9)
else:
    generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=4)
if args.jit:
    torch._C._jit_set_texpr_fuser_enabled(False)
    # generate_kwargs["jit"] = True
    raise NotImplementedError("Torch JIT Not supported yet!")

# dtype
if args.dtype == "bfloat16":
    amp_enabled = True
    amp_dtype = torch.bfloat16
elif args.dtype == "float16":
    amp_enabled = True
    amp_dtype = torch.float16
else:
    amp_enabled = False
    amp_dtype = torch.float32

# load model
model_id = "EleutherAI/gpt-j-6B"
if args.use_tpp and args.load_sharded_model:
    config = GPTJConfig.from_pretrained(model_id)
    config.return_dict = return_dict=not args.jit
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            config, torch_dtype=amp_dtype,
        )
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_id, low_cpu_mem_usage=True, return_dict=not args.jit, torch_dtype=amp_dtype
    )
tokenizer = AutoTokenizer.from_pretrained(model_id)
get_memory_usage("Host", args)
if not args.load_sharded_model:
    model = model.eval().to(device)
    get_memory_usage("Device", args)
model = model.to(memory_format=torch.channels_last)

# to hpu graph
if args.device == "hpu":
    model = ht.hpu.wrap_in_hpu_graph(model)
    get_memory_usage("Graph", args)
# to ipex
if args.ipex:
    model = ipex.optimize(model.eval(), dtype=amp_dtype, inplace=True)
    get_memory_usage("Ipex", args)

if args.use_tpp:
    dist_init()
    from tpp_pytorch_extension.llm.fused_gptj_infer import OptimizeModelForGPTJ
    OptimizeModelForGPTJ(model, dtype=amp_dtype, device=device)

if my_size > 1:
    sharded_model_path = f"./sharded_model/r{my_rank}_{my_size}"
    model_file = f"{sharded_model_path}/model.pt"
    if args.load_sharded_model == True:
        model.load_state_dict(torch.load(model_file))
    elif args.save_sharded_model == True:
        os.makedirs(sharded_model_path, exist_ok=True)
        torch.save(model.state_dict(), model_file)
model = model.eval().to(device)

for n, p in model.named_parameters():
    print(f"{n}: {list(p.shape)}   {p.dtype} {type(p)}")

# input prompt
current_path = pathlib.Path(__file__).parent.resolve()
with open(str(current_path) + "/prompt.json") as f:
    prompt_pool = json.load(f)
if args.prompt is not None:
    prompt = args.prompt
elif args.input_tokens in prompt_pool:
    prompt = prompt_pool[args.input_tokens]
else:
    raise SystemExit("[ERROR] Plese use --prompt if want to use custom input.")

input_size = tokenizer(prompt, return_tensors="pt").input_ids.size(dim=1)
print("---- Prompt size:", input_size)
if input_size + args.max_new_tokens > 2049:
    raise SystemExit(
        "[WARN] Token indices sequence length is longer than the specified maximum "
        + "sequence length for this model (2049 > 2048). Running this sequence through the model will result in indexing errors"
    )

# start
total_time = 0.0
num_iter = args.num_iter
num_warmup = args.num_warmup
prompt = [prompt] * args.batch_size
total_list = []
record_shapes = True


def trace_handler(prof):
    print(
        prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1),
        flush=True,
    )
    prof.export_chrome_trace("my_trace.log" + str(prof.step_num) + ".json")
    prof.profiler.print_op_timings(prof.profiler, prefix="gptj_time_" + str(my_rank))


with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU],
    schedule=torch.profiler.schedule(wait=0, warmup=9, active=1),
    record_shapes=record_shapes,
    on_trace_ready=trace_handler,
) as prof:
    with torch.inference_mode(), torch.autocast(
        device_type=args.device,
        enabled=amp_enabled,
        dtype=amp_dtype if amp_enabled else None,
    ):
        for i in range(num_iter):
            if args.use_tpp:
                tpx.reset_debug_timers()
            get_memory_usage("Iteration: " + str(i), args)
            tic = time.time()
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            # gen_tokens, latency_list = model.generate(input_ids, max_new_tokens=args.max_new_tokens, **generate_kwargs)
            gen_tokens = model.generate(
                input_ids, max_new_tokens=args.max_new_tokens, **generate_kwargs
            )
            gen_text = tokenizer.batch_decode(gen_tokens)[0]
            if args.device == "xpu":
                torch.xpu.synchronize()
            elif args.device == "cuda":
                torch.cuda.synchronize()
            elif args.device == "hpu":
                gen_tokens.to("cpu")
            toc = time.time()
            if args.profile:
                prof.step()
            if args.use_tpp:
                tpx.print_debug_timers()
            print(gen_text, flush=True)
            if i >= num_warmup:
                total_time += toc - tic
                # total_list.append(latency_list)

latency = total_time / (num_iter - num_warmup)
# first_latency = np.mean([x[0] for x in total_list])
# average_2n = list(chain(*[x[1:] for x in total_list]))
# average_2n.sort()
# average_2n_latency = np.mean(average_2n)
# p90_latency = average_2n[int(len(average_2n)*0.9)]
print("\n", "-" * 10, "Summary:", "-" * 10)
print("Inference latency: %.3f sec." % latency)
# print("First token average latency: %.3f sec." % first_latency)
# print("Average 2... latency: %.3f sec." % average_2n_latency)
# print("P90 2... latency: %.3f sec." % p90_latency)
