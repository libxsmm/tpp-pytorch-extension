import torch
from torch import nn

import argparse
import time

# import tpp_pytorch_extension as tpx
# from tpp_pytorch_extension import nn as tpp_replace

# from sglang.srt.models.qwen2 import Qwen2MLP as Qwen3MLP
# from sglang.srt.distributed import parallel_state
# from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

from torch import distributed as dist
import os


def initialize_sglang_environment(tp_size=1):
    # A. Set Global Server Args (Fixes the new ValueError)
    args = ServerArgs(model_path="dummy", tp_size=tp_size)
    set_global_server_args_for_scheduler(args)

    # B. Set Environment Variables (Fixes the RANK/WORLD_SIZE error)
    os.environ.update({
        'MASTER_ADDR': 'localhost',
        'MASTER_PORT': '12355',
        'RANK': '0',
        'WORLD_SIZE': str(tp_size)
    })

    # C. Initialize World & TP Groups (Fixes AssertionErrors)
    parallel_state.init_distributed_environment(world_size=tp_size, rank=0, backend="nccl" if torch.cuda.is_available() else "gloo")
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=tp_size)

class SGLANG_FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act="silu",
            quant_config=None,
            # prefix=add_prefix("mlp", prefix),
        )
    def forward(self, x):
        down_proj = self.mlp(x)
        return down_proj


class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        # self.act_fn =  torch.nn.functional.silu()

    def forward(self, x):
        down_proj = self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

# create config
class Config:
    def __init__(self, hidden_size, intermediate_size):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

# create a model of multiple FFN layers

class FFNModel(nn.Module):
    def __init__(self, config, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([FFN(config) for _ in range(num_layers)])
        # self.layers = nn.ModuleList([SGLANG_FFN(config) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
# Assume the input is of shape (batch_size, seq_len, hidden_size)
def benchmark_ffn_model(model, input_tensor, num_iter=1000):
    
    # Warmup
    with torch.inference_mode(), torch.amp.autocast(device_type='cpu', enabled=(input_tensor.dtype == torch.bfloat16), dtype=input_tensor.dtype):
        for _ in range(3):
            model(input_tensor)
    
    # Time inference
    start = time.perf_counter()
    # tpx.reset_debug_timers()
    with torch.inference_mode(), torch.amp.autocast(device_type='cpu', enabled=(input_tensor.dtype == torch.bfloat16), dtype=input_tensor.dtype):
        for _ in range(num_iter):
            output = model(input_tensor)
    # tpx.print_debug_timers(detailed=False)
    elapsed = time.perf_counter() - start
    print(f"Total inference time for {num_iter} iterations: {elapsed:.2f} seconds")
    avg_time = (elapsed / num_iter) * 1000
    print(f"Inference time: {avg_time:.2f} ms (average over {num_iter} iterations)")
    return output, avg_time

# Run the benchmark based upon user input for batch size, sequence length, hidden size, intermediate size and number of layers
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Benchmark FFN Model")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for benchmarking")
    parser.add_argument("--seq_len", type=int, default=1, help="Sequence length for benchmarking")
    parser.add_argument("--hidden_size", type=int, default=512, help="Hidden size for FFN model")
    parser.add_argument("--intermediate_size", type=int, default=512, help="Intermediate size for FFN model")
    parser.add_argument("--num_layers", type=int, default=100, help="Number of FFN layers in the model")
    parser.add_argument("--num_iter", type=int, default=1000, help="Number of iterations for benchmarking")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"], help="Data type for model and input")
    parser.add_argument("--use_torchcompile", type=int, default=0, choices=[0, 1], help="Whether to use torch.compile for the model")
    args = parser.parse_args()
    
    # print the arguments
    print(f"Arguments: batch_size={args.batch_size}, seq_len={args.seq_len}, hidden_size={args.hidden_size}, intermediate_size={args.intermediate_size}, num_layers={args.num_layers}, num_iter={args.num_iter}, dtype={args.dtype}, use_torchcompile={args.use_torchcompile}")

    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16
    }
    dtype = dtype_map[args.dtype]
    config = Config(args.hidden_size, args.intermediate_size)
    
    # initialize_sglang_environment(tp_size=1)
    model = FFNModel(config, args.num_layers).to(dtype)
    model.eval()

    # tpp_replace.OptimizeForLinear(model)
    if args.use_torchcompile:
        print("Using torch.compile for the model")
        model = torch.compile(model)
    
    input_tensor = torch.randn(args.batch_size, args.seq_len, args.hidden_size, dtype=dtype)
    output, avg_time = benchmark_ffn_model(model, input_tensor, num_iter=args.num_iter)


    time_per_layer = 1000 * avg_time / args.num_layers
    print("Output shape:", output.shape)
    print(f"Average time per layer: {time_per_layer:.2f} us")
    token_len = args.batch_size * args.seq_len
    float_operations = (6 * token_len * args.hidden_size * args.intermediate_size) + (token_len * args.intermediate_size)
    size_of_dtype = torch.tensor([], dtype=dtype).element_size()
    read_data = (3 * args.hidden_size * args.intermediate_size + 2 * token_len * args.hidden_size + token_len * args.intermediate_size) * size_of_dtype

    flops = float_operations / (time_per_layer * 1e-6)
    bandwidth = read_data / (time_per_layer * 1e-6)

    # print(f"Total FLOPs for the model: {float_operations:.2e}")
    # print(f"Total data read for the model: {read_data / 1e9:.2f} GB")
    print(f"FLOPs: {flops / 1e12:.2f} TF/s, Bandwidth: {bandwidth / 1e9:.2f} GB/s")