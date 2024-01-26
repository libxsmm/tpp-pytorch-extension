import torch
import time
import json
import pathlib
import argparse
import os
import inspect
from typing import Tuple

try:
    import tpp_pytorch_extension as tpx
    from tpp_pytorch_extension.llm.llm_commom import jit_trace_model
except:
    pass

# args
parser = argparse.ArgumentParser("Generation script", add_help=False)
parser.add_argument(
    "--dtype",
    type=str,
    choices=["float32", "bfloat16", "float16"],
    default="bfloat16",
    help="bfloat16, float32 or float16",
)
parser.add_argument("--num-iter", default=10, type=int, help="num iter")
parser.add_argument("--num-warmup", default=3, type=int, help="num warmup")
parser.add_argument("--batch-size", default=1, type=int, help="batch size")
parser.add_argument("--size", default=8192, type=int, help="buffer size")
parser.add_argument("--profile", action="store_true")
parser.add_argument("--dist-backend", default="mpi", type=str)
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

dist_init()
from tpp_pytorch_extension.llm.llm_common import shm_allreduce, set_pg

set_pg()

x = torch.ones([1024])
print(f"before x: {x}")
shm_allreduce(x)
print(f"after x: {x}")


sizes = [1024 * 4096, 4 * 4096]
sizes = [2**i for i in range(24, 9, -1)]
# sz = 16*1024
for sz in sizes:
    t = torch.ones([sz]).to(torch.bfloat16)
    for _ in range(10):
        shm_allreduce(t)

    iters = 1000
    t0 = time.time()
    for _ in range(iters):
        shm_allreduce(t)
    t1 = time.time()

    print(f"#Elem: {sz}  Avg allreduce time: {(t1-t0)*1e6/iters:.3f} usec")
