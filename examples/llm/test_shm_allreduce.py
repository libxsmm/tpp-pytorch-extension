import torch
import time
import json
import pathlib
import argparse
import os
import numpy as np
import inspect
from typing import Tuple

try:
    import tpp_pytorch_extension as tpx
    from tpp_pytorch_extension.llm.llm_commom import jit_trace_model
except:
    pass


def comma_separated_ints(value):
    vals = value.split(",")
    for val in vals:
        try:
            int(val)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "%s is not a valid comma separated list of ints" % value
            )

    return value


# args
parser = argparse.ArgumentParser("Allreduce test and benchmark script", add_help=False)
parser.add_argument(
    "--dtype",
    type=str,
    choices=["float32", "bfloat16", "float16"],
    default="bfloat16",
    help="bfloat16, float32 or float16",
)
parser.add_argument("--num-iter", default=1000, type=int, help="num iter")
parser.add_argument("--num-warmup", default=10, type=int, help="num warmup")
parser.add_argument(
    "--sizes", default=None, type=comma_separated_ints, help="buffer size"
)
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
# By default shm_allreduce try to use shm implementation
# set USE_SHM_ALLREDUCE=0 env to force disable it
from tpp_pytorch_extension.llm.llm_common import shm_allreduce, set_pg

set_pg()

dtype = torch.bfloat16
if args.dtype == "bfloat16":
    dtype = torch.bfloat16
elif args.dtype == "float32":
    dtype = torch.float32
elif args.dtype == "float16":
    dtype = torch.half

x = torch.ones([1025]).to(dtype)
print(f"before x: {x}")
shm_allreduce(x)
print(f"after x: {x}")

if args.sizes is not None:
    sizes = np.fromstring(args.sizes, dtype=int, sep=",")
else:
    sizes = [2**i for i in range(9, 25)]

iters = args.num_iter
warmup = args.num_warmup

for sz in sizes:
    total_time = 0.0
    t_ref = torch.ones([sz]).to(dtype)
    for _ in range(warmup):
        t = t_ref.clone()
        shm_allreduce(t)

    for _ in range(iters):
        t = t_ref.clone()
        t0 = time.time()
        shm_allreduce(t)
        t1 = time.time()
        total_time += t1 - t0

    print(
        f"#Elem: {sz:10d}  Avg allreduce time: {(total_time)*1e6/iters:10.3f} usec dtype: {t.dtype}"
    )
