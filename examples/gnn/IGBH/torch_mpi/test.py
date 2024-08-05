import torch
import torch.distributed as dist
import os

import torch_mpi

# os.environ['RANK'] = os.environ.get('PMI_RANK', 0)
# os.environ['WORLD_SIZE'] = os.environ.get('PMI_SIZE', 0)
# os.environ['MASTER_ADDR'] = '127.0.0.1'
# os.environ['MASTER_PORT'] = '29500'

if dist.is_mpi_available():
    dist.init_process_group(backend="mpi")
    print("BackEnd", dist.get_backend())
    print("rank", dist.get_rank())
    x = torch.tensor(dist.get_rank())
    dist.all_reduce(x)
    print(x)
else:
    print("MPI backend is not available")
