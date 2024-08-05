import os, sys
import torch as th
import torch.distributed as dist

def init_mpi(dist_backend, dist_url):
    world_size = -1
    if dist_backend == 'ccl':
        try:
            import oneccl_bindings_for_pytorch
        except:
            print("CCL backend requested but import oneccl_bindings_for_pytorch failed")
            raise
    elif dist_backend == 'mpi':
        if not th.distributed.is_mpi_available():
            try:
                import torch_mpi
                print("imported torch_mpi.........")
            except:
                print("MPI backend requested but not available try installing torch_mpi module")
                raise
        else:
            raise ValueError(f"{dist_backend} backend requested but not supported")

    if dist_url == "env://" and world_size == -1:
        world_size = int(os.environ.get("PMI_SIZE", -1))
        if world_size == -1: world_size = int(os.environ["WORLD_SIZE"])


    distributed = world_size > 1
    if distributed:
        rank = int(os.environ.get("PMI_RANK", -1))
        if rank == -1: rank = int(os.environ["RANK"])
        dist.init_process_group(backend=dist_backend, init_method=dist_url,
                                world_size=world_size, rank=rank)
    if rank == 0:
        print("Rank: ", rank ," World_size: ", world_size)
    return rank, world_size
