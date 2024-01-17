import os
import torch
import habana_frameworks.torch.core as htcore
import platform
torch.manual_seed(0)
#load hpu backend for PyTorch
device = torch.device('hpu')

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12340'
    #Import the distributed package for HCCL, set the backend to HCCL
    import habana_frameworks.torch.distributed.hccl
    torch.distributed.init_process_group(backend='hccl', rank=rank, world_size=world_size)
    
def cleanup():
    torch.distributed.destroy_process_group()
def allReduce(rank):
    _tensor = torch.ones(8).to(device)
    torch.distributed.all_reduce(_tensor)
    _tensor_cpu = _tensor.cpu()
def run_allreduce(rank, world_size):
    setup(rank, world_size)
    for i in range(100):
        allReduce(rank)
    cleanup()
def main():
    #Run Habana's Initialize HPU function to collect the world size and rank
    from habana_frameworks.torch.distributed.hccl import initialize_distributed_hpu
    world_size, rank, local_rank = initialize_distributed_hpu()
    print(f'######## init done, world size: {world_size}, rank: {rank} ########')
    run_allreduce(rank, world_size)
    print(f'######## Done rank: {rank} ########')
if __name__ == '__main__':
    main()