import sys
sys.path.append("./")

from options import Options, HParams
from utils.experiman import ExperiMan
import os
import torch
import torch.multiprocessing as mp
import logging
from utils import dist_util
import torch.distributed as dist

def main(opt, manager):
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."
    n_gpus = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(dist_util.find_free_port())
    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, opt, manager))

def run(rank, n_gpus, opt, manager):
    torch.cuda.set_device(rank)
    # Set up seed in every process
    manager.set_rank(rank)
    manager._setup_seed()
    manager._setup_logger(logging.DEBUG if rank in [-1, 0] else logging.WARN)
    if manager.is_master():
        manager._third_party_tools = ('tensorboard',)
        # Set up tensorboard in master (rank == 0)
        manager._setup_third_party_tools()

    logger = manager.get_logger()
    logger.info(f"======> N gpus / World size: {n_gpus}")
    logger.info(f"======> Log in Process: {rank}")

    dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)

    if rank == 1:
        print(opt.train.demo)


if __name__ == "__main__":
    manager = ExperiMan(name='default')
    parser = manager.get_basic_arg_parser()
    parser.set_defaults(seed=0)
    parser.set_defaults(run_name="TestDDP")
    parser.set_defaults(gpu="4,5,6")
    opt = Options(parser).parse()  # get training options

    manager.setup(opt)
    opt = HParams(**vars(opt))
    main(opt, manager)