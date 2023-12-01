from re import X
import matplotlib

matplotlib.use('Agg')

import os, sys
import random
import yaml
from argparse import ArgumentParser
from time import gmtime, strftime
from shutil import copy
import torch
import numpy as np

from frames_dataset import FramesDataset

import modules
from modules.model import MRFA
from modules.util import convert_dict_to_attrit_dict


from train import train
from reconstruction import reconstruction
from animate_ddp import animate
#from animate import animate


if __name__ == "__main__":

    if sys.version_info[0] < 3:
        raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--mode", default="train", choices=["train", "reconstruction", "animate"])
    parser.add_argument("--log_dir", default='./log', help="path to log into")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")
    parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print model architecture")
    parser.add_argument("--local_rank", default=-1, type=int, help="distributed machine")
    parser.set_defaults(verbose=False)

    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    distributed = opt.local_rank >= 0
    # distributed = False
    if distributed:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        rank = torch.distributed.get_rank()
        local_rank = opt.local_rank if opt.local_rank != -1 else torch.cuda.current_device()
        device = torch.device('cuda:{}'.format(local_rank))
        torch.cuda.set_device(device)
        print(local_rank, device, torch.cuda.device_count())
        world_size = torch.distributed.get_world_size()
        print("world_size : ", world_size)
        config['train_params']['batch_size'] = int(config['train_params']['batch_size'] / world_size)
    else:
        device = torch.device('cuda:{}'.format(opt.device_ids[0]))
        world_size=1

    if opt.checkpoint is not None and opt.mode != "train":
        log_dir = os.path.join(*os.path.split(opt.checkpoint)[:-1])
    else:
        # log_dir = os.path.join(opt.log_dir, os.path.basename(opt.config).split('.')[0])
        log_dir = opt.log_dir + '_' + os.path.basename(opt.config).split('.')[0]
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
        copy(opt.config, log_dir)

    cfg = convert_dict_to_attrit_dict(config)
    model = MRFA(cfg)
    model.to(device)
    
    dataset = FramesDataset(is_train=(opt.mode == 'train'), **config['dataset_params'])

    if opt.mode == 'train':
        print("Training...")
        train(config, model, opt.checkpoint, log_dir, dataset, opt.device_ids, local_rank=rank, world_size=world_size)
    elif opt.mode == 'reconstruction':
        print("Reconstruction...")
        reconstruction(config, model, opt.checkpoint, log_dir, dataset)
    elif opt.mode == 'animate':
        print("Animate...")
        animate(config, model, opt.checkpoint, log_dir, dataset, local_rank=local_rank)
