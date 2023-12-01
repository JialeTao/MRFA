from tqdm import trange
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from logger import Logger
import math

from torch.optim.lr_scheduler import MultiStepLR, LambdaLR

from frames_dataset import DatasetRepeater


def train(config, model, checkpoint, log_dir, dataset, device_ids, local_rank=-1, world_size=1):
    train_params = config['train_params']
    
    start_epoch = 0
    optimizer = torch.optim.Adam([{'params': model.encoder.parameters()}, {'params': model.decoder.parameters()},{'params': model.dense_motion.parameters()}], lr=train_params['lr'], betas=(0.5, 0.999))
    scheduler = MultiStepLR(optimizer, train_params['epoch_milestones'], gamma=0.1, last_epoch=start_epoch - 1)
    if train_params['bg_start'] < train_params['num_epochs']:
        optimizer_bg = torch.optim.Adam([{'params': model.bg_predictor.parameters()}], lr=train_params['lr'], betas=(0.5, 0.999))
        scheduler_bg = MultiStepLR(optimizer_bg, train_params['epoch_milestones'], gamma=0.1, last_epoch=start_epoch - 1)
    
    if checkpoint is not None:
        model = torch.nn.DataParallel(model)
        save_dict = torch.load(checkpoint, map_location='cuda:{}'.format(torch.cuda.current_device()))['model']
        state_dict = {k:v for k,v in save_dict.items() if 'decoder.pos_embedding' not in k}
        model.load_state_dict(state_dict, strict=False)
        model = model.module


    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])
    distributed = local_rank >= 0
    if distributed:
        train_sampler = DistributedSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=False, num_workers=8, drop_last=True, sampler=train_sampler)

        device = torch.device('cuda:{}'.format(torch.cuda.current_device()))
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                            find_unused_parameters=True,
                                                            device_ids=[torch.cuda.current_device()],
                                                            output_device=torch.cuda.current_device())
    else:
        dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=6, drop_last=True)
        model = nn.DataParallelWithCallback(model, device_ids=device_ids)

    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):
            if distributed:
                dataloader.sampler.set_epoch(epoch)
            for x in dataloader:
                optimizer.zero_grad()
                if epoch >= train_params['bg_start']:
                    optimizer_bg.zero_grad()
                rec, warp_img, losses_generator, kp_s, kp_d = model(x, epoch)
                loss_values = [val.mean() for val in losses_generator.values()]
                loss = sum(loss_values)
                loss.backward()
                if  train_params['clip_grad']:
                    nn.utils.clip_grad_norm_(model.module.encoder.parameters(), max_norm=train_params['clip'], norm_type = math.inf)
                    nn.utils.clip_grad_norm_(model.module.dense_motion.parameters(), max_norm=train_params['clip'], norm_type = math.inf)
                    if epoch >= train_params['bg_start']:
                        nn.utils.clip_grad_norm_(model.module.bg_predictor.parameters(), max_norm=train_params['clip'], norm_type = math.inf)
                optimizer.step()
                if epoch >= train_params['bg_start']:
                    optimizer_bg.step()

                if distributed:
                    for key in losses_generator:
                        torch.distributed.reduce(losses_generator[key], dst=0)
                        losses_generator[key] = losses_generator[key] / world_size
                losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
                if distributed:
                    if local_rank == 0:
                        logger.log_iter(losses=losses)
                else:
                    logger.log_iter(losses=losses)

            scheduler.step()
            if epoch >= train_params['bg_start']:
                scheduler_bg.step()

            if not distributed or (distributed and local_rank == 0):
                rec = torch.cat([warp_img, rec], dim=3)
                out = {'rec': rec}
                out['kp_s'] = kp_s
                out['kp_d'] = kp_d
                logger.log_epoch(epoch, {'model': model, 'optimizer': optimizer,}, inp=x, out=out)
