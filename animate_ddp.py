import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from frames_dataset import PairedDataset
from logger import Logger, Visualizer
from modules.util import AntiAliasInterpolation2d
import imageio
from scipy.spatial import ConvexHull
import numpy as np



def normalize_kp(kp_source, kp_driving, kp_driving_initial, adapt_movement_scale=False,
                 use_relative_movement=False, use_relative_jacobian=False):
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source['kp'][0].data.cpu().numpy()).volume
        driving_area = ConvexHull(kp_driving_initial['kp'][0].data.cpu().numpy()).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    kp_new = {k: v for k, v in kp_driving.items()}

    if use_relative_movement:
        kp_value_diff = (kp_driving['kp'] - kp_driving_initial['kp'])
        kp_value_diff *= adapt_movement_scale
        kp_new['kp'] = kp_value_diff + kp_source['kp']

        if use_relative_jacobian:
            jacobian_diff = torch.matmul(kp_driving['jacobian'], torch.inverse(kp_driving_initial['jacobian']))
            kp_new['jacobian'] = torch.matmul(jacobian_diff, kp_source['jacobian'])

    return kp_new


def animate(config, model, checkpoint, log_dir, dataset,  local_rank=-1):
    log_dir = os.path.join(log_dir, 'animation')
    # png_dir = os.path.join(log_dir, 'png')
    animate_params = config['animate_params']
    adapt_movement_scale = animate_params['adapt_movement_scale']
    relative_movement = animate_params['use_relative_movement']
    relative_jacobian = animate_params['use_relative_jacobian']

    dataset = PairedDataset(initial_dataset=dataset, number_of_pairs=animate_params['num_pairs'])
    print(len(dataset))
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    train_sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, sampler=train_sampler)
    device = torch.device('cuda:{}'.format(local_rank))

    if checkpoint is not None:
        # Logger.load_cpk(checkpoint, generator=generator, kp_detector=kp_detector)
        model = torch.nn.DataParallel(model)
        checkpoint = torch.load(checkpoint, map_location='cuda:{}'.format(local_rank))
        model.load_state_dict(checkpoint['model'], strict=False)
        model = model.module
        model.eval().to(device)

        kp_detector = model.encoder
        dense_motion_network = model.dense_motion
        decoder = model.decoder
        down =  AntiAliasInterpolation2d(3, 0.25).to(device)
    else:
        raise AttributeError("Checkpoint should be specified for mode='animate'.")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # if not os.path.exists(png_dir):
        # os.makedirs(png_dir)


    from frames_dataset import read_video
    for it, x in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            predictions = []
            visualizations = []

            if torch.cuda.is_available():
                x['driving_video'] = x['driving_video'].cuda()
                x['source_video'] = x['source_video'].cuda()
            driving = x['driving_video']
            source = x['source_video'][:, :, 0, :, :]
            kp_source = kp_detector(source)
            kp_driving_initial = kp_detector(driving[:, :, 0])
            # ## relative animation
            for frame_idx in tqdm(range(driving.shape[2])):
                driving_frame = driving[:, :, frame_idx]
                kp_driving = kp_detector(driving_frame)
                kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                    kp_driving_initial=kp_driving_initial, use_relative_movement=relative_movement,
                                    use_relative_jacobian=relative_jacobian, adapt_movement_scale=adapt_movement_scale)
                
                dense_motion = dense_motion_network(source, kp_norm, kp_source, bg_param=None)
                if cfg.train_params['prior_model'] == 'tpsm':
                    kp_s_value = kp_source['kp'].view(source.shape[0], -1, 5, 2).mean(2)
                    kp_d_value = kp_norm['kp'].view(driving.shape[0], -1, 5, 2).mean(2)
                else:
                    kp_s_value = kp_source['kp']
                    kp_d_value = kp_norm['kp']
                out, warp_img, occlusion = decoder(kp_s_value, kp_d_value, dense_motion, img=down(source), img_full=source)
                visualization = Visualizer(**config['visualizer_params']).visualize(source=source, driving=driving_frame, out=out)
                visualizations.append(visualization)
            # ## relative animation

            # ## absolute animation
            # for frame_idx in range(driving.shape[2]):
            #     driving_frame = driving[:, :, frame_idx]
            #     input = {'source': source, 'driving': driving}
            #     out, warp_img, _, kp_s, kp_d = model(input)

            #     visualization = Visualizer(**config['visualizer_params']).visualize(source=source,
            #                                                                         driving=driving_frame, out=out)
            #     visualization = visualization
            #     visualizations.append(visualization)
            # ## absolute animation

            result_name = "-".join([x['driving_name'][0], x['source_name'][0]])
            image_name = result_name + '.mp4'
            imageio.mimsave(os.path.join(log_dir, image_name), visualizations)
