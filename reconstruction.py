import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss, conv2d
import torch.nn.functional as F
from logger import Logger, Visualizer
import numpy as np
import imageio
import lpips


def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    pixel_max = 1.0
    psnr = 20 * torch.log10(pixel_max / torch.sqrt(mse))
    return psnr


def reconstruction(config, model, checkpoint, log_dir, dataset):
    png_dir = os.path.join(log_dir, 'reconstruction/png')
    log_dir = os.path.join(log_dir, 'reconstruction')
    model = torch.nn.DataParallel(model)
    if checkpoint is not None:
        Logger.load_cpk(checkpoint, model=model)
    else:
        print('warining: reconstruction without checkpoiont, make sure you are using the trained models...')
        # raise AttributeError("Checkpoint should be specified for mode='reconstruction'.")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=6)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(png_dir):
        os.makedirs(png_dir)

    loss_lpips = lpips.LPIPS(net='vgg').cuda()
    loss_list = []
    loss_list_lpips = []
    loss_list_psnr = []
    loss_list_ssim= []


    model.eval()

    from frames_dataset import read_video
    for it, x in tqdm(enumerate(dataloader)):
        if config['reconstruction_params']['num_videos'] is not None:
            if it > config['reconstruction_params']['num_videos']:
                break
        with torch.no_grad():
            predictions = []
            visualizations = []
            if torch.cuda.is_available():
                x['video'] = x['video'].cuda()
            for frame_idx in range(x['video'].shape[2]):
                source = x['video'][:, :, 0]
                driving = x['video'][:, :, frame_idx]
                input = {'source': source, 'driving': driving}
                out, warp_img, _, kp_s, kp_d = model(x=input, is_train=False)
                predictions.append(np.transpose(out.data.cpu().numpy(), [0, 2, 3, 1])[0])

                visualization = Visualizer(**config['visualizer_params']).visualize(source=source,
                                                                                    driving=driving, out=torch.cat([warp_img, out], dim=3),  kp_s=kp_s, kp_d=kp_d )
                visualizations.append(visualization)
                loss_list.append(torch.abs(out - driving).mean().cpu().numpy())
                loss_list_lpips.append(loss_lpips.forward(driving, out).mean().cpu().numpy())
                loss_list_psnr.append(psnr(driving, out).mean().cpu().numpy())
               
            predictions = np.concatenate(predictions, axis=1)
            imageio.imsave(os.path.join(png_dir, x['name'][0] + '.png'), (255 * predictions).astype(np.uint8))

            image_name = x['name'][0] + config['reconstruction_params']['format']
            imageio.mimsave(os.path.join(log_dir, image_name), visualizations)

    print(len(loss_list))
    print("Reconstruction loss: %s" % np.mean(loss_list))
    print("lpipis loss: %s" % np.mean(loss_list_lpips))
    print("psnr loss: %s" % np.mean(loss_list_psnr))
    return loss_list
