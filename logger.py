import numpy as np
import torch
import torch.nn.functional as F
import imageio

import os
from skimage.draw import circle, line

import matplotlib.pyplot as plt
import collections
import cv2


class Logger:
    def __init__(self, log_dir, checkpoint_freq=100, visualizer_params=None, zfill_num=8, log_file_name='log.txt'):

        self.loss_list = []
        self.cpk_dir = log_dir
        self.visualizations_dir = os.path.join(log_dir, 'train-vis')
        if not os.path.exists(self.visualizations_dir):
            try:
                os.makedirs(self.visualizations_dir)
            except FileExistsError:
                pass
        self.log_file = open(os.path.join(log_dir, log_file_name), 'a')
        self.zfill_num = zfill_num
        self.visualizer = Visualizer(**visualizer_params)
        self.checkpoint_freq = checkpoint_freq
        self.epoch = 0
        self.best_loss = float('inf')
        self.names = None

    def log_scores(self, loss_names):
        loss_mean = np.array(self.loss_list).mean(axis=0)

        loss_string = "; ".join(["%s - %.5f" % (name, value) for name, value in zip(loss_names, loss_mean)])
        loss_string = str(self.epoch).zfill(self.zfill_num) + ") " + loss_string

        print(loss_string, file=self.log_file)
        print(loss_string)
        self.loss_list = []
        self.log_file.flush()

    def visualize_rec(self, inp, out):
        # image = self.visualizer.visualize(inp['driving'][0:,:,:,:], inp['source'][0:,:,:,:], out[0:,:,:,:])
        # image = self.visualizer.visualize(inp['driving'], inp['source'], out['rec'], seg=out['seg'])
        image = self.visualizer.visualize(inp['driving'], inp['source'], out['rec'], out['kp_s'], out['kp_d'])
        imageio.imsave(os.path.join(self.visualizations_dir, "%s-rec.png" % str(self.epoch).zfill(self.zfill_num)), image)

    def save_cpk(self, emergent=False, models=None):
        if models is not None:
            cpk = {k: v.state_dict() for k, v in models.items()}
        else:
            cpk = {k: v.state_dict() for k, v in self.models.items()}
        cpk['epoch'] = self.epoch
        cpk_path = os.path.join(self.cpk_dir, '%s-checkpoint.pth' % str(self.epoch).zfill(self.zfill_num))
        if not (os.path.exists(cpk_path) and emergent):
            torch.save(cpk, cpk_path)

    @staticmethod
    def load_cpk(checkpoint_path, model, optimizer=None):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        return checkpoint['epoch']

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if 'models' in self.__dict__:
            self.save_cpk()
        self.log_file.close()

    def log_iter(self, losses):
        losses = collections.OrderedDict(losses.items())
        if self.names is None:
            self.names = list(losses.keys())
        self.loss_list.append(list(losses.values()))

    def log_epoch(self, epoch, models, inp, out):
        self.epoch = epoch
        self.models = models
        if (self.epoch + 1) % self.checkpoint_freq == 0:
            self.save_cpk()
        self.log_scores(self.names)
        self.visualize_rec(inp, out)


class Visualizer:
    def __init__(self, draw_border=False, colormap='gist_rainbow', kp_size=5):
        self.draw_border = draw_border
        self.colormap = plt.get_cmap(colormap)
        self.kp_size = kp_size

    def draw_image_with_kp(self, image, kp_array):
        image = np.copy(image)
        image = image.copy()
        spatial_size = np.array(image.shape[:2][::-1])[np.newaxis]
        kp_array = spatial_size * (kp_array + 1) / 2
        num_kp = kp_array.shape[0]
    
        for kp_ind, kp in enumerate(kp_array):
            rr, cc = circle(kp[1], kp[0], self.kp_size, shape=image.shape[:2])
            image[rr, cc] = np.array(self.colormap(kp_ind / num_kp ))[:3]
            # # draw number around kp
            # # print(image.shape)
            # # image = image.transpose(1, 0, 2,).copy()
            # image = (255 * image).copy()
            # image = cv2.putText(image.astype(np.uint8), str(kp_ind), (int(kp[0]), int(kp[1])), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255))
            # image = image / 255
        return image
    def create_image_column_with_kp(self, images, kp):
        image_array = np.array([self.draw_image_with_kp(v, k) for v, k in zip(images, kp)])
        return self.create_image_column(image_array)
    def create_image_column(self, images):
        if self.draw_border:
            images = np.copy(images)
            images[:, :, [0, -1]] = (1, 1, 1)
            images[:, :, [0, -1]] = (1, 1, 1)
        return np.concatenate(list(images), axis=0)

    def create_image_grid(self, *args):
        out = []
        for arg in args:
            if type(arg) == tuple:
                out.append(self.create_image_column_with_kp(arg[0], arg[1]))
            else:
                out.append(self.create_image_column(arg))
        return np.concatenate(out, axis=1)

    def visualize(self, driving, source, out, kp_s=None, kp_d=None):

        images = []

        # image with keypoints
        source = source.data.cpu()
        source = np.transpose(source, [0, 2, 3, 1])
        images.append(source)
        # images.append((source, kp_s.data.cpu().numpy()))
        driving = driving.data.cpu().numpy()
        driving = np.transpose(driving, [0, 2, 3, 1])
        images.append(driving)
        # images.append((driving, kp_d.data.cpu().numpy()))
        out = out.data.cpu().numpy()
        out = np.transpose(out, [0, 2, 3, 1])
        images.append(out)

        image = self.create_image_grid(*images)
        image = (255 * image).astype(np.uint8)
        return image
