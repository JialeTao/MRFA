import os
from skimage import io, img_as_float32
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
import imageio
from imageio import mimread

import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from augmentation import AllAugmentationTransform
import glob

import time

def read_video(name, frame_shape=(256,256), read_first_frame=False):
    """
    Read video which can be:
      - an image of concatenated frames
      - '.mp4' and'.gif'
      - folder with videos
    """

    if os.path.isdir(name):
        frames = sorted(os.listdir(name))
        num_frames = len(frames)
        if not read_first_frame:
            video_array = np.array(
                [img_as_float32(io.imread(os.path.join(name, frames[idx]))) for idx in range(num_frames)])
        else:
            video_array = np.array(
                [img_as_float32(io.imread(os.path.join(name, frames[idx]))) for idx in range(1)])
    elif name.lower().endswith('.png') or name.lower().endswith('.jpg'):
        image = io.imread(name)

        if len(image.shape) == 2 or image.shape[2] == 1:
            image = gray2rgb(image)

        if image.shape[2] == 4:
            image = image[..., :3]

        image = img_as_float32(image)

        video_array = np.moveaxis(image, 1, 0)

        video_array = video_array.reshape((-1,) + frame_shape)
        video_array = np.moveaxis(video_array, 1, 2)
    elif name.lower().endswith('.gif') or name.lower().endswith('.mp4') or name.lower().endswith('.mov'):
        #video = np.array(mimread(name,memtest=False))
        reader = imageio.get_reader(name)
        driving_video = []
        try:
            for im in reader:
                driving_video.append(im)
                if read_first_frame:
                    break
        except RuntimeError:
            pass
        reader.close()
        video = np.array(driving_video)
        if len(video.shape) == 3:
            video = np.array([gray2rgb(frame) for frame in video])
        if video.shape[-1] == 4:
            video = video[..., :3]
        video_array = img_as_float32(video)
    else:
        raise Exception("Unknown file extensions  %s" % name)

    return video_array


class FramesDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    """

    def __init__(self, root_dir, frame_shape=(256, 256, 3), id_sampling=False, is_train=True,
                 pairs_list=None, augmentation_params=None,read_first_frame=False,split_len=None):
        self.root_dir = root_dir
        # self.videos = os.listdir(root_dir)
        self.frame_shape = tuple(frame_shape)
        self.pairs_list = pairs_list
        self.id_sampling = id_sampling
        self.read_first_frame = read_first_frame


        if id_sampling:
            if split_len is not None:
                train_videos = {os.path.basename(video)[0:split_len] for video in os.listdir(os.path.join(root_dir, 'train'))}
            else:
                train_videos = {os.path.basename(video).split('#')[0] for video in os.listdir(os.path.join(root_dir, 'train'))}
            train_videos = list(train_videos)
        else:
            train_videos = os.listdir(os.path.join(root_dir, 'train'))
        test_videos = os.listdir(os.path.join(root_dir, 'test'))
        self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')

        if is_train:
            self.videos = train_videos
        else:
            self.videos = test_videos

        # ## For animation on all dataset
        # train_videos = os.listdir(os.path.join(root_dir, 'train'))
        # for i in range(len(train_videos)):
        #     train_videos[i] = os.path.join('train', train_videos[i])
        # for i in range(len(test_videos)):
        #     test_videos[i] = os.path.join('test', test_videos[i])
        # self.videos = train_videos+test_videos
        # self.root_dir = root_dir
        # ## For animation on all dataset

        self.is_train = is_train

        if self.is_train:
            self.transform = AllAugmentationTransform(**augmentation_params)
        else:
            self.transform = None

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):

        if self.is_train and self.id_sampling:   
            name = self.videos[idx]
            path = np.random.choice(glob.glob(os.path.join(self.root_dir, name + '*.mp4')))
        else:
            name = self.videos[idx]
            path = os.path.join(self.root_dir, name)
        video_name = os.path.basename(path)

        if self.is_train:
            frames = os.listdir(path)
            num_frames = len(frames)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2))

            video_array = []
            for idx in frame_idx:
                try:
                    img = img_as_float32(io.imread(os.path.join(path, frames[idx])))
                except TypeError:
                    img = img_as_float32(io.imread(os.path.join(path, frames[idx].decode())))
                if len(img.shape) == 2:
                    img = gray2rgb(img)
                if img.shape[-1] == 4:
                    img = img[..., :3]
                # img = (img-0.5) / 0.5

                video_array.append(img)
        else:
            video_array = read_video(path, frame_shape=self.frame_shape, read_first_frame=self.read_first_frame)
            num_frames = len(video_array)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2)) if self.is_train else range(num_frames)
            video_array = video_array[frame_idx]

        if self.transform is not None:
            video_array = self.transform(video_array)

        out = {}
        if self.is_train:
            source = np.array(video_array[0], dtype='float32')
            driving = np.array(video_array[1], dtype='float32')
            # out['driving'] = np.concatenate(driving.transpose((0, 3, 1, 2)))
            out['driving'] = driving.transpose((2, 0, 1))
            out['source'] = source.transpose((2, 0, 1))
        else:
            video = np.array(video_array, dtype='float32')
            out['video'] = video.transpose((3, 0, 1, 2))

        out['name'] = video_name

        return out


class DatasetRepeater(Dataset):
    """
    Pass several times over the same dataset for better i/o performance
    """

    def __init__(self, dataset, num_repeats=100):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return self.num_repeats * self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset[idx % self.dataset.__len__()]


class PairedDataset(Dataset):
    """
    Dataset of pairs for animation.
    """

    def __init__(self, initial_dataset, number_of_pairs, seed=0):
        self.initial_dataset = initial_dataset
        pairs_list = self.initial_dataset.pairs_list

        np.random.seed(seed)

        if pairs_list is None:
            max_idx = min(number_of_pairs, len(initial_dataset))
            nx, ny = max_idx, max_idx
            xy = np.mgrid[:nx, :ny].reshape(2, -1).T
            number_of_pairs = min(xy.shape[0], number_of_pairs)
            self.pairs = xy.take(np.random.choice(xy.shape[0], number_of_pairs, replace=False), axis=0)
        else:
            videos = self.initial_dataset.videos
            name_to_index = {name: index for index, name in enumerate(videos)}
            pairs = pd.read_csv(pairs_list)
            pairs = pairs[np.logical_and(pairs['source'].isin(videos), pairs['driving'].isin(videos))]

            number_of_pairs = min(pairs.shape[0], number_of_pairs)
            self.pairs = []
            self.start_frames = []
            for ind in range(number_of_pairs):
                self.pairs.append((name_to_index[pairs['driving'].iloc[ind]], name_to_index[pairs['source'].iloc[ind]]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        self.initial_dataset.read_first_frame=False
        first = self.initial_dataset[pair[0]]
        self.initial_dataset.read_first_frame=True
        second = self.initial_dataset[pair[1]]
        first = {'driving_' + key: value for key, value in first.items()}
        second = {'source_' + key: value for key, value in second.items()}

        return {**first, **second}
