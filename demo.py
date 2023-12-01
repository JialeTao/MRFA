import matplotlib
matplotlib.use('Agg')
import os, sys
import yaml
from argparse import ArgumentParser
from tqdm import tqdm

import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
from frames_dataset import read_video
# from modules.raft import RaftFlow
# from modules.kp_detector import KPDetector, TPSKPDetector
# from modules.dense_motion import DenseMotionNetwork, TPSDenseMotionNetwork
# from modules.bg_motion_predictor import BGMotionPredictor
# from modules.transformer.pose_tokenpose_b import get_pose_net
from modules.model import MRFA
from modules.util import convert_dict_to_attrit_dict, AntiAliasInterpolation2d

from animate_ddp import normalize_kp
from scipy.spatial import ConvexHull

down =  AntiAliasInterpolation2d(3, 0.25).cuda()

if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

def load_checkpoints(cfg, checkpoint_path, cpu=False):

    # with open(config_path) as f:
    #     config = yaml.load(f)
    # cfg = convert_dict_to_attrit_dict(config)
    model = MRFA(cfg).cuda().eval()
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])

    kp_detector = model.module.encoder
    dense_motion_network = model.module.dense_motion
    decoder = model.module.decoder

    return kp_detector, dense_motion_network, decoder


def make_animation(cfg, source_image, driving_video, kp_detector, dense_motion_network, decoder, relative=True, adapt_movement_scale=True, cpu=False):
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2).cuda()
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3).cuda()
        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving[:, :, 0])

        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            if not cpu:
                driving_frame = driving_frame.cuda()
            kp_driving = kp_detector(driving_frame)
            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                   use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
            
            dense_motion = dense_motion_network(source, kp_norm, kp_source, bg_param=None)
            if cfg.train_params['prior_model'] == 'tpsm':
                kp_s_value = kp_source['kp'].view(source.shape[0], -1, 5, 2).mean(2)
                kp_d_value = kp_norm['kp'].view(driving.shape[0], -1, 5, 2).mean(2)
            else:
                kp_s_value = kp_source['kp']
                kp_d_value = kp_norm['kp']
            out, warp_img, occlusion = decoder(kp_s_value, kp_d_value, dense_motion, img=down(source), img_full=source)
            predictions.append(np.transpose(out.data.cpu().numpy(), [0, 2, 3, 1])[0])
    return predictions

def find_best_frame(source, driving, cpu=False):
    import face_alignment

    def normalize_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True,
                                      device='cpu' if cpu else 'cuda')
    kp_source = fa.get_landmarks(255 * source)[0]
    kp_source = normalize_kp(kp_source)
    norm  = float('inf')
    frame_num = 0
    for i, image in tqdm(enumerate(driving)):
        kp_driving = fa.get_landmarks(255 * image)[0]
        kp_driving = normalize_kp(kp_driving)
        new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
        if new_norm < norm:
            norm = new_norm
            frame_num = i
    return frame_num

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--checkpoint", default='vox-cpk.pth.tar', help="path to checkpoint to restore")

    parser.add_argument("--source_image", default='sup-mat/source.png', help="path to source image")
    parser.add_argument("--driving_video", default='sup-mat/source.png', help="path to driving video")
    parser.add_argument("--result_video", default='result.mp4', help="path to output")
    parser.add_argument("--img_shape", default=256, type=int, help="input shape")

    parser.add_argument("--relative", dest="relative", action="store_true", help="use relative or absolute keypoint coordinates")
    parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true", help="adapt movement scale based on convex hull of keypoints")

    parser.add_argument("--find_best_frame", dest="find_best_frame", action="store_true",
                        help="Generate from the frame that is the most alligned with source. (Only for faces, requires face_aligment lib)")

    parser.add_argument("--best_frame", dest="best_frame", type=int, default=None,
                        help="Set frame to start from.")

    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")


    parser.set_defaults(relative=False)
    parser.set_defaults(adapt_scale=False)

    opt = parser.parse_args()

    with open(opt.config) as f:
        config = yaml.load(f)
    cfg = convert_dict_to_attrit_dict(config)

    source_image = imageio.imread(opt.source_image)
    driving_video = read_video(opt.driving_video, frame_shape=(opt.img_shape, opt.img_shape))
    source_image = resize(source_image, (opt.img_shape, opt.img_shape))[..., :3]
    # driving_video = [resize(frame, (opt.img_shape, opt.img_shape))[..., :3] for frame in driving_video]
    fps = 25

    # reader = imageio.get_reader(opt.driving_video)
    # fps = reader.get_meta_data()['fps']
    # driving_video = []
    # try:
    #     for im in reader:
    #         driving_video.append(im)
    # except RuntimeError:
    #     pass
    # reader.close()

   
    kp_detector, dense_motion_network, decoder = load_checkpoints(cfg=cfg, checkpoint_path=opt.checkpoint, cpu=opt.cpu)

    if opt.find_best_frame or opt.best_frame is not None:
        i = opt.best_frame if opt.best_frame is not None else find_best_frame(source_image, driving_video, cpu=opt.cpu)
        print ("Best frame: " + str(i))
        driving_forward = driving_video[i:]
        driving_backward = driving_video[:(i+1)][::-1]
        predictions_forward = make_animation(source_image, driving_forward, generator, kp_detector, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu)
        predictions_backward = make_animation(source_image, driving_backward, generator, kp_detector, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu)
        predictions = predictions_backward[::-1] + predictions_forward[1:]
    else:
        predictions = make_animation(cfg, source_image, driving_video, kp_detector, dense_motion_network, decoder, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu)
    imageio.mimsave(opt.result_video, [img_as_ubyte(frame) for frame in predictions], fps=fps)

