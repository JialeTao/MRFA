## **[NeurIPS 2023] Learning Motion Refinement for Unsupervised Face Animation**
Codes of the NeurIPS 2023 paper "Learning Motion Refinement for Unsupervised Face Animation" will be released here in a few days.

<!-- ### **Updates:** -->
**2023.12.02:** Codes are released.

## **Environments**
The model is trained with PyTorch version 1.10 and Python version 3.8. Basic installations are given in requiremetns.txt.

    pip install -r requirements.txt

## **Datasets**
Following [FOMM](https://github.com/AliaksandrSiarohin/first-order-model) to download the **Voxceleb1** dataset, and following [CelebV-HQ](https://github.com/CelebV-HQ/CelebV-HQ) to download the **CelebV-HQ** dataset. After downloading and pre-processing, the dataset should be placed in the `./data` folder or you can change the parameter `root_dir` in the config file. Note that we save the video dataset in PNG frames for better training IO performance. The tree structure of the dataset path is given in the following.

    |-- data/voxceleb1-png, data/celebvhq256-png
        |-- train
            |-- video1
                |-- 00000.png
                |-- 00001.png
                |-- ...
            |-- video2
                |-- 00000.png
                |-- 00001.png
                |-- ...
            |-- ...
        |-- test
            |-- video1
                |-- 00000.png
                |-- 00001.png
                |-- ...
            |-- video2
                |-- 00000.png
                |-- 00001.png
                |-- ...
            |-- ...

## **Training**
We train the model on 8 NVIDIA 3090 cards or 4 NVIDIA A100 cards and use the Pytorch DistributedDataPrallel.

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 run.py --config config/dataset.yaml

Please note that we utilize [MTIA](https://github.com/JialeTao/MTIA) as our prior motion model. However, it is also possible to train alternative motion models, such as [FOMM](https://github.com/AliaksandrSiarohin/first-order-model) and [TPSM](https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model). This can be achieved by modifying the parameter `train_params.prior_model` in the configuration file. Moreover, changing the parameter `raft_flow.prior_only` to `True` results in training a prior-motion-based animation model.
## **Evaluation**
Evaluate video reconstruction with the following command, for more metrics, we recommend seeing [FOMM-Pose-Evaluation](https://github.com/AliaksandrSiarohin/pose-evaluation).

    CUDA_VISIBLE_DEVICES=0 python run.py --mode reconstruction --config path/to/config --checkpoint path/to/model.pth  

## **Demo**
To make a demo animation, specify the driving video and source image, the resulting video will be saved to result.mp4.

    python demo.py --config path/to/config --checkpoint path/to/model.pth --driving_video path/to/video.mp4 --source_image path/to/image.png --result_video ./result.mp4 --adapt_scale --relative

## **Pretrained models**
Coming soon

## **Citation**
    @inproceedings{
    tao2023learning,
    title={Learning Motion Refinement for Unsupervised Face Animation},
    author={Jiale Tao and Shuhang Gu and Wen Li and Lixin Duan},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
    url={https://openreview.net/forum?id=m9uHv1Pxq7}
    }

## **Acknowledgements**
The implementation is partially borrowed from [MTIA](https://github.com/JialeTao/MTIA), [TPSM](https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model), and [FOMM](https://github.com/AliaksandrSiarohin/first-order-model). We thank the authors for their excellent work.
