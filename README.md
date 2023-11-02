# Applying PVT to Object Detection

Our detection code is developed on top of [MMDetection v2.13.0](https://github.com/open-mmlab/mmdetection/tree/v2.13.0).

For details see [Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions](https://arxiv.org/pdf/2102.12122.pdf). 

If you use this code for a paper please cite:

PVTv1
```
@misc{wang2021pyramid,
      title={Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions}, 
      author={Wenhai Wang and Enze Xie and Xiang Li and Deng-Ping Fan and Kaitao Song and Ding Liang and Tong Lu and Ping Luo and Ling Shao},
      year={2021},
      eprint={2102.12122},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

PVTv2
```
@misc{wang2021pvtv2,
      title={PVTv2: Improved Baselines with Pyramid Vision Transformer}, 
      author={Wenhai Wang and Enze Xie and Xiang Li and Deng-Ping Fan and Kaitao Song and Ding Liang and Tong Lu and Ping Luo and Ling Shao},
      year={2021},
      eprint={2106.13797},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

pest-PVT
```
@misc{
      title={Accurate Multi-Class and Dense Detection of Small Pests Base on Transformer}, 
      author={Chen Hongrui and Wen Changji and Zhang Long and Ma Zhenyu and Zhang Tian and Liu Tianyu and Wang Guangyao and Yu Helong and Yang Ce and Ren Junfeng},
      year={2023},
      primaryClass={cs.CV}
}
```


## Usage

Install [MMDetection v2.13.0](https://github.com/open-mmlab/mmdetection/tree/v2.13.0).

or

```
pip install mmdet==2.13.0 --user
```

Apex (optional):
```
git clone https://github.com/NVIDIA/apex
cd apex
python setup.py install --cpp_ext --cuda_ext --user
```

If you would like to disable apex, modify the type of runner as `EpochBasedRunner` and comment out the following code block in the configuration files:
```
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)
```

## Data preparation

Prepare COCO according to the guidelines in [MMDetection v2.13.0](https://github.com/open-mmlab/mmdetection/tree/v2.13.0).


## Results and models

- pest-PVT on pest24


| Method     | Backbone | Pretrain    | Lr schd | Aug | box AP | mask AP | Config                                               | Download |
|------------|----------|-------------|:-------:|:---:|:------:|:-------:|------------------------------------------------------|----------|
| pest-PVT  | PVTv2-b2 | ImageNet-1K |    1x   |  No |  77.24  |    -    | [config](configs/atss_pvtv2_dyhead3_ass.py) | Uploading soon


## Evaluation
To evaluate pest-PVT on a single node with 8 gpus run:
```
dist_test.sh configs/atss_pvtv2_dyhead3_ass.py /path/to/checkpoint_file 8 --out results.pkl --eval bbox
```

## Training
To train pest-PVT on a single node with 8 gpus for 12 epochs run:

```
dist_train.sh configs/atss_pvtv2_dyhead3_ass.py /path/to/checkpoint_file 8
```
./dist_train.sh .configs/atss_pvtv2_dyhead3_ass.py /path/to/checkpoint_file 1
自己训练数据集需要改动mmdet文件，具体：https://blog.csdn.net/qq_39435411/article/details/120756080

python test.py ./work_dirs/atss_pvtv2_dyhead3_ass/atss_pvtv2_dyhead3_ass.py ./work_dirs/atss_pvtv2_dyhead3_ass/latest.pth --out aaa.pkl --eval mAP

## Demo
```
python demo.py demo.jpg /path/to/config_file /path/to/checkpoint_file
```


## Calculating FLOPS & Params

```
python get_flops.py configs/atss_pvtv2_dyhead3_ass.py
```
This should give
```
Input shape: (3, 1280, 800)
Flops: 7.82 GFLOPs
Params: 24.74 M
```

# License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.
