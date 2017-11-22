# PyTorch implementation of popular two-stream frameworks for video action recognition
============================

Current release is the PyTorch implementation of the "Towards Good Practices for Very Deep Two-Stream ConvNets". You can refer to paper for more details at [Arxiv](https://arxiv.org/abs/1507.02159).

For future, I will add PyTorch implementation for the following papers:

```
Temporal Segment Networks: Towards Good Practices for Deep Action Recognition,
Limin Wang, Yuanjun Xiong, Zhe Wang, Yu Qiao, Dahua Lin, Xiaoou Tang, Luc Van Gool
ECCV 2016

Deep Temporal Linear Encoding Networks
Ali Diba, Vivek Sharma, Luc Van Gool
https://arxiv.org/abs/1611.06678

Hidden Two-Stream Convolutional Networks for Action Recognition
Yi Zhu, Zhenzhong Lan, Shawn Newsam, Alexander G. Hauptmann
https://arxiv.org/abs/1704.00389
```

Install
=========

Tested on PyTorch:

```
OS: Ubuntu 16.04
Package manager: Conda
Python: 3.5
CUDA: 8.0
```

Code also works for Python 2.7.

Training
========

For spatial stream (single RGB frame), run:

`python main_single_gpu.py DATA_PATH -m rgb -a rgb_vgg16 --new_length=1 --epochs 250 --lr 0.001 --lr_steps 100 200`

For temporal stream (10 consecutive optical flow images), run:

`python main_single_gpu.py DATA_PATH -m flow -a flow_vgg16 --new_length=10 --epochs 750 --lr 0.005 --lr_steps 250 500`

`DATA_PATH` is where you store RGB frames or optical flow images. Change the parameters passing to argparse as you need.

Testing
========

Go into "scripts/eval_ucf101_pytorch" folder, run `python spatial_demo.py` to obtain spatial stream result, and run `python temporal_demo.py` to obtain temporal stream result. Change those label files before running the script. 

Currenly, for VGG16 model, I can obtain a 78.5% accuracy for spatial stream and 80.4% for temporal stream on the split 1 of UCF101 dataset. The spatial result is close to the number reported in original paper, but flow result is 5% away. Maybe there are subtle bugs in the flow model or how I did the preprocessing of flow images is wrong. Welcome any comments if you found the reason why there is a performance gap. If you need the pre-trained models, just email me. I will find a place to hold the models soon for easy sharing.

I am experimenting with ResNet and DenseNet now, will release the code in a couple of days. Stay tuned. 

Related Projects
====================

[TSN](https://github.com/yjxiong/temporal-segment-networks): Temporal Segment Networks: Towards Good Practices for Deep Action Recognition

[Hidden Two-Stream](https://github.com/bryanyzhu/Hidden-Two-Stream): Hidden Two-Stream Convolutional Networks for Action Recognition



