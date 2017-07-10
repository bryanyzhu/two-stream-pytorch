# PyTorch implementation of popular two-stream frameworks for video action recognition
============================

Current release is the PyTorch implementation of the "Towards Good Practices for Very Deep Two-Stream ConvNets". You can refer to paper for more details at [Arxiv](https://arxiv.org/abs/1507.02159).

For future, I will add PyTorch implementation for the following papers:

Temporal Segment Networks: Towards Good Practices for Deep Action Recognition,
Limin Wang, Yuanjun Xiong, Zhe Wang, Yu Qiao, Dahua Lin, Xiaoou Tang, Luc Van Gool
ECCV 2016

Deep Temporal Linear Encoding Networks
Ali Diba, Vivek Sharma, Luc Van Gool
https://arxiv.org/abs/1611.06678

Hidden Two-Stream Convolutional Networks for Action Recognition
Yi Zhu, Zhenzhong Lan, Shawn Newsam, Alexander G. Hauptmann
https://arxiv.org/abs/1704.00389


Install
=========
Tested on PyTorch:

OS: Ubuntu 16.04
Package manager: Conda
Python: 3.5
CUDA: 8.0

Code also works for Python 2.7.

Training
========

Simply run:

`python main_single_gpu.py DATA_PATH`

`DATA_PATH` is where you store RGB frames or optical flow images. Change the parameters passing to argparse as you need.

Testing
========

`Will release soon.`

Related Projects
====================
[TSN](https://github.com/yjxiong/temporal-segment-networks): Temporal Segment Networks: Towards Good Practices for Deep Action Recognition
[Hidden Two-Stream](https://github.com/bryanyzhu/Hidden-Two-Stream): Hidden Two-Stream Convolutional Networks for Action Recognition



