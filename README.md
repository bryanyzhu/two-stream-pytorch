# PyTorch implementation of popular two-stream frameworks for video action recognition

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

## Installation

Tested on PyTorch:
```
OS: Ubuntu 16.04
Python: 3.5
CUDA: 8.0
OpenCV3
dense_flow
```
To successfully install [dense_flow](https://github.com/yjxiong/dense_flow/tree/opencv-3.1)(branch opencv-3.1), you probably need to install opencv3 with [opencv_contrib](https://github.com/opencv/opencv_contrib). (For opencv-2.4.13, dense_flow will be installed more easily without  opencv_contrib, but you should run code of this repository under opencv3 to avoid error)

Code also works for Python 2.7.

## Data Preparation
Download data [UCF101](http://crcv.ucf.edu/data/UCF101.php) and use `unrar x UCF101.rar` to extract the videos.

Convert video to frames and extract optical flow
```
python build_of.py --src_dir ./UCF-101 --out_dir ./ucf101_frames --df_path <path to dense_flow>
```
build file lists for training and validation
```
python build_file_list.py --frame_path ./ucf101_frames --out_list_path ./settings
```

## Training

For spatial stream (single RGB frame), run:
```
python main_single_gpu.py DATA_PATH -m rgb -a rgb_resnet152 --new_length=1
--epochs 250 --lr 0.001 --lr_steps 100 200
```

For temporal stream (10 consecutive optical flow images), run:
```
python main_single_gpu.py DATA_PATH -m flow -a flow_resnet152
--new_length=10 --epochs 350 --lr 0.001 --lr_steps 200 300
```

`DATA_PATH` is where you store RGB frames or optical flow images. Change the parameters passing to argparse as you need.

## Testing

Go into "scripts/eval_ucf101_pytorch" folder, run `python spatial_demo.py` to obtain spatial stream result, and run `python temporal_demo.py` to obtain temporal stream result. Change those label files before running the script.

For ResNet152, I can obtain a 85.60% accuracy for spatial stream and 85.71% for temporal stream on the split 1 of UCF101 dataset. The result looks promising.
[Pre-trained RGB_ResNet152 Model](https://drive.google.com/open?id=1BU8TyW7u-skmkQFAVlQhA_5ZZvugZXAt)
[Pre-trained Flow_ResNet152 Model](https://drive.google.com/open?id=1KPoPYAslsdOMXbtqfi2y8TTn7zDEz898)

For VGG16, I can obtain a 78.5% accuracy for spatial stream and 80.4% for temporal stream on the split 1 of UCF101 dataset. The spatial result is close to the number reported in original paper, but flow result is 5% away. There are several reasons, maybe the pretained VGG16 model in PyTorch is differnt from Caffe, maybe there are subtle bugs in my VGG16 flow model. Welcome any comments if you found the reason why there is a performance gap.
[Pre-trained RGB_VGG16 Model](https://drive.google.com/open?id=1o-83QlDXN1EC4HVgNfJtDNvYCs72A26O)
[Pre-trained Flow_VGG16 Model](https://drive.google.com/open?id=1mATFI0QAHj6OgzJLzw9fhXNH1kpzQmDo)

I am experimenting with memory efficient DenseNet now, will release the code in a couple of days. Stay tuned.

## Related Projects

[TSN](https://github.com/yjxiong/temporal-segment-networks): Temporal Segment Networks: Towards Good Practices for Deep Action Recognition

[Hidden Two-Stream](https://github.com/bryanyzhu/Hidden-Two-Stream): Hidden Two-Stream Convolutional Networks for Action Recognition
