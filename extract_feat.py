    import os
import sys
import time
import argparse
import shutil

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

from utils import video_transforms
import models
import datasets

from utils.log import log
from IPython import embed

def build_model(model_name,model_path,gpus_list,num_class):
    if not os.path.exists(model_path):
        log.l.info('model path error, please check.')
        exit()
    model=models.__dict__[model_name](pretrained=False,num_classes=num_class)
    model.load_state_dict(torch.load(model_path)['state_dict'])
    if len(gpus_list)!=1:
        nn.DataParallel(model,device_ids=gpus_list)
    model.cuda(gpus_list[0])
    return model

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Two-Stream Action Recognition')
    parser.add_argument('--model_path',default='checkpoints/model_best.pth.tar',type=str)
    parser.add_argument('--layer_name',default='None',type=str)
    parser.add_argument('--data',default='../data/ucf101/flows_tvl1',type=str)
    parser.add_argument('--dataset',default='ucf101',type=str)
    parser.add_argument('--modality',default='rgb',type=str)
    parser.add_argument('--save_path',default='feat',type=str)
    parser.add_argument('--model_name',default='vgg16',type=str)
    parser.add_argument('--gpus',default='0,1,2,3',type=str)
    parser.add_argument('--split',default=1,type=int)
    args=parser.parse_args()
    return args

def extracting_feature_func(train_loader,model,save_path,layer_name):
    pass

def extracting_features():
    global args,best_prec1
    args=parse_args()
    '''
    read the parameters
    '''
    gpus_list=[int(_) for _ in args.gpus.split(',')]
    model_path=args.model_path
    modality=args.modality
    save_path=args.save_path
    layer_name=args.layer_name
    data_root=args.data

    '''
    build model
    '''
    if args.dataset=='ucf101':
        num_class=101
    else:
        num_class=51# TBA
    model_name='{}_{}'.format(modality,args.model_name)
    model=build_model(model_name,model_path,gpus_list,num_class)
    model.eval()

    if args.modality == "rgb":
        new_length=1
        scale_ratios = [1.0, 0.875, 0.75, 0.66]
        is_color = True
        clip_mean = [0.485, 0.456, 0.406] * new_length
        clip_std = [0.229, 0.224, 0.225] * new_length
    elif args.modality == "flow":
        new_length=10
        scale_ratios = [1.0, 0.875, 0.75]
        is_color = False
        clip_mean = [0.5, 0.5] * new_length
        clip_std = [0.5, 0.5] * new_length
    else:
        log.l.info("No such modality. Only rgb and flow supported.")
        exit()

    normalize = video_transforms.Normalize(mean=clip_mean,
                                 std=clip_std)
    val_transform = video_transforms.Compose([
            video_transforms.Scale((224,224)),
            #video_transforms.CenterCrop((224)),
            video_transforms.ToTensor(),
            normalize,
        ])

    train_setting_file = "train_%s_split%d.txt" % (args.modality, args.split)
    train_split_file = os.path.join('settings/', args.dataset, train_setting_file)
    val_setting_file = "val_%s_split%d.txt" % (args.modality, args.split)
    val_split_file = os.path.join('settings/', args.dataset, val_setting_file)
    if not os.path.exists(train_split_file) or not os.path.exists(val_split_file):
        log.l.info("No split file exists in %s directory. Preprocess the dataset first" % ('settings/'))

    videos,video_len,classes=[],[],[]
    for file_name in [train_setting_file,val_setting_file]:
        with open(file_name,'rb')as fp:
            lines=fp.readlines()
            for line in lines:
                line_list=line.strip().split(' ')
                videos.append(line_list[0])
                video_len.append(int(line_list[1]))
                classes.append(int(line_list[2]))
    log.l.info('detect {} videos at all, containing {} frames in total'.format(len(videos),sum(video_len)))
    new_width,new_height=340,256


    embed()
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    extracting_feature_func(train_loader,model,save_path,layer_name)
    extracting_feature_func(val_loader,model,save_path,layer_name)

if __name__=='__main__':
    extracting_features()