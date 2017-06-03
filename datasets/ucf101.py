import torch.utils.data as data

import os
import sys
import random
import numpy as np
import cv2

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def cv2_loader(path):
    return cv2.imread(path)

def read_split_file(root, split_file):

    if not os.path.exists(split_file):
        print("Split file for ucf101 dataset doesn't exist.")
        sys.exit()
    else:
        clips = []
        with open(split_file) as split_f:  
            data = split_f.readlines()
            for line in data:
                line_info = line.split()
                clip_path = os.path.join(root, line_info[0])
                duration = int(line_info[1])
                target = int(line_info[2])
                item = (clip_path, duration, target)
                clips.append(item)
    return clips
           

class ucf101(data.Dataset):

    def __init__(self, root, split_file, phase, new_length=1, transform=None, target_transform=None,
                 video_transform=None, loader=cv2_loader):
        classes, class_to_idx = find_classes(root)
        clips = read_split_file(root, split_file)
        
        if len(clips) == 0:
            raise(RuntimeError("Found 0 video clips in subfolders of: " + root + "\n"
                               "Check your data directory."))
        
        self.root = root
        self.split_file = split_file
        self.phase = phase
        self.clips = clips
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.new_length = new_length
        self.transform = transform
        self.target_transform = target_transform
        self.video_transform = video_transform
        self.loader = loader

    def __getitem__(self, index):
        path, duration, target = self.clips[index]
        frame_list = os.listdir(path)
        frame_list.sort()
        if self.phase == "train":
            sampled_frameID = random.randint(0, duration-self.new_length)
        elif self.phase == "val":
            if duration >= self.new_length:
                sampled_frameID = int((duration - self.new_length + 1)/2)
            else:
                sampled_frameID = 0
        else:
            print("No such phase. Only train and val are supported.")
            
        sampled_list = []
        for frame_id in range(self.new_length):
            fname = os.path.join(path, frame_list[sampled_frameID+frame_id])
            if is_image_file(fname):
                img = self.loader(fname)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                sampled_list.append(img)
        clip_input = np.concatenate(sampled_list, axis=2)

        if self.transform is not None:
            clip_input = self.transform(clip_input)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.video_transform is not None:
            clip_input = self.video_transform(clip_input)

        return clip_input, target

    def __len__(self):
        return len(self.clips)
