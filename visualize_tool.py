import os,sys
import numpy as np
import matplotlib.pyplot as plt
import argparse

#visualize_item='Loss'# alternative with 'Prec@1'

def visualize_statics(vector,gap):
    plt.plot(vector[::gap])
    plt.show()

def analysis_training_contents(contents):
    key_contents=[]
    for line in contents:
        if len(line.split(' '))>6:
            if line.split(' ')[5]=='Epoch:':
                Item_info=line.split('\t')[item_ind]
                key_contents.append(float(Item_info.split(' ')[1]))
    if if_show:
        visualize_statics(key_contents,args.gap)
    return key_contents

def judge_train_log(contents):
    is_training_log=0
    for line in contents:
        if len(line.split(' '))>6:
            if line.split(' ')[5]=='Epoch:':
                is_training_log=1
                return is_training_log
    return is_training_log

def analysis_file(log_file):
    with open(log_file,'rb')as fp:
        contents=fp.readlines()
    is_training_log=judge_train_log(contents)
    if is_training_log:
        analysis_training_contents(contents)

def main():
    global args
    parser=argparse.ArgumentParser(description='this is the argument parser')
    parser.add_argument('--log_path',default=None,type=str)
    parser.add_argument('--item',default=None,type=str)
    parser.add_argument('--gap',default=1,type=int)
    args=parser.parse_args()

    global item_ind;item_ind=3 if args.item=='Loss' else 4
    global if_show;if_show=True

    analysis_file(args.log_path)
if __name__=='__main__':
    main()