import os
import sys
import time
import argparse
import shutil
import numpy as np
import torch
from torch.autograd import Variable
from torch import optim
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data as data
import torch.nn.functional as F
from utils import video_transforms
import models
import datasets
from torch.nn.utils import clip_grad_norm
from utils.log import log
from IPython import embed

'''init the hyper-params:'''
gt_root='settings/ucf101/'
print_train_info_frequency=10
print_val_info_frequency=100

val_frequency=1
lr_init=0.0001
weight_decay=0.0005
testing_batch_size=1
training_batch_size=10
use_cuda=False;use_cuda = use_cuda and torch.cuda.is_available()

class data_extractor(data.Dataset):#feat
    def __init__(self,root='../data/ucf101/feature_rgb_pool_2',
               type='train',
               frames_len=12,
               target_size=101,
               feat_len=1024,
               gt_file_name='train_flow_split1.txt',
               test_type=1):
        self.root=root
        self.type=type
        self.frames_len=frames_len
        self.target_size=target_size
        self.feat_len=feat_len
        self.gt_file=os.path.join(gt_root,gt_file_name)
        self.file_names,self.labels=self.get_files(self.gt_file)
        self.test_type=test_type

    def __getitem__(self,index):
        file_name=self.file_names[index]
        label=self.labels[index]
        file_path=os.path.join(self.root,'{}.npy'.format(file_name))
        data=np.load(file_path)
        len_data=len(data)
        if self.type=='train':
            i_start=np.random.randint(len_data-self.frames_len)
            out_data=data[i_start:i_start+self.frames_len]
            return out_data,label
        elif self.type=='val':
            if self.test_type==1:
                self.segments=np.int((len_data-self.frames_len)/(self.frames_len/2))+1
                tick=self.frames_len/2#(len_data-self.frames_len+1)/self.segments
                offsets=np.array([int(tick * x) for x in range(self.segments)])
                #out_data=np.zeros([self.segments,self.frames_len,self.feat_len])
                out_data=np.zeros([training_batch_size,self.frames_len,self.feat_len])
                offsets=offsets[:training_batch_size]
                for i,start_i in enumerate(offsets):
                    out_data[i]=data[start_i:start_i+self.frames_len]
                return out_data,self.segments,label# return 3 infos
            ##TBA new types
    def get_files(self,gt_path):
        with open(gt_path,'rb')as fp:
            lines=fp.readlines()
        file_names=[]
        labels=[]
        for line in lines:
            file_names.append(line.strip().split(' ')[0])
            labels.append(int(line.strip().split(' ')[2]))
        return file_names,labels
    def __len__(self):
        return len(self.file_names)

class LSTM(nn.Module):
    def __init__(self,
                 input_len=1024,
                 hidden_len=1024,
                 model_len=12,
                 target_size=101,
                 n_layers=1,
                 is_softmax=True):
        super(LSTM,self).__init__()
        self.hidden_len=hidden_len
        self.is_softmax=is_softmax
        self.lstm=nn.LSTM(input_size=input_len,hidden_size=hidden_len,num_layers=n_layers,batch_first=True)
        self.target_size=target_size
        self.n_layers=n_layers
        self.hidden2class=nn.Linear(hidden_len,target_size)
        self.hidden=self.init_hidden()
        self.log_softmax=nn.LogSoftmax()
        self.dropout=nn.Dropout(0.5)

    def init_hidden(self,x=None):
        if x==None:
            return (Variable(torch.zeros(self.n_layers, training_batch_size, self.hidden_len)),
                Variable(torch.zeros(self.n_layers, training_batch_size, self.hidden_len)))
        else:
            return (Variable(x[0].data),Variable(x[1].data))

    def forward(self,x):
        lstm_out,self.hidden_out=self.lstm(
            x,self.hidden
        )
        cls_scores=self.hidden2class(self.dropout(lstm_out[:,-1,:])).view(-1,self.target_size)
        if self.is_softmax:
            cls_scores=self.log_softmax(cls_scores)
        self.hidden=self.init_hidden(self.hidden_out)
        return cls_scores

def build_model(model_name,model_parames,is_softmax=True):
    if model_name=='lstm':
        feat_len=model_parames['feat_len']
        model_len=model_parames['model_len']
        hidden_len=model_parames['hidden_len']
        n_layers=model_parames['n_layers']
        model_tmp=LSTM(feat_len,hidden_len,model_len,target_size=101,is_softmax=is_softmax,n_layers=n_layers)
    return model_tmp

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Two-Stream Action Recognition')
    parser.add_argument('--feature_path',type=str)
    parser.add_argument('--model',type=str,default='lstm')
    parser.add_argument('--new_length',type=int,default=12)
    parser.add_argument('--feat_len',type=int,default=1024)
    parser.add_argument('--hidden_len',type=int,default=512)
    parser.add_argument('--split',default=1,type=int)
    parser.add_argument('--modality',default='rgb',type=str)
    parser.add_argument('--workers',default=4,type=int)
    parser.add_argument('--clip_gradient',default=0,type=int)
    parser.add_argument('--n_layers',default=1,type=int)
    args=parser.parse_args()
    return args

def main():
    global args
    args=parse_args()
    log.l.info('Input command: \n{}'.format('python '+' '.join(sys.argv)))
    log.l.info('Input args: \n{}'.format(args))
    model_parames={
        'feat_len':args.feat_len,
        'model_len':args.new_length,
        'hidden_len':args.hidden_len,
        'n_layers':args.n_layers
    }
    best_prec1=0
    is_softmax=True
    global model
    model=build_model(args.model,model_parames,is_softmax=is_softmax)
    if is_softmax:
        loss_function=nn.NLLLoss()
    else:
        loss_function=nn.CrossEntropyLoss()
    optimizer=optim.SGD(model.parameters(),
                        lr=0.01,
                        momentum=0.9,
                        weight_decay=weight_decay)


    gt_file_name='{}_{}_split{}.txt'.format('train',args.modality,args.split)
    train_data_loader=data_extractor(root=args.feature_path,type='train',frames_len=args.new_length,target_size=101,feat_len=1024,gt_file_name=gt_file_name)
    train_data=torch.utils.data.DataLoader(train_data_loader,batch_size=training_batch_size,shuffle=True,num_workers=args.workers,drop_last=True)
    gt_file_name='{}_{}_split{}.txt'.format('val',args.modality,args.split)
    val_data_loader=data_extractor(root=args.feature_path,type='val',frames_len=args.new_length,target_size=101,feat_len=1024,gt_file_name=gt_file_name,test_type=1)
    val_data=torch.utils.data.DataLoader(val_data_loader,batch_size=testing_batch_size,shuffle=False,num_workers=args.workers)

    '''
    learning policy parameters:
    '''
    lr_steps=[3,6]
    starting_epoch=0
    ending_epoch=8
    #testing the validate process first before run the code
    #validate(val_data,loss_function,200)
    for epoch in range(starting_epoch,ending_epoch):
        #adjust the learning params and apply train.
        adjust_learning_rate(optimizer,epoch,lr_steps)
        train(train_data,loss_function,optimizer,epoch)

        if (epoch+1) % val_frequency==0 or epoch==ending_epoch-1:
            prec1=validate(val_data,loss_function,(epoch+1)*len(train_data))
            #remember the best prec@1 and save checkpoint
            is_best=prec1>best_prec1
            best_prec1=max(prec1,best_prec1)
            save_checkpoint({
                'epoch':epoch+1,
                'arch':args.model,
                'state_dict':model.state_dict(),
                'best_prec1':best_prec1,
            },is_best)


def train(train_data,criterion,optimizer,epoch):
    batch_time=AverageMeter()
    data_time=AverageMeter()
    losses=AverageMeter()
    top1=AverageMeter()
    top5=AverageMeter()

    #model.train()

    end=time.time()
    for i,(input,target) in enumerate(train_data):
        #measure the data loading time
        data_time.update(time.time()-end)
        if use_cuda:
            input=input.cuda(async=True)
        input_var=torch.autograd.Variable(input)
        target_var=torch.autograd.Variable(target)

        output=model(input_var)
        loss=criterion(output,target_var)

        prec1,prec5=accuracy(output.data,target,topk=(1,5))
        losses.update(loss.data[0],input.size(0))
        top1.update(prec1[0],input.size(0))
        top5.update(prec5[0],input.size(0))

        optimizer.zero_grad()
        loss.backward()

        if args.clip_gradient!=0 :
            total_norm=clip_grad_norm(model.parameters(),args.clip_gradient)
            if total_norm>args.clip_gradient:
                log.l.info("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))

        optimizer.step()
        batch_time.update(time.time()-end)
        end=time.time()

        if i%print_train_info_frequency==0:
            log.l.info(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_data), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'])))

def validate(val_data,criterion,iter,logger=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    #switch to evaluate mode
    #model.eval()

    end=time.time()
    for i,(input,len_data,target) in enumerate(val_data):
        if use_cuda:
            input=input.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)
        #embed()
        len_data=len_data.int().numpy()[0]
        # fix the input variable to fixed length and compute output
        #input_var=wrapper(input_var)
        output = model(input_var.float().view(training_batch_size,args.new_length,args.feat_len))[:len_data].mean(0).view(testing_batch_size,-1)
        loss = criterion(output, target_var)

        prec1, prec5 = accuracy(output.data, target, topk=(1,5))

        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_val_info_frequency == 0:
            log.l.info(('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_data), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5)))
    log.l.info(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
          .format(top1=top1, top5=top5, loss=losses)))
    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    filename = '_'.join(( args.model,args.modality.lower(), filename))
    torch.save(state, filename)
    if is_best:
        best_name = '_'.join((args.model, args.modality.lower(), 'model_best.pth.tar'))
        shutil.copyfile(filename, best_name)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = lr_init * decay
    decay = weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr# * param_group['lr']
        #param_group['weight_decay'] = decay * param_group['decay_mult']

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__=='__main__':
    main()