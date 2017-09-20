import torch
from torch import nn


LAYER_BUILDER_DICT=dict()


def parse_expr(expr):
    parts = expr.split('<=')
    return parts[0].split(','), parts[1], parts[2].split(',')


def get_basic_layer(info, channels=None, conv_bias=False):
    id = info['id']
    attr = info['attrs'] if 'attrs' in info else list()

    out, op, in_vars = parse_expr(info['expr'])
    assert(len(out) == 1)
    assert(len(in_vars) == 1)
    mod, out_channel, = LAYER_BUILDER_DICT[op](attr, channels, conv_bias)

    return id, out[0], mod, out_channel, in_vars[0]


def build_conv(attr, channels=None, conv_bias=False):
    out_channels = attr['num_output']
    ks = attr['kernel_size'] if 'kernel_size' in attr else (attr['kernel_w'], attr['kernel_h'])
    padding = attr['pad'] if 'pad' in attr else 0
    stride = attr['stride'] if 'stride' in attr else 1

    conv = nn.Conv2d(channels, out_channels, ks, stride, padding, bias=conv_bias)

    return conv, out_channels


def build_pooling(attr, channels=None, conv_bias=False):
    method = attr['mode']
    pad = attr['pad'] if 'pad' in attr else 0
    if method == 'max':
        pool = nn.MaxPool2d(attr['kernel_size'], attr['stride'], pad,
                            ceil_mode=True) # all Caffe pooling use ceil model
    elif method == 'ave':
        pool = nn.AvgPool2d(attr['kernel_size'], attr['stride'], pad,
                            ceil_mode=True)  # all Caffe pooling use ceil model
    else:
        raise ValueError("Unknown pooling method: {}".format(method))

    return pool, channels


def build_relu(attr, channels=None, conv_bias=False):
    return nn.ReLU(inplace=True), channels


def build_bn(attr, channels=None, conv_bias=False):
    return nn.BatchNorm2d(channels, momentum=0.1), channels


def build_linear(attr, channels=None, conv_bias=False):
    return nn.Linear(channels, attr['num_output']), channels


def build_dropout(attr, channels=None, conv_bias=False):
    return nn.Dropout(p=attr['dropout_ratio']), channels


LAYER_BUILDER_DICT['Convolution'] = build_conv

LAYER_BUILDER_DICT['Pooling'] = build_pooling

LAYER_BUILDER_DICT['ReLU'] = build_relu

LAYER_BUILDER_DICT['Dropout'] = build_dropout

LAYER_BUILDER_DICT['BN'] = build_bn

LAYER_BUILDER_DICT['InnerProduct'] = build_linear

