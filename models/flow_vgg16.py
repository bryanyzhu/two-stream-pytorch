import torch.nn as nn
import torch.utils.model_zoo as model_zoo

import math
import collections
import numpy as np
import torch

__all__ = ['VGG', 'flow_vgg16']


model_urls = {'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth'}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.9),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.8),
        )
        # Change the dropout value to 0.9 and 0.8 for flow model
        self.fc_action = nn.Linear(4096, num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.fc_action(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 20
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def change_key_names(old_params, in_channels):
    new_params = collections.OrderedDict()
    layer_count = 0
    for layer_key in old_params.keys():
        if layer_count < 26:
            if layer_count == 0:
                rgb_weight = old_params[layer_key]
                rgb_weight_mean = torch.mean(rgb_weight, dim=1)
                flow_weight = rgb_weight_mean.repeat(1,in_channels,1,1)
                new_params[layer_key] = flow_weight
                layer_count += 1
                # print(layer_key, new_params[layer_key].size())
            else:
                new_params[layer_key] = old_params[layer_key]
                layer_count += 1
                # print(layer_key, new_params[layer_key].size())

    return new_params

def flow_vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D']), **kwargs)
    # TODO: hardcoded for now for 10 optical flow images, set it as an argument later 
    in_channels = 20            
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
        pretrained_dict = model_zoo.load_url(model_urls['vgg16'])
        model_dict = model.state_dict()

        new_pretrained_dict = change_key_names(pretrained_dict, in_channels)
        # 1. filter out unnecessary keys
        new_pretrained_dict = {k: v for k, v in new_pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(new_pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)

    return model


