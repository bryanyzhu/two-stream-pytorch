#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser(description="Convert a Caffe model and its learned parameters to torch")
parser.add_argument('model', help='network spec, usually a ProtoBuf text message')
parser.add_argument('weights', help='network parameters, usually in a name like *.caffemodel ')
parser.add_argument('--model_yaml', help="translated model spec yaml file")
parser.add_argument('--dump_weights', help="translated model parameters to be used by torch")
parser.add_argument('--model_version', help="the version of Caffe's model spec, usually 2", default=2)

args = parser.parse_args()

from . import caffe_pb2
from google.protobuf import text_format
from pprint import pprint
import yaml
import numpy as np
import torch


class CaffeVendor(object):
    def __init__(self, net_name, weight_name, version=2):
        print("loading model spec...")
        self._net_pb = caffe_pb2.NetParameter()
        text_format.Merge(open(net_name).read(), self._net_pb)
        self._weight_dict = {}
        self._init_dict = []

        if weight_name is not None:
            print("loading weights...")
            self._weight_pb = caffe_pb2.NetParameter()
            self._weight_pb.ParseFromString(open(weight_name, 'rb').read())
            for l in self._weight_pb.layer:
                self._weight_dict[l.name] = l

        print("parsing...")
        self._parse_net(version)

    def _parse_net(self, version):
        self._name = str(self._net_pb.name)
        self._layers = self._net_pb.layer if version == 2 else self._net_pb.layers
        self._parsed_layers = [self._layer2dict(x, version) for x in self._layers]

        self._net_dict = {
            'name': self._name,
            'inputs': [],
            'layers': [],
        }

        self._weight_array_dict = {}

        for info, blob, is_data in self._parsed_layers:
            if not is_data and info is not None:
                self._net_dict['layers'].append(info)

            self._weight_array_dict.update(blob)

    @staticmethod
    def _parse_blob(blob):
        flat_data = np.array(blob.data)
        shaped_data = flat_data.reshape(list(blob.shape.dim))
        return shaped_data

    def _layer2dict(self, layer, version):
        attr_dict = {}
        params = []
        weight_params = []
        fillers = []

        for field, value in layer.ListFields():
            if field.name == 'top':
                tops = [v.replace('-', '_').replace('/', '_') for v in value]
            elif field.name == 'name':
                layer_name = str(value).replace('-', '_').replace('/', '_')
            elif field.name == 'bottom':
                bottoms = [v.replace('-', '_').replace('/', '_') for v in value]
            elif field.name == 'include':
                if value[0].phase == 1 and op == 'Data':
                    print('found 1 testing data layer')
                    return None, dict(), dict(), False
            elif field.name == 'type':
                if version == 2:
                    op = value
                else:
                    raise NotImplemented
            elif field.name == 'loss_weight':
                pass
            elif field.name == 'param':
                pass
            else:
                # other params
                try:
                    for f, v in value.ListFields():
                        if 'filler' in f.name:
                            pass
                        elif f.name == 'pool':
                          attr_dict['mode'] = 'max' if v == 0 else 'ave'
                        else:
                          attr_dict[f.name] = v

                except:
                    print(field.name, value)
                    raise

        expr_temp = '{top}<={op}<={input}'

        if layer.name in self._weight_dict:
            blobs = [self._parse_blob(x) for x in self._weight_dict[layer.name].blobs]
        else:
            blobs = []

        blob_dict = dict()
        if len(blobs) > 0:
            blob_dict['{}.weight'.format(layer_name)] = torch.from_numpy(blobs[0])
            blob_dict['{}.bias'.format(layer_name)] = torch.from_numpy(blobs[1])
            if op == 'BN':
                blob_dict['{}.running_mean'.format(layer_name)] = torch.from_numpy(blobs[2])
                blob_dict['{}.running_var'.format(layer_name)] = torch.from_numpy(blobs[3])

        expr = expr_temp.format(top=','.join(tops), input=','.join(bottoms), op=op)

        out_dict = {
            'id': layer_name,
            'expr': expr,
        }

        if len(attr_dict) > 0:
            out_dict['attrs'] = attr_dict

        return out_dict, blob_dict, False

    @property
    def text_form(self):
        return str(self._net_pb)

    @property
    def info(self):
        return {
            'name': self._name,
            'layers': [x.name for x in self._layers]
        }

    @property
    def yaml(self):
        return yaml.dump(self._net_dict)

    def dump_weights(self, filename):
        # print self._weight_array_dict.keys()
        torch.save(self._weight_array_dict, open(filename, 'wb'))

# build output
cv = CaffeVendor(args.model, args.weights, int(args.model_version))

if args.model_yaml is not None:
    open(args.model_yaml, 'w').write(cv.yaml)

if args.dump_weights is not None:
    cv.dump_weights(args.dump_weights)
