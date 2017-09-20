# Tensorflow Model Zoo for Torch7 and PyTorch

This is a porting of tensorflow pretrained models made by [Remi Cadene](http://remicadene.com) and [Micael Carvalho](http://micaelcarvalho.com). Special thanks to Moustapha Cissé. All models have been tested on Imagenet.

This work was inspired by [inception-v3.torch](https://github.com/Moodstocks/inception-v3.torch).


## Using pretrained models

### Torch7

#### Requirements 

Please install [torchnet-vision](https://github.com/Cadene/torchnet-vision).
```
luarocks install --server=http://luarocks.org/dev torchnet-vision
```

Models available:

- inceptionv3
- inceptionv4
- inceptionresnetv2
- resnet{18, 34, 50, 101, 152, 200}
- overfeat
- vggm
- vgg16

#### Simple example

```lua
require 'image'
tnt = require 'torchnet'
vision = require 'torchnet-vision'
model = vision.models.inceptionresnetv2
net = model.load()

augmentation = tnt.transform.compose{
   vision.image.transformimage.randomScale{
   	minSize = 299, maxSize = 350
   },
   vision.image.transformimage.randomCrop(299),
   vision.image.transformimage.colorNormalize{
      mean = model.mean, std  = model.std
   },
   function(img) return img:float() end
}

net:evaluate()
output = net:forward(augmentation(image.lena()))
```

### PyTorch

Currently available in this repo only On pytorch/vision maybe!

Models available:

- inceptionv4
- inceptionresnetv2

#### Simple example

```python
import torch
from inceptionv4.pytorch_load import inceptionv4
net = inceptionv4()
input = torch.autograd.Variable(torch.ones(1,3,299,299))
output = net.forward(input)
```


## Reproducing the porting

### Requirements
 
- Tensorflow
- Torch7
- PyTorch
- hdf5 for python3
- hdf5 for lua

### Example of commands

In Tensorflow: Download tensorflow parameters and extract them in `./dump` directory.
```
python3 inceptionv4/tensorflow_dump.py
```

In Torch7 or PyTorch: Create the network, load the parameters, launch few tests and save the network in `./save` directory.
```
th inceptionv4/torch_load.lua
python3 inceptionv4/pytorch_load.py
```
