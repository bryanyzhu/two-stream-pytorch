require 'nn'
local hdf5 = require 'hdf5'
torch.setdefaulttensortype('torch.FloatTensor')
require 'image'

local function SpatialConvBatchNormReLU(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
   local std_epsilon = 0.001
   local conv = nn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
   conv:noBias()
   local module = nn.Sequential()
   module:add(conv)
   module:add(nn.SpatialBatchNormalization(nOutputPlane, std_epsilon, nil, true))
   module:add(nn.ReLU(true))
   return module
end

local function SpatialAveragePoolingNoCIP(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
   local module = nn.SpatialAveragePooling(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
   module.count_include_pad = false
   return module
end

local function Tower(layers)
   local tower = nn.Sequential()
   for i=1,#layers do
      tower:add(layers[i])
   end
   return tower
end

local function FilterConcat(towers)
   local concat = nn.DepthConcat(2)
   for i=1,#towers do
      concat:add(towers[i])
   end
   return concat
end

local function Mixed_5b()
   local module = FilterConcat({
      SpatialConvBatchNormReLU(192, 96, 1, 1, 1, 1),
      Tower({
         SpatialConvBatchNormReLU(192, 48, 1, 1, 1, 1),
         SpatialConvBatchNormReLU(48, 64, 5, 5, 1, 1, 2, 2)
      }),
      Tower({
         SpatialConvBatchNormReLU(192, 64, 1, 1, 1, 1),
         SpatialConvBatchNormReLU(64, 96, 3, 3, 1, 1, 1, 1),
         SpatialConvBatchNormReLU(96, 96, 3, 3, 1, 1, 1, 1)
      }),
      Tower({
         SpatialAveragePoolingNoCIP(3, 3, 1, 1, 1, 1),
         SpatialConvBatchNormReLU(192, 64, 1, 1, 1, 1)
      })
   })
   return module
end

local function Block35(scale)
   local scale = scale or 0.17

   local branchs = FilterConcat({
      SpatialConvBatchNormReLU(320, 32, 1, 1, 1, 1),
      Tower({
         SpatialConvBatchNormReLU(320, 32, 1, 1, 1, 1),
         SpatialConvBatchNormReLU(32, 32, 3, 3, 1, 1, 1, 1)
      }),
      Tower({
         SpatialConvBatchNormReLU(320, 32, 1, 1, 1, 1),
         SpatialConvBatchNormReLU(32, 48, 3, 3, 1, 1, 1, 1),
         SpatialConvBatchNormReLU(48, 64, 3, 3, 1, 1, 1, 1)
      })
   })

   local shortcut = nn.ConcatTable(2)
   shortcut:add(nn.Identity())
   shortcut:add(
      Tower({
         branchs,
         nn.SpatialConvolution(128, 320, 1, 1, 1, 1),
         nn.MulConstant(scale)
      })
   )

   local module = nn.Sequential()
   module:add(shortcut)
   module:add(nn.CAddTable(true))
   module:add(nn.ReLU(true))
   return module
end

local function Mixed_6a()
   local module = FilterConcat({
      SpatialConvBatchNormReLU(320, 384, 3, 3, 2, 2),
      Tower({
         SpatialConvBatchNormReLU(320, 256, 1, 1, 1, 1),
         SpatialConvBatchNormReLU(256, 256, 3, 3, 1, 1, 1, 1),
         SpatialConvBatchNormReLU(256, 384, 3, 3, 2, 2)
      }),
      nn.SpatialMaxPooling(3, 3, 2, 2)
   })
   return module
end

local function Block17(scale)
   local scale = scale or 0.10

   local branchs = FilterConcat({
      SpatialConvBatchNormReLU(1088, 192, 1, 1, 1, 1),
      Tower({
         SpatialConvBatchNormReLU(1088, 128, 1, 1, 1, 1),
         SpatialConvBatchNormReLU(128, 160, 7, 1, 1, 1, 3, 0),
         SpatialConvBatchNormReLU(160, 192, 1, 7, 1, 1, 0, 3)
      })
   })

   local shortcut = nn.ConcatTable(2)
   shortcut:add(nn.Identity())
   shortcut:add(
      Tower({
         branchs,
         nn.SpatialConvolution(384, 1088, 1, 1, 1, 1),
         nn.MulConstant(scale)
      })
   )

   local module = nn.Sequential()
   module:add(shortcut)
   module:add(nn.CAddTable(true))
   module:add(nn.ReLU(true))
   return module
end

local function Mixed_7a()
   local module = FilterConcat({
      Tower({
         SpatialConvBatchNormReLU(1088, 256, 1, 1, 1, 1),
         SpatialConvBatchNormReLU(256, 384, 3, 3, 2, 2)
      }),
      Tower({
         SpatialConvBatchNormReLU(1088, 256, 1, 1, 1, 1),
         SpatialConvBatchNormReLU(256, 288, 3, 3, 2, 2)
      }),
      Tower({
         SpatialConvBatchNormReLU(1088, 256, 1, 1, 1, 1),
         SpatialConvBatchNormReLU(256, 288, 3, 3, 1, 1, 1, 1),
         SpatialConvBatchNormReLU(288, 320, 3, 3, 2, 2)
      }),
      nn.SpatialMaxPooling(3, 3, 2, 2)
   })
   return module
end

local function Block8(scale, noReLU)
   local scale = scale or 0.20

   local branchs = FilterConcat({
      SpatialConvBatchNormReLU(2080, 192, 1, 1, 1, 1),
      Tower({
         SpatialConvBatchNormReLU(2080, 192, 1, 1, 1, 1),
         SpatialConvBatchNormReLU(192, 224, 3, 1, 1, 1, 1, 0),
         SpatialConvBatchNormReLU(224, 256, 1, 3, 1, 1, 0, 1)
      })
   })

   local shortcut = nn.ConcatTable(2)
   shortcut:add(nn.Identity())
   shortcut:add(
      Tower({
         branchs,
         nn.SpatialConvolution(448, 2080, 1, 1, 1, 1),
         nn.MulConstant(scale)
      })
   )

   local module = nn.Sequential()
   module:add(shortcut)
   module:add(nn.CAddTable(true))
   if not noReLU then
      module:add(nn.ReLU(true))
   end
   return module
end

local function InceptionResnetV2(nclass)
   local nclass = nclass or 1001
   local net = nn.Sequential()
   net:add(SpatialConvBatchNormReLU(3, 32, 3, 3, 2, 2))
   net:add(SpatialConvBatchNormReLU(32, 32, 3, 3, 1, 1))
   net:add(SpatialConvBatchNormReLU(32, 64, 3, 3, 1, 1, 1, 1))
   net:add(nn.SpatialMaxPooling(3, 3, 2, 2))
   net:add(SpatialConvBatchNormReLU(64, 80, 1, 1, 1, 1))
   net:add(SpatialConvBatchNormReLU(80, 192, 3, 3, 1, 1))
   net:add(nn.SpatialMaxPooling(3, 3, 2, 2))
   net:add(Mixed_5b())
   net:add(Tower({
      Block35(),
      Block35(),
      Block35(),
      Block35(),
      Block35(),
      Block35(),
      Block35(),
      Block35(),
      Block35(),
      Block35() -- 10th
   }))
   net:add(Mixed_6a())
   net:add(Tower({
      Block17(),
      Block17(),
      Block17(),
      Block17(),
      Block17(),
      Block17(),
      Block17(),
      Block17(),
      Block17(),
      Block17(),
      Block17(),
      Block17(),
      Block17(),
      Block17(),
      Block17(),
      Block17(),
      Block17(),
      Block17(),
      Block17(),
      Block17() -- 20th
   }))
   net:add(Mixed_7a())
   net:add(Tower({
      Block8(),
      Block8(),
      Block8(),
      Block8(),
      Block8(),
      Block8(),
      Block8(),
      Block8(),
      Block8() -- 9th
   }))
   net:add(Block8(1.0, true))
   net:add(SpatialConvBatchNormReLU(2080, 1536, 1, 1, 1, 1))
   net:add(nn.SpatialAveragePooling(8, 8))
   net:add(nn.View(1536))
   net:add(nn.Linear(1536, nclass))
   return net
end

----------------
-- Load --
----------------

local function load_conv2d(module, name)
   -- local name = name or 'Conv2d_1a_3x3'
   local h5f = hdf5.open('dump/InceptionResnetV2/'..name..'.h5', 'r')
  
   local conv = module:get(1) -- Spatial Convolution
   local weights = h5f:read("weights"):all():permute(4, 3, 1, 2)
   conv.weight:copy(weights)

   local bn = module:get(2) -- Spatial Batch Normalization
   --local gamma = h5f:read("gamma"):all() 
   bn.weight:copy(torch.ones(bn.weight:size(1))) -- gamma is set to 1
   local beta = h5f:read("beta"):all()
   bn.bias:copy(beta)
   local mean = h5f:read("mean"):all()
   bn.running_mean:copy(mean)
   local var = h5f:read("var"):all()
   bn.running_var:copy(var)

   h5f:close()
end

local function load_conv2d_nobn(module, name)
   local name = name or 'Conv2d_1a_3x3'
   local h5f = hdf5.open('dump/InceptionResnetV2/'..name..'.h5', 'r')
   local conv = module -- Spatial Convolution

   local weights = h5f:read("weights"):all():permute(4, 3, 1, 2)
   conv.weight:copy(weights)

   local biases = h5f:read("biases"):all()
   conv.bias:copy(biases)

   h5f:close()
end

local function load_linear(module, name)
  local h5f = hdf5.open('dump/InceptionResnetV2/'..name..'.h5', 'r')
  local weights = h5f:read('weights'):all():t()
  local biases = h5f:read('biases'):all()
  module.weight:copy(weights)
  module.bias:copy(biases)
  h5f:close()
end

local function load_mixed_5b(module)
   load_conv2d(module:get(1), 'Mixed_5b/Branch_0/Conv2d_1x1')
   load_conv2d(module:get(2):get(1), 'Mixed_5b/Branch_1/Conv2d_0a_1x1')
   load_conv2d(module:get(2):get(2), 'Mixed_5b/Branch_1/Conv2d_0b_5x5')
   load_conv2d(module:get(3):get(1), 'Mixed_5b/Branch_2/Conv2d_0a_1x1')
   load_conv2d(module:get(3):get(2), 'Mixed_5b/Branch_2/Conv2d_0b_3x3')
   load_conv2d(module:get(3):get(3), 'Mixed_5b/Branch_2/Conv2d_0c_3x3')
   load_conv2d(module:get(4):get(2), 'Mixed_5b/Branch_3/Conv2d_0b_1x1')
end

local function load_block35(module, name)
   load_conv2d(module:get(1):get(2):get(1):get(1), name..'/Branch_0/Conv2d_1x1')
   load_conv2d(module:get(1):get(2):get(1):get(2):get(1), name..'/Branch_1/Conv2d_0a_1x1')
   load_conv2d(module:get(1):get(2):get(1):get(2):get(2), name..'/Branch_1/Conv2d_0b_3x3')
   load_conv2d(module:get(1):get(2):get(1):get(3):get(1), name..'/Branch_2/Conv2d_0a_1x1')
   load_conv2d(module:get(1):get(2):get(1):get(3):get(2), name..'/Branch_2/Conv2d_0b_3x3')
   load_conv2d(module:get(1):get(2):get(1):get(3):get(3), name..'/Branch_2/Conv2d_0c_3x3')
   load_conv2d_nobn(module:get(1):get(2):get(2), name..'/Conv2d_1x1')
end

local function load_mixed_6a(module)
   load_conv2d(module:get(1), 'Mixed_6a/Branch_0/Conv2d_1a_3x3')
   load_conv2d(module:get(2):get(1), 'Mixed_6a/Branch_1/Conv2d_0a_1x1')
   load_conv2d(module:get(2):get(2), 'Mixed_6a/Branch_1/Conv2d_0b_3x3')
   load_conv2d(module:get(2):get(3), 'Mixed_6a/Branch_1/Conv2d_1a_3x3')
end

local function load_block17(module, name)
   load_conv2d(module:get(1):get(2):get(1):get(1), name..'/Branch_0/Conv2d_1x1')
   load_conv2d(module:get(1):get(2):get(1):get(2):get(1), name..'/Branch_1/Conv2d_0a_1x1')
   load_conv2d(module:get(1):get(2):get(1):get(2):get(2), name..'/Branch_1/Conv2d_0b_1x7')
   load_conv2d(module:get(1):get(2):get(1):get(2):get(3), name..'/Branch_1/Conv2d_0c_7x1')
   load_conv2d_nobn(module:get(1):get(2):get(2), name..'/Conv2d_1x1')
end

local function load_mixed_7a(module)
   load_conv2d(module:get(1):get(1), 'Mixed_7a/Branch_0/Conv2d_0a_1x1')
   load_conv2d(module:get(1):get(2), 'Mixed_7a/Branch_0/Conv2d_1a_3x3')
   load_conv2d(module:get(2):get(1), 'Mixed_7a/Branch_1/Conv2d_0a_1x1')
   load_conv2d(module:get(2):get(2), 'Mixed_7a/Branch_1/Conv2d_1a_3x3')
   load_conv2d(module:get(3):get(1), 'Mixed_7a/Branch_2/Conv2d_0a_1x1')
   load_conv2d(module:get(3):get(2), 'Mixed_7a/Branch_2/Conv2d_0b_3x3')
   load_conv2d(module:get(3):get(3), 'Mixed_7a/Branch_2/Conv2d_1a_3x3')
end

local function load_block8(module, name)
   load_conv2d(module:get(1):get(2):get(1):get(1), name..'/Branch_0/Conv2d_1x1')
   load_conv2d(module:get(1):get(2):get(1):get(2):get(1), name..'/Branch_1/Conv2d_0a_1x1')
   load_conv2d(module:get(1):get(2):get(1):get(2):get(2), name..'/Branch_1/Conv2d_0b_1x3')
   load_conv2d(module:get(1):get(2):get(1):get(2):get(3), name..'/Branch_1/Conv2d_0c_3x1')
   load_conv2d_nobn(module:get(1):get(2):get(2), name..'/Conv2d_1x1')
end

local function load(net)
   load_conv2d(net:get(1), 'Conv2d_1a_3x3')
   load_conv2d(net:get(2), 'Conv2d_2a_3x3')
   load_conv2d(net:get(3), 'Conv2d_2b_3x3')

   load_conv2d(net:get(5), 'Conv2d_3b_1x1')
   load_conv2d(net:get(6), 'Conv2d_4a_3x3')

   load_mixed_5b(net:get(8))

   for i=1, 10 do
      load_block35(net:get(9):get(i), 'Repeat/block35_'..i)
   end

   load_mixed_6a(net:get(10))

   for i=1, 20 do
      load_block17(net:get(11):get(i), 'Repeat_1/block17_'..i)
   end

   load_mixed_7a(net:get(12))

   for i=1, 9 do
      load_block8(net:get(13):get(i), 'Repeat_2/block8_'..i)
   end

   load_block8(net:get(14), 'Block8')
   load_conv2d(net:get(15), 'Conv2d_7b_1x1')
   load_linear(net:get(18), 'Logits')
end

----------
-- Test --
----------

local function test_conv2d(module, name, opt)
   local name = name or 'Conv2d_1a_3x3'
   local h5f = hdf5.open('dump/InceptionResnetV2/'..name..'.h5', 'r')

   local conv_out = h5f:read("conv_out"):all()
   conv_out = conv_out:transpose(2,4)
   conv_out = conv_out:transpose(3,4)
 
   local relu_out = h5f:read("relu_out"):all()
   relu_out = relu_out:transpose(2,4)
   relu_out = relu_out:transpose(3,4)
 
   h5f:close()
 
   if opt.cuda then
     conv_out = conv_out:cuda()
     relu_out = relu_out:cuda()
   end
 
   print(name..' conv_out', torch.dist(module:get(1).output, conv_out))
   print(name..' relu_out', torch.dist(module:get(3).output, relu_out))
   print('')
end

local function test_conv2d_nobn(module, name, opt)
   local name = name or 'Conv2d_1a_3x3'
   local h5f = hdf5.open('dump/InceptionResnetV2/'..name..'.h5', 'r')

   local conv_out = h5f:read("conv_out"):all()
   conv_out = conv_out:transpose(2,4)
   conv_out = conv_out:transpose(3,4)
 
   h5f:close()
 
   if opt.cuda then
     conv_out = conv_out:cuda()
   end
 
   print(name..' conv_out', torch.dist(module.output, conv_out))
   print('')
end

local function test_linear(module, name, opt)
   local h5f = hdf5.open('dump/InceptionResnetV2/'..name..'.h5', 'r')
   local out = h5f:read("out"):all()
   h5f:close()
   local softmax = nn.SoftMax()
   if opt.cuda then
      softmax:cuda()
      out = out:cuda()
   end
   local output = softmax:forward(module.output)
   print(name..' linear_out', torch.dist(output, out))
   print('')
end

local function test_mixed_5b(module, opt)
   test_conv2d(module:get(1), 'Mixed_5b/Branch_0/Conv2d_1x1', opt)
   test_conv2d(module:get(2):get(1), 'Mixed_5b/Branch_1/Conv2d_0a_1x1', opt)
   test_conv2d(module:get(2):get(2), 'Mixed_5b/Branch_1/Conv2d_0b_5x5', opt)
   test_conv2d(module:get(3):get(1), 'Mixed_5b/Branch_2/Conv2d_0a_1x1', opt)
   test_conv2d(module:get(3):get(2), 'Mixed_5b/Branch_2/Conv2d_0b_3x3', opt)
   test_conv2d(module:get(3):get(3), 'Mixed_5b/Branch_2/Conv2d_0c_3x3', opt)
   test_conv2d(module:get(4):get(2), 'Mixed_5b/Branch_3/Conv2d_0b_1x1', opt)
end

local function test_block35(module, name, opt)
   test_conv2d(module:get(1):get(2):get(1):get(1), name..'/Branch_0/Conv2d_1x1', opt)
   test_conv2d(module:get(1):get(2):get(1):get(2):get(1), name..'/Branch_1/Conv2d_0a_1x1', opt)
   test_conv2d(module:get(1):get(2):get(1):get(2):get(2), name..'/Branch_1/Conv2d_0b_3x3', opt)
   test_conv2d(module:get(1):get(2):get(1):get(3):get(1), name..'/Branch_2/Conv2d_0a_1x1', opt)
   test_conv2d(module:get(1):get(2):get(1):get(3):get(2), name..'/Branch_2/Conv2d_0b_3x3', opt)
   test_conv2d(module:get(1):get(2):get(1):get(3):get(3), name..'/Branch_2/Conv2d_0c_3x3', opt)
   test_conv2d_nobn(module:get(1):get(2):get(2), name..'/Conv2d_1x1', opt)
end

local function test_mixed_6a(module, opt)
   test_conv2d(module:get(1), 'Mixed_6a/Branch_0/Conv2d_1a_3x3', opt)
   test_conv2d(module:get(2):get(1), 'Mixed_6a/Branch_1/Conv2d_0a_1x1', opt)
   test_conv2d(module:get(2):get(2), 'Mixed_6a/Branch_1/Conv2d_0b_3x3', opt)
   test_conv2d(module:get(2):get(3), 'Mixed_6a/Branch_1/Conv2d_1a_3x3', opt)
end

local function test_block17(module, name, opt)
   test_conv2d(module:get(1):get(2):get(1):get(1), name..'/Branch_0/Conv2d_1x1', opt)
   test_conv2d(module:get(1):get(2):get(1):get(2):get(1), name..'/Branch_1/Conv2d_0a_1x1', opt)
   test_conv2d(module:get(1):get(2):get(1):get(2):get(2), name..'/Branch_1/Conv2d_0b_1x7', opt)
   test_conv2d(module:get(1):get(2):get(1):get(2):get(3), name..'/Branch_1/Conv2d_0c_7x1', opt)
   test_conv2d_nobn(module:get(1):get(2):get(2), name..'/Conv2d_1x1', opt)
end

local function test_mixed_7a(module, opt)
   test_conv2d(module:get(1):get(1), 'Mixed_7a/Branch_0/Conv2d_0a_1x1', opt)
   test_conv2d(module:get(1):get(2), 'Mixed_7a/Branch_0/Conv2d_1a_3x3', opt)
   test_conv2d(module:get(2):get(1), 'Mixed_7a/Branch_1/Conv2d_0a_1x1', opt)
   test_conv2d(module:get(2):get(2), 'Mixed_7a/Branch_1/Conv2d_1a_3x3', opt)
   test_conv2d(module:get(3):get(1), 'Mixed_7a/Branch_2/Conv2d_0a_1x1', opt)
   test_conv2d(module:get(3):get(2), 'Mixed_7a/Branch_2/Conv2d_0b_3x3', opt)
   test_conv2d(module:get(3):get(3), 'Mixed_7a/Branch_2/Conv2d_1a_3x3', opt)
end

local function test_block8(module, name)
   load_conv2d(module:get(1):get(2):get(1):get(1), name..'/Branch_0/Conv2d_1x1')
   load_conv2d(module:get(1):get(2):get(1):get(2):get(1), name..'/Branch_1/Conv2d_0a_1x1')
   load_conv2d(module:get(1):get(2):get(1):get(2):get(2), name..'/Branch_1/Conv2d_0b_1x3')
   load_conv2d(module:get(1):get(2):get(1):get(2):get(3), name..'/Branch_1/Conv2d_0c_3x1')
   load_conv2d_nobn(module:get(1):get(2):get(2), name..'/Conv2d_1x1')
end

local function test(net, opt)
   net:evaluate()
   local input = torch.ones(1,299,299,3) -- [0,1]
   --local img = image.load('lena_299.png') * 255.0 -- [0,255]
   --input[1] = img:float()
   input[{1,1,1,1}] = -1
   input = input:transpose(2,4)
   input = input:transpose(3,4)
   local softmax = nn.SoftMax()
   if opt.cuda then
      input = input:cuda()
      softmax:cuda()
   end
   local output = net:forward(input)

   test_conv2d(net:get(1), 'Conv2d_1a_3x3', opt)
   test_conv2d(net:get(2), 'Conv2d_2a_3x3', opt)
   test_conv2d(net:get(3), 'Conv2d_2b_3x3', opt)

   test_conv2d(net:get(5), 'Conv2d_3b_1x1', opt)
   test_conv2d(net:get(6), 'Conv2d_4a_3x3', opt)

   test_mixed_5b(net:get(8), opt)

   for i=1, 10 do
      test_block35(net:get(9):get(i), 'Repeat/block35_'..i, opt)
   end

   test_mixed_6a(net:get(10), opt)

   for i=1, 20 do
      test_block17(net:get(11):get(i), 'Repeat_1/block17_'..i, opt)
   end

   test_mixed_7a(net:get(12), opt)

   for i=1, 9 do
      test_block8(net:get(13):get(i), 'Repeat_2/block8_'..i, opt)
   end

   test_block8(net:get(14), 'Block8', opt)
   test_conv2d(net:get(15), 'Conv2d_7b_1x1', opt)
   test_linear(net:get(18), 'Logits', opt)
end

------------------------------------------------------------------
-- Main
------------------------------------------------------------------

local function main()
   local opt = {
      cuda = true
   }
   local net = InceptionResnetV2()
   print(net)
   load(net)
   print('loaded')

   if opt.cuda then
      require 'cunn'
      require 'cutorch'
      --require 'cudnn'
      net:cuda()
   end

   test(net, opt)

   os.execute('mkdir -p save')
   torch.save('save/inceptionresnetv2.t7', net:clearState():float())
end

main()