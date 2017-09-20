require 'nn'
local hdf5 = require 'hdf5'
torch.setdefaulttensortype('torch.FloatTensor')
require 'image'

local function SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
  local std_epsilon = 0.001
  local m = nn.Sequential()
  m:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH))
  m:add(nn.SpatialBatchNormalization(nOutputPlane, std_epsilon, nil, true))
  m:add(nn.ReLU())
  return m
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

local function Stem()
  local stem = nn.Sequential()
  stem:add(SpatialConvolution(3, 32, 3, 3, 2, 2)) -- 32x149x149
  stem:add(SpatialConvolution(32, 32, 3, 3, 1, 1)) -- 32x147x147
  stem:add(SpatialConvolution(32, 64, 3, 3, 1, 1, 1, 1)) -- 64x147x147
  stem:add(FilterConcat(
    {
      nn.SpatialMaxPooling(3, 3, 2, 2), -- 64x73x73
      SpatialConvolution(64, 96, 3, 3, 2, 2) -- 96x73x73
    }
  )) -- 160x73x73
  stem:add(FilterConcat(
    {
      Tower(
        {
          SpatialConvolution(160, 64, 1, 1, 1, 1), -- 64x73x73
          SpatialConvolution(64, 96, 3, 3, 1, 1) -- 96x71x71
        }
      ),
      Tower(
        {
          SpatialConvolution(160, 64, 1, 1, 1, 1), -- 64x73x73
          SpatialConvolution(64, 64, 7, 1, 1, 1, 3, 0), -- 64x73x73
          SpatialConvolution(64, 64, 1, 7, 1, 1, 0, 3), -- 64x73x73
          SpatialConvolution(64, 96, 3, 3, 1, 1) -- 96x71x71
        }
      )
    }
  )) -- 192x71x71
  stem:add(FilterConcat(
    {
      SpatialConvolution(192, 192, 3, 3, 2, 2), -- 192x35x35
      nn.SpatialMaxPooling(3, 3, 2, 2) -- 192x35x35
    }
  )) -- 384x35x35
  return stem
end

local function Inception_A()
  local avgpool = nn.SpatialAveragePooling(3, 3, 1, 1, 1, 1)
  avgpool.count_include_pad = false
  local inception = FilterConcat(
    {
      SpatialConvolution(384, 96, 1, 1, 1, 1), -- 96x35x35
      Tower(
        {
          SpatialConvolution(384, 64, 1, 1, 1, 1), -- 64x35x35
          SpatialConvolution(64, 96, 3, 3, 1, 1, 1, 1) -- 96x35x35
        }
      ),
      Tower(
        {
          SpatialConvolution(384, 64, 1, 1, 1, 1), -- 64x35x35
          SpatialConvolution(64, 96, 3, 3, 1, 1, 1, 1), -- 96x35x35
          SpatialConvolution(96, 96, 3, 3, 1, 1, 1, 1), -- 96x35x35
        }
      ),
      Tower(
        {
          avgpool, -- 384x35x35
          SpatialConvolution(384, 96, 1, 1, 1, 1) -- 96x35x35
        }
      )
    }
  ) -- 384x35x35
  -- 384 ifms / ofms
  return inception
end

local function Reduction_A()
  local inception = FilterConcat(
    {
      SpatialConvolution(384, 384, 3, 3, 2, 2), -- 384x17x17
      Tower(
        {
          SpatialConvolution(384, 192, 1, 1, 1, 1), -- 192x35x35
          SpatialConvolution(192, 224, 3, 3, 1, 1, 1, 1), -- 224x35x35
          SpatialConvolution(224, 256, 3, 3, 2, 2), -- 256x17x17
        }
      ),
      nn.SpatialMaxPooling(3, 3, 2, 2) -- 384x17x17
    }
  ) -- 1024x17x17
  -- 384 ifms, 1024 ofms
  return inception
end

local function Inception_B()
  local avgpool = nn.SpatialAveragePooling(3, 3, 1, 1, 1, 1)
  avgpool.count_include_pad = false
  local inception = FilterConcat(
    {
      SpatialConvolution(1024, 384, 1, 1, 1, 1), -- 384x17x17
      Tower(
        {
          SpatialConvolution(1024, 192, 1, 1, 1, 1), -- 192x17x17
          SpatialConvolution(192, 224, 7, 1, 1, 1, 3, 0), -- 224x17x17
          SpatialConvolution(224, 256, 1, 7, 1, 1, 0, 3) -- 256x17x17
        }
      ),
      Tower(
        {
          SpatialConvolution(1024, 192, 1, 1, 1, 1), -- 192x17x17
          SpatialConvolution(192, 192, 1, 7, 1, 1, 0, 3), -- 192x17x17
          SpatialConvolution(192, 224, 7, 1, 1, 1, 3, 0), -- 224x17x17
          SpatialConvolution(224, 224, 1, 7, 1, 1, 0, 3), -- 224x17x17
          SpatialConvolution(224, 256, 7, 1, 1, 1, 3, 0), -- 256x17x17
        }
      ),
      Tower(
        {
          avgpool, -- 1024x17x17
          SpatialConvolution(1024, 128, 1, 1, 1, 1) -- 128x17x17
        }
      )
    }
  ) -- 1024x17x17
  -- 1024 ifms / ofms
  return inception
end

local function Reduction_B()
  local inception = FilterConcat(
    {
      Tower(
        {
          SpatialConvolution(1024, 192, 1, 1, 1, 1), -- 192x17x17
          SpatialConvolution(192, 192, 3, 3, 2, 2) -- 192x8x8
        }
      ),
      Tower(
        {
          SpatialConvolution(1024, 256, 1, 1, 1, 1), -- 256x17x17
          SpatialConvolution(256, 256, 7, 1, 1, 1, 3, 0), -- 256x17x17
          SpatialConvolution(256, 320, 1, 7, 1, 1, 0, 3), -- 320x17x17
          SpatialConvolution(320, 320, 3, 3, 2, 2) -- 320x8x8
        }
      ),
      nn.SpatialMaxPooling(3, 3, 2, 2) -- 1024x8x8
    }
  ) -- 1536x8x8
  -- 1024 ifms, 1536 ofms
  return inception
end

local function Inception_C()
  local avgpool = nn.SpatialAveragePooling(3, 3, 1, 1, 1, 1)
  avgpool.count_include_pad = false
  local inception = FilterConcat(
    {
      SpatialConvolution(1536, 256, 1, 1, 1, 1), -- 256x8x8
      Tower(
        {
          SpatialConvolution(1536, 384, 1, 1, 1, 1), -- 384x8x8
          FilterConcat(
            {
              SpatialConvolution(384, 256, 3, 1, 1, 1, 1, 0), -- 256x8x8
              SpatialConvolution(384, 256, 1, 3, 1, 1, 0, 1) -- 256x8x8
            }
          ) -- 512x8x8
        }
      ),
      Tower(
        {
          SpatialConvolution(1536, 384, 1, 1, 1, 1), -- 384x8x8
          SpatialConvolution(384, 448, 1, 3, 1, 1, 0, 1), -- 448x8x8
          SpatialConvolution(448, 512, 3, 1, 1, 1, 1, 0), -- 512x8x8
          FilterConcat(
            {
              SpatialConvolution(512, 256, 3, 1, 1, 1, 1, 0), -- 256x8x8
              SpatialConvolution(512, 256, 1, 3, 1, 1, 0, 1) -- 256x8x8
            }
          ) -- 512x8x8
        }
      ),
      Tower(
        {
          avgpool, -- 1536x8x8
          SpatialConvolution(1536, 256, 1, 1, 1, 1) -- 256x8x8
        }
      )
    }
  ) -- 1536x8x8
  -- 1536 ifms / ofms
  return inception
end

----------------
-- Load --
----------------

local function load_conv2d(module, name)
  local name = name or 'Conv2d_1a_3x3'
  local h5f = hdf5.open('dump/InceptionV4/'..name..'.h5', 'r')
  
  local conv = module:get(1) -- Spatial Convolution
  local weights = h5f:read("weights"):all():permute(4, 3, 1, 2)
  conv.weight:copy(weights)
  conv:noBias()

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

local function load_linear(module, name)
  print(module.weight:size())
  local h5f = hdf5.open('dump/InceptionV4/'..name..'.h5', 'r')
  local weights = h5f:read('weights'):all():t()
  local biases = h5f:read('biases'):all()
  module.weight:copy(weights)
  module.bias:copy(biases)

  print(weights:size())
  print(biases:size())
  print(module.bias:size())

  h5f:close()
end

local function load_mixed_4a_7a(module, name)
  load_conv2d(module:get(1):get(1), name..'/Branch_0/Conv2d_0a_1x1')
  load_conv2d(module:get(1):get(2), name..'/Branch_0/Conv2d_1a_3x3')
  load_conv2d(module:get(2):get(1), name..'/Branch_1/Conv2d_0a_1x1')
  load_conv2d(module:get(2):get(2), name..'/Branch_1/Conv2d_0b_1x7')
  load_conv2d(module:get(2):get(3), name..'/Branch_1/Conv2d_0c_7x1')
  load_conv2d(module:get(2):get(4), name..'/Branch_1/Conv2d_1a_3x3')
end

local function load_mixed_5(module, name)
  local name = name or 'Mixed_5b'
  load_conv2d(module:get(1), name..'/Branch_0/Conv2d_0a_1x1')
  load_conv2d(module:get(2):get(1), name..'/Branch_1/Conv2d_0a_1x1')
  load_conv2d(module:get(2):get(2), name..'/Branch_1/Conv2d_0b_3x3')
  load_conv2d(module:get(3):get(1), name..'/Branch_2/Conv2d_0a_1x1')
  load_conv2d(module:get(3):get(2), name..'/Branch_2/Conv2d_0b_3x3')
  load_conv2d(module:get(3):get(3), name..'/Branch_2/Conv2d_0c_3x3')
  load_conv2d(module:get(4):get(2), name..'/Branch_3/Conv2d_0b_1x1') -- pb
end

local function load_mixed_6(module, name)
  local name = name or 'Mixed_6b'
  load_conv2d(module:get(1), name..'/Branch_0/Conv2d_0a_1x1')
  load_conv2d(module:get(2):get(1), name..'/Branch_1/Conv2d_0a_1x1')
  load_conv2d(module:get(2):get(2), name..'/Branch_1/Conv2d_0b_1x7')
  load_conv2d(module:get(2):get(3), name..'/Branch_1/Conv2d_0c_7x1')
  load_conv2d(module:get(3):get(1), name..'/Branch_2/Conv2d_0a_1x1')
  load_conv2d(module:get(3):get(2), name..'/Branch_2/Conv2d_0b_7x1')
  load_conv2d(module:get(3):get(3), name..'/Branch_2/Conv2d_0c_1x7')
  load_conv2d(module:get(3):get(4), name..'/Branch_2/Conv2d_0d_7x1')
  load_conv2d(module:get(3):get(5), name..'/Branch_2/Conv2d_0e_1x7')
  load_conv2d(module:get(4):get(2), name..'/Branch_3/Conv2d_0b_1x1')
end

local function load_mixed_7(module, name)
  local name = name or 'Mixed_7b'
  load_conv2d(module:get(1), name..'/Branch_0/Conv2d_0a_1x1')
  load_conv2d(module:get(2):get(1), name..'/Branch_1/Conv2d_0a_1x1')
  load_conv2d(module:get(2):get(2):get(1), name..'/Branch_1/Conv2d_0b_1x3') -- Beware if inverse ??? TODO
  load_conv2d(module:get(2):get(2):get(2), name..'/Branch_1/Conv2d_0c_3x1')
  load_conv2d(module:get(3):get(1), name..'/Branch_2/Conv2d_0a_1x1')
  load_conv2d(module:get(3):get(2), name..'/Branch_2/Conv2d_0b_3x1')
  load_conv2d(module:get(3):get(3), name..'/Branch_2/Conv2d_0c_1x3')
  load_conv2d(module:get(3):get(4):get(1), name..'/Branch_2/Conv2d_0d_1x3') -- Beware
  load_conv2d(module:get(3):get(4):get(2), name..'/Branch_2/Conv2d_0e_3x1')
  load_conv2d(module:get(4):get(2), name..'/Branch_3/Conv2d_0b_1x1')
end

local function load(net)
  load_conv2d(net:get(1):get(1), 'Conv2d_1a_3x3')
  load_conv2d(net:get(1):get(2), 'Conv2d_2a_3x3')
  load_conv2d(net:get(1):get(3), 'Conv2d_2b_3x3')
  load_conv2d(net:get(1):get(4):get(2) ,'Mixed_3a/Branch_1/Conv2d_0a_3x3')
  
  load_mixed_4a_7a(net:get(1):get(5), 'Mixed_4a')
  
  load_conv2d(net:get(1):get(6):get(1) ,'Mixed_5a/Branch_0/Conv2d_1a_3x3')

  load_mixed_5(net:get(2), 'Mixed_5b') -- pb
  load_mixed_5(net:get(3), 'Mixed_5c')
  load_mixed_5(net:get(4), 'Mixed_5d')
  load_mixed_5(net:get(5), 'Mixed_5e')

  load_conv2d(net:get(6):get(1) ,'Mixed_6a/Branch_0/Conv2d_1a_3x3')
  load_conv2d(net:get(6):get(2):get(1) ,'Mixed_6a/Branch_1/Conv2d_0a_1x1')
  load_conv2d(net:get(6):get(2):get(2) ,'Mixed_6a/Branch_1/Conv2d_0b_3x3')
  load_conv2d(net:get(6):get(2):get(3) ,'Mixed_6a/Branch_1/Conv2d_1a_3x3')

  load_mixed_6(net:get(7), 'Mixed_6b')
  load_mixed_6(net:get(8), 'Mixed_6c')
  load_mixed_6(net:get(9), 'Mixed_6d')
  load_mixed_6(net:get(10), 'Mixed_6e')
  load_mixed_6(net:get(11), 'Mixed_6f')
  load_mixed_6(net:get(12), 'Mixed_6g')
  load_mixed_6(net:get(13), 'Mixed_6h')

  load_mixed_4a_7a(net:get(14), 'Mixed_7a')

  load_mixed_7(net:get(15), 'Mixed_7b')
  load_mixed_7(net:get(16), 'Mixed_7c')
  load_mixed_7(net:get(17), 'Mixed_7d')

  load_linear(net:get(20), 'Logits')
end

----------
-- Test --
----------

local function test_conv2d(module, name)
  local name = name or 'Conv2d_1a_3x3'
  local h5f = hdf5.open('dump/InceptionV4/'..name..'.h5', 'r')

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

local function test_mixed_4a_7a(module, name)
  test_conv2d(module:get(1):get(1), name..'/Branch_0/Conv2d_0a_1x1')
  test_conv2d(module:get(1):get(2), name..'/Branch_0/Conv2d_1a_3x3')
  test_conv2d(module:get(2):get(1), name..'/Branch_1/Conv2d_0a_1x1')
  test_conv2d(module:get(2):get(2), name..'/Branch_1/Conv2d_0b_1x7')
  test_conv2d(module:get(2):get(3), name..'/Branch_1/Conv2d_0c_7x1')
  test_conv2d(module:get(2):get(4), name..'/Branch_1/Conv2d_1a_3x3')
end

local function test_mixed_5(module, name)
  local name = name or 'Mixed_5b'
  test_conv2d(module:get(1), name..'/Branch_0/Conv2d_0a_1x1')
  test_conv2d(module:get(2):get(1), name..'/Branch_1/Conv2d_0a_1x1')
  test_conv2d(module:get(2):get(2), name..'/Branch_1/Conv2d_0b_3x3')
  test_conv2d(module:get(3):get(1), name..'/Branch_2/Conv2d_0a_1x1')
  test_conv2d(module:get(3):get(2), name..'/Branch_2/Conv2d_0b_3x3')
  test_conv2d(module:get(3):get(3), name..'/Branch_2/Conv2d_0c_3x3')
  test_conv2d(module:get(4):get(2), name..'/Branch_3/Conv2d_0b_1x1')
end

local function test_mixed_6(module, name)
  local name = name or 'Mixed_6b'
  test_conv2d(module:get(1), name..'/Branch_0/Conv2d_0a_1x1')
  test_conv2d(module:get(2):get(1), name..'/Branch_1/Conv2d_0a_1x1')
  test_conv2d(module:get(2):get(2), name..'/Branch_1/Conv2d_0b_1x7')
  test_conv2d(module:get(2):get(3), name..'/Branch_1/Conv2d_0c_7x1')
  test_conv2d(module:get(3):get(1), name..'/Branch_2/Conv2d_0a_1x1')
  test_conv2d(module:get(3):get(2), name..'/Branch_2/Conv2d_0b_7x1')
  test_conv2d(module:get(3):get(3), name..'/Branch_2/Conv2d_0c_1x7')
  test_conv2d(module:get(3):get(4), name..'/Branch_2/Conv2d_0d_7x1')
  test_conv2d(module:get(3):get(5), name..'/Branch_2/Conv2d_0e_1x7')
  test_conv2d(module:get(4):get(2), name..'/Branch_3/Conv2d_0b_1x1')
end

local function test_mixed_7(module, name)
  local name = name or 'Mixed_7b'
  test_conv2d(module:get(1), name..'/Branch_0/Conv2d_0a_1x1')
  test_conv2d(module:get(2):get(1), name..'/Branch_1/Conv2d_0a_1x1')
  test_conv2d(module:get(2):get(2):get(1), name..'/Branch_1/Conv2d_0b_1x3') -- Beware if inverse ??? TODO
  test_conv2d(module:get(2):get(2):get(2), name..'/Branch_1/Conv2d_0c_3x1')
  test_conv2d(module:get(3):get(1), name..'/Branch_2/Conv2d_0a_1x1')
  test_conv2d(module:get(3):get(2), name..'/Branch_2/Conv2d_0b_3x1')
  test_conv2d(module:get(3):get(3), name..'/Branch_2/Conv2d_0c_1x3')
  test_conv2d(module:get(3):get(4):get(1), name..'/Branch_2/Conv2d_0d_1x3') -- Beware
  test_conv2d(module:get(3):get(4):get(2), name..'/Branch_2/Conv2d_0e_3x1')
  test_conv2d(module:get(4):get(2), name..'/Branch_3/Conv2d_0b_1x1')
end

local function test_module(module, name)
  local name = name or 'Mixed_5b/Branch_3/AvgPool_0a_3x3'
  local h5f = hdf5.open('dump/InceptionV4/'..name..'.h5', 'r')
  local out = h5f:read("out"):all()
  out = out:transpose(2,4)
  out = out:transpose(3,4)
  h5f:close()
  if opt.cuda then
    out = out:cuda()
  end
  print(name..' test_out', torch.dist(module.output, out))
  print('')
end

local function test_linear(module, name)
  local h5f = hdf5.open('dump/InceptionV4/'..name..'.h5', 'r')
  local out = h5f:read("out"):all()
  h5f:close()
  if opt.cuda then
    out = out:cuda()
  end
  print(name..' linear_out', torch.dist(module.output, out))
  print('')
end

local function test(net)
  net:evaluate()
  local input = torch.zeros(1,3,299,299) -- [0,1]
  local img = image.load('lena_299.png') * 255.0 -- [0,255]
  input[1] = img:float()
  if opt.cuda then
    input = input:cuda()
  end
  local output = net:forward(input)

  test_conv2d(net:get(1):get(1), 'Conv2d_1a_3x3')
  test_conv2d(net:get(1):get(2), 'Conv2d_2a_3x3')
  test_conv2d(net:get(1):get(3), 'Conv2d_2b_3x3')
  test_conv2d(net:get(1):get(4):get(2) ,'Mixed_3a/Branch_1/Conv2d_0a_3x3')

  test_mixed_4a_7a(net:get(1):get(5), 'Mixed_4a')

  test_conv2d(net:get(1):get(6):get(1) ,'Mixed_5a/Branch_0/Conv2d_1a_3x3')

  -- test_module(net:get(2):get(4):get(1), 'Mixed_5b/Branch_3/AvgPool_0a_3x3')
  test_mixed_5(net:get(2), 'Mixed_5b')
  test_mixed_5(net:get(3), 'Mixed_5c')
  test_mixed_5(net:get(4), 'Mixed_5d')
  test_mixed_5(net:get(5), 'Mixed_5e')

  test_conv2d(net:get(6):get(1) ,'Mixed_6a/Branch_0/Conv2d_1a_3x3')
  test_conv2d(net:get(6):get(2):get(1) ,'Mixed_6a/Branch_1/Conv2d_0a_1x1')
  test_conv2d(net:get(6):get(2):get(2) ,'Mixed_6a/Branch_1/Conv2d_0b_3x3')
  test_conv2d(net:get(6):get(2):get(3) ,'Mixed_6a/Branch_1/Conv2d_1a_3x3')

  test_mixed_6(net:get(7), 'Mixed_6b')
  test_mixed_6(net:get(8), 'Mixed_6c')
  test_mixed_6(net:get(9), 'Mixed_6d')
  test_mixed_6(net:get(10), 'Mixed_6e')
  test_mixed_6(net:get(11), 'Mixed_6f')
  test_mixed_6(net:get(12), 'Mixed_6g')
  test_mixed_6(net:get(13), 'Mixed_6h')

  test_mixed_4a_7a(net:get(14), 'Mixed_7a')

  test_mixed_7(net:get(15), 'Mixed_7b')
  -- test_module(net:get(15):get(2), 'Mixed_7b/Branch_1/concat')
  -- test_module(net:get(15):get(3), 'Mixed_7b/Branch_2/concat')
  test_mixed_7(net:get(16), 'Mixed_7c')
  test_mixed_7(net:get(17), 'Mixed_7d')

  test_linear(net:get(21), 'Logits')
end

------------------------------------------------------------------
-- Main
------------------------------------------------------------------

opt = {
  cuda = true
}

net = nn.Sequential()
print("-- Stem")
net:add(Stem())           -- 3x299x299 ==> 384x35x35
print("-- Inception-A x 4")
for i=1,4 do
  net:add(Inception_A())  -- 384x35x35 ==> 384x35x35
end
print("-- Reduction-A")
net:add(Reduction_A())    -- 384x35x35 ==> 1024x17x17
print("-- Inception-B x 7")
for i=1,7 do
  net:add(Inception_B())  -- 1024x17x17 ==> 1024x17x17
end
print("-- Reduction-B")
net:add(Reduction_B())    -- 1024x17x17 ==> 1536x8x8
print("-- Inception-C x 3")
for i=1,3 do
  net:add(Inception_C())  -- 1536x8x8 ==> 1536x8x8
end
print("-- Average Pooling")
local avgpool = nn.SpatialAveragePooling(8, 8)
avgpool.count_include_pad = false
net:add(avgpool) -- 1536x8x8 ==> 1536x1x1
net:add(nn.View(1536))
print("-- Fully Connected")
net:add(nn.Linear(1536, 1001))  -- 1536 ==> 1000
net:add(nn.SoftMax())
print(net)

load(net)

if opt.cuda then
  require 'cunn'
  require 'cutorch'
  --require 'cudnn'
  net:cuda()
end

test(net)

os.execute('mkdir -p save')
torch.save('save/inceptionv4.t7', net:clearState():float())