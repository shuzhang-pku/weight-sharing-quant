import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, Callable, List, Optional, Type, Union
import random


from qfa.layers import IdentityLayer
from qfa.elastic_nn.modules.dynamic_q_op import DynamicWeightQuantizer, DynamicActivationQuantizer
from qfa.elastic_nn.modules.dynamic_q_layers import *
from qfa.imagenet_codebase.networks.mobilenet_v3 import MobileNetV3, MobileInvertedResidualBlock
from qfa.imagenet_codebase.utils import FLOPsTable

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, bits_list=[2,3,4,8,32]):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            DynamicQConvLayer(in_channel_list=in_channels, out_channel_list=out_channels,
                              kernel_size=3, stride=stride, use_bn=True, act_func='relu',bits_list=bits_list),
            DynamicQConvLayer(in_channel_list=out_channels, out_channel_list=out_channels * BasicBlock.expansion,
                              kernel_size=3, use_bn=True, act_func=None,bits_list=bits_list)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = DynamicQConvLayer(in_channel_list=in_channels, out_channel_list=out_channels * BasicBlock.expansion,
                              kernel_size=1, stride=stride, use_bn=True, act_func=None, bits_list=bits_list)

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class Bottleneck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * Bottleneck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * Bottleneck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * Bottleneck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * Bottleneck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * Bottleneck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))
    

class QResNet(nn.Module):
    def __init__(self, block, num_block, num_classes=100, bits_list: List[int] = [2,3,4,8,32]):
        super().__init__()

        self.in_channels = 64

        self.conv1 = DynamicQConvLayer(in_channel_list=3, out_channel_list=64,
                              kernel_size=3,  use_bn=True, act_func='relu', bits_list=bits_list)
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1, bits_list=bits_list)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2, bits_list=bits_list)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2, bits_list=bits_list)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2, bits_list=bits_list)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = DynamicQLinearLayer(in_features_list=512 * block.expansion, out_features=num_classes, bits_list=bits_list)
        
        
        self.flops_table = FLOPsTable()
        self.bits_list = int2list(bits_list, 1)
        self.bits_list.sort()
        self.quantizers = []
        self.quantizer_dict = {}

        for n, m in self.named_modules():
            if type(m) in [DynamicWeightQuantizer, DynamicActivationQuantizer]:
                self.quantizers.append(m)
                self.quantizer_dict[n] = m

    def _make_layer(self, block, out_channels, num_blocks, stride, bits_list: List[int] = [2,3,4,32]):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, bits_list))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output
    
    @staticmethod
    def name():
        return 'QResNet'
    
    @property
    def module_str(self):
        #todo: module str of bit
        '''
        _str = self.first_conv.module_str + '\n'
        _str += self.blocks[0].module_str + '\n'

        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            for idx in active_idx:
                _str += self.blocks[idx].module_str + '\n'

        _str += self.final_expand_layer.module_str + '\n'
        _str += self.feature_mix_layer.module_str + '\n'
        _str += self.classifier.module_str + '\n'
        '''
        return 'QResNet(need mix-precision informantion in detail)'
    

    def load_state_dict(self, src_model_dict):
        self.load_weights_from_float_net(src_model_dict)

    def load_weights_from_float_net(self, src_model_dict):
        import re
        conv_pattern = re.compile(r'.*conv(\d+).weight')
        model_dict = self.state_dict()
        for key in src_model_dict:
            if key in model_dict:
                new_key = key
            else:
                new_key = None
                if key == 'conv1.weight':
                    print('load first conv')
                    new_key = 'first_conv.conv.conv.weight'
                elif conv_pattern.match(key):
                    new_key = key
                    new_key = new_key.replace('.weight','.conv.conv.weight')
                elif key == 'fc.weight':
                    new_key = 'fc.linear.linear.weight'
                elif key == 'fc.bias':
                    new_key = 'fc.linear.linear.bias'
            if new_key:
                model_dict[new_key] = src_model_dict[key]
            # assert new_key in model_dict, '%s' % new_key
            # model_dict[new_key] = src_model_dict[key]
        super(QResNet, self).load_state_dict(model_dict)


    def override_quantizer(self):
        for k, q in self.quantizer_dict.items():
            if 'first_conv' in k:
                q.active_bit = 32
            if 'fc' in k:
                if q.active_bit < 8:
                    q.active_bit = 8

        bit_list = [b for b in self.bits_list if b != 32]
        for n, m in self.named_modules():
            if isinstance(m, DynamicPointQConv2d):
                w_bits = m.w_quantizer.active_bit
                a_bits = m.a_quantizer.active_bit
                if w_bits != a_bits and 32 in [w_bits, a_bits]:
                    if random.random() < 0.2:
                        m.w_quantizer.active_bit = 32
                        m.a_quantizer.active_bit = 32
                    else:
                        if m.w_quantizer.active_bit == 32:
                            m.w_quantizer.active_bit = random.choice(bit_list)
                        if m.a_quantizer.active_bit == 32:
                            m.a_quantizer.active_bit = random.choice(bit_list)

            if isinstance(m, DynamicSeparableQConv2d):
                for ks in m.w_quantizers.keys():
                    w_bits = m.w_quantizers[str(ks)].active_bit
                    a_bits = m.a_quantizers[str(ks)].active_bit
                    if w_bits != a_bits and 32 in [w_bits, a_bits]:
                        if random.random() < 0.2:
                            m.w_quantizers[str(ks)].active_bit = 32
                            m.a_quantizers[str(ks)].active_bit = 32
                        else:
                            if m.w_quantizers[str(ks)].active_bit == 32:
                                m.w_quantizers[str(ks)].active_bit = random.choice(bit_list)
                            if m.a_quantizers[str(ks)].active_bit == 32:
                                m.a_quantizers[str(ks)].active_bit = random.choice(bit_list)

            if isinstance(m, DynamicQLinear):
                w_bits = m.w_quantizer.active_bit
                a_bits = m.a_quantizer.active_bit
                if w_bits != a_bits and 32 in [w_bits, a_bits]:
                    if random.random() < 0.666:
                        m.w_quantizer.active_bit = 32
                        m.a_quantizer.active_bit = 32
                    else:
                        m.w_quantizer.active_bit = 8
                        m.a_quantizer.active_bit = 8

        bits_list = [q.active_bit for q in self.quantizers]
        return bits_list

    def set_max_net(self):
        return self.set_sandwich_subnet('max')
    
    def set_active_subnet(self, b=None):
        bits = int2list(b, len(self.quantizers))
        for i, q in enumerate(self.quantizers):
        
            if type(q) in [DynamicActivationQuantizer]:
                q.active_bit = 8
            elif bits[i] is not None:
                q.active_bit = bits[i]
            #q.active_bit = bits[i]
        return self.override_quantizer()
    
    def set_sandwich_subnet(self, mode = 'max', fix_bit = None):
        assert mode in ['min', 'max']
        aggregate = min if mode == 'min' else max
        if fix_bit:
            bits_setting = fix_bit
        else:
            bits_setting = random.choice(self.bits_list)
        bits_setting = self.set_active_subnet( bits_setting)
        return bits_setting
    

    def set_constraint(self, include_list, constraint_type='depth'):
        if constraint_type == 'bits':
            self.__dict__['_bits_include_list'] = include_list.copy()
        else:
            raise NotImplementedError

    def clear_constraint(self):
        self.__dict__['_bits_include_list'] = None

    #todo
    def get_pareto_config(self, res):

        bits_setting = []
        for i, q in enumerate(self.quantizers):
            bits_setting.append(q.active_bit)

        pareto_cfg = {
            'b': bits_setting,
        }

        bs = [[self.first_conv.conv.w_quantizer.active_bit, self.first_conv.conv.a_quantizer.active_bit]]
        for block in self.blocks:
            inv_w, inv_a = 4, 4
            inv = block.mobile_inverted_conv.inverted_bottleneck
            if inv:
                inv_w = inv.conv.w_quantizer.active_bit
                inv_a = inv.conv.a_quantizer.active_bit
            dep = block.mobile_inverted_conv.depth_conv.conv
            kernel_size = dep.active_kernel_size
            dep_w = dep.w_quantizers[str(kernel_size)].active_bit
            dep_a = dep.a_quantizers[str(kernel_size)].active_bit
            pw = block.mobile_inverted_conv.point_linear.conv
            pw_w = pw.w_quantizer.active_bit
            pw_a = pw.a_quantizer.active_bit
            bs.append([inv_w, inv_a, dep_w, dep_a, pw_w, pw_a])
        bs += [
            [self.final_expand_layer.conv.w_quantizer.active_bit,
             self.final_expand_layer.conv.a_quantizer.active_bit],
            [self.feature_mix_layer.conv.w_quantizer.active_bit,
             self.feature_mix_layer.conv.a_quantizer.active_bit],
            [self.classifier.linear.w_quantizer.active_bit,
             self.classifier.linear.a_quantizer.active_bit],
        ]

        pareto_cfg['bs'] = bs

        return pareto_cfg



    def sample_active_subnet(self, res=None, subnet_seed=None):
        if subnet_seed is not None:
            random.seed(subnet_seed)

        return self.sample_active_subnet_helper()

    def sample_active_subnet_helper(self):
        bits_candidates = self.bits_list if self.__dict__.get('_bits_include_list', None) is None else \
            self.__dict__['_bits_include_list']

        # sample bits
        bits_setting = []
        if not isinstance(bits_candidates[0], list):
            bits_candidates = [bits_candidates for _ in range(len(self.quantizers))]
        for b_set in bits_candidates:
            b = random.choice(b_set)
            bits_setting.append(b)

        bits_setting = self.set_active_subnet(bits_setting)
        return bits_setting

    #todo未出现过?
    def get_active_subnet(self, preserve_weight=True):
        first_conv = self.first_conv.get_active_subnet(in_channel=3, preserve_weight=preserve_weight)
        blocks = [MobileInvertedResidualBlock(
            self.blocks[0].mobile_inverted_conv.get_active_subnet(
                in_channel=self.first_conv.active_out_channel,
                preserve_weight=preserve_weight),
            copy.deepcopy(self.blocks[0].shortcut))]

        final_expand_layer = self.final_expand_layer.get_active_subnet(
            in_channel=self.blocks[-1].mobile_inverted_conv.active_out_channel, preserve_weight=preserve_weight)
        feature_mix_layer = self.feature_mix_layer.get_active_subnet(
            in_channel=self.final_expand_layer.active_out_channel, preserve_weight=preserve_weight)
        classifier = self.classifier.get_active_subnet(
            in_features=self.feature_mix_layer.active_out_channel, preserve_weight=preserve_weight)

        input_channel = blocks[0].mobile_inverted_conv.out_channels
        # blocks
        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            stage_blocks = []
            for idx in active_idx:
                stage_blocks.append(
                    MobileInvertedResidualBlock(
                        self.blocks[idx].mobile_inverted_conv.get_active_subnet(in_channel=input_channel,
                                                                                preserve_weight=preserve_weight),
                        copy.deepcopy(self.blocks[idx].shortcut)
                    )
                )
                input_channel = stage_blocks[-1].mobile_inverted_conv.out_channels
            blocks += stage_blocks

        _subnet = MobileNetV3(first_conv, blocks, final_expand_layer, feature_mix_layer, classifier)
        return _subnet




def qresnet18():
    """ return a ResNet 18 object
    """
    return QResNet(BasicBlock, [2, 2, 2, 2])

def qresnet34(bits_list = [2,3,4,8,32]):
    """ return a ResNet 34 object
    """
    return QResNet(BasicBlock, [3, 4, 6, 3], bits_list=bits_list)

def qresnet50():
    """ return a ResNet 50 object
    """
    return QResNet(Bottleneck, [3, 4, 6, 3])
