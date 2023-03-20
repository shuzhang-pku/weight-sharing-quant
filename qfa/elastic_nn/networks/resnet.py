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


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        bits_list: List[int] = [2,3,4,32]
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = DynamicQConvLayer(
            in_channel_list = inplanes, out_channel_list=width, bits_list=bits_list,
            kernel_size=1, stride=stride, dilation = dilation,use_bn=True, act_func='relu', signed= True
        )
        self.conv2 = DynamicQConvLayer(
            in_channel_list = width, out_channel_list=width, bits_list=bits_list,
            kernel_size=3, stride=stride,dilation=dilation, use_bn=True, act_func='relu', signed= True
        )

        self.conv3 = DynamicQConvLayer(
            in_channel_list = width, out_channel_list=planes * self.expansion, bits_list=bits_list,
            kernel_size=1, stride=stride, dilation=dilation,use_bn=True, act_func=None, signed= True
        )

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)

        out = self.conv2(out)

        out = self.conv3(out)


        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
    def get_active_subnet():
        pass
    


class QResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        bits_list: List[int] = [2,3,4,32]
    ) -> None:
        super().__init__()

        #mix-precision-specific
        self.flops_table = FLOPsTable()
        self.bits_list = int2list(bits_list, 1)
        self.bits_list.sort()
        self.quantizers = []
        self.quantizer_dict = {}
        for n, m in self.named_modules():
            if type(m) in [DynamicWeightQuantizer, DynamicActivationQuantizer]:
                self.quantizers.append(m)
                self.quantizer_dict[n] = m


        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.first_conv = DynamicQConvLayer(3, self.inplanes, kernel_size= 7 ,stride=2,use_bn=True, act_func='relu', bits_list= bits_list)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.blocks=[self.layer1, self.layer2, self.layer3, self.layer4]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = DynamicQLinearLayer(in_features_list=512 * block.expansion, out_features_list=num_classes, bits_list=bits_list)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
        bits_list: List[int] =[2,3,4,32]
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample =DynamicQConvLayer(self.inplanes, planes * block.expansion, bits_list=bits_list,
                                kernel_size = 1, stride=stride, use_bn=True)

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer, bits_list
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    bits_list=bits_list
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.first_conv(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
    
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
        model_dict = self.state_dict()
        for key in src_model_dict:
            if key in model_dict:
                new_key = key
            elif 'first_conv' in key or 'blocks.0' in key or 'final_expand_layer' in key or 'feature_mix_layer' in key:
                rep_list = [
                    ('conv.weight', 'conv.conv.weight'),
                    ('bn.weight', 'conv.gamma'),
                    ('bn.bias', 'conv.beta'),
                    ('bn', 'conv')
                ]
                new_key = key
                for v1, v2 in rep_list:
                    new_key = new_key.replace(v1, v2)
            elif 'blocks' in key:
                rep_list = [
                    ('bn.bn.weight', 'conv.gamma'),
                    ('bn.bn.bias', 'conv.beta'),
                    ('bn.bn', 'conv')
                ]
                new_key = key
                for v1, v2 in rep_list:
                    new_key = new_key.replace(v1, v2)
            elif 'classifier' in key:
                new_key = key.replace('.linear.', '.linear.linear.')
            else:
                raise ValueError(key)
            if new_key not in model_dict:
                assert 'w_q' in key, '%s' % new_key
            else:
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
            if bits[i] is not None:
                q.active_bit = bits[i]

        return self.override_quantizer()
    
    def set_sandwich_subnet(self, mode):
        assert mode in ['min', 'max']
        aggregate = min if mode == 'min' else max
        bits_setting = random.choice(self.bits_list)

        bits_setting = self.set_active_subnet( bits_setting)
        return {

            'b': bits_setting,
        }

    

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

        return {
            'b': bits_setting,
        }

    #todo
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




def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    bits_list: List[int] = [2,3,4,32]
) -> QResNet:


    model = QResNet(block, layers, bits_list = bits_list)



    return model

def resnet34(bits_list: List[int] = [2,3,4,32]):

    return _resnet(BasicBlock, [3, 4, 6, 3], bits_list)



def resnet50(bits_list: List[int] = [2,3,4,32]) -> QResNet:
 


    return _resnet(Bottleneck, [3, 4, 6, 3],bits_list)