import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import aux_funcs as af
import model_funcs as mf


class BlockWOutput(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_channels, out_channels, params, stride=1):
        super(BlockWOutput, self).__init__()

        add_output = params[0]
        num_classes = params[1]
        input_size = params[2]
        self.output_id = params[3]

        self.depth = 2

        conv_layers = []
        conv_layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False))
        conv_layers.append(nn.BatchNorm2d(in_channels))
        conv_layers.append(nn.ReLU())
        conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        conv_layers.append(nn.BatchNorm2d(out_channels))
        conv_layers.append(nn.ReLU())

        self.layers = nn.Sequential(*conv_layers)

        if add_output:
            self.output = af.InternalClassifier(input_size, out_channels, num_classes)
            self.no_output = False

        else:
            self.forward = self.only_forward
            self.output = nn.Sequential()
            self.no_output = True

    def forward(self, x):
        fwd = self.layers(x)
        return fwd, 1, self.output(fwd)

    def only_output(self, x):
        fwd = self.layers(x)
        return self.output(fwd)

    def only_forward(self, x):
        return self.layers(x), 0, None

class MobileNet_SDN(nn.Module):
    # (128,2) means conv channels=128, conv stride=2, by default conv stride=1
    def __init__(self, params):
        super(MobileNet_SDN, self).__init__()
        self.cfg = params['cfg']
        self.num_classes = int(params['num_classes'])
        self.augment_training = params['augment_training']
        self.input_size = int(params['input_size'])
        self.add_output = params['add_ic']
        self.train_func = mf.sdn_train
        self.test_func = mf.sdn_test
        self.num_output = sum(self.add_output) + 1
        self.in_channels = 32
        self.cur_input_size = self.input_size

        self.init_depth = 1
        self.end_depth = 1
        self.cur_output_id = 0

        self.init_rpf_channels = params['init_rpf_pannel']
        self.use_rpf = params['use_rpf']

        init_conv = []
        if self.use_rpf:
            init_conv.append(nn.Conv2d(3, self.init_rpf_channels, kernel_size=3, stride=1, padding=1, bias=False))
            init_conv.append(nn.Conv2d(3, self.in_channels - self.init_rpf_channels, kernel_size=3, stride=1, padding=1, bias=False))
        else:
            init_conv.append(nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False))
        init_conv.append(nn.BatchNorm2d(self.in_channels))
        init_conv.append(nn.ReLU(inplace=True))
        self.init_conv = nn.Sequential(*init_conv)

        self.early_stop = False
        self.need_info = False

        self.layers = nn.ModuleList()
        self.layers.extend(self._make_layers(in_channels=self.in_channels))

        end_layers = []
        if self.input_size == 32: # cifar10 and cifar100
            end_layers.append(nn.AvgPool2d(2))
        elif self.input_size == 64: # tiny imagenet
            end_layers.append(nn.AvgPool2d(4))

        end_layers.append(af.Flatten())
        end_layers.append(nn.Linear(1024, self.num_classes))
        self.end_layers = nn.Sequential(*end_layers)

    def _make_layers(self, in_channels):
        layers = []

        for block_id, x in enumerate(self.cfg):
            out_channels = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            if stride == 2:
                self.cur_input_size = int(self.cur_input_size/2)

            add_output = self.add_output[block_id]
            params  = (add_output, self.num_classes, self.cur_input_size, self.cur_output_id)
            layers.append(BlockWOutput(in_channels, out_channels, params, stride))
            in_channels = out_channels
            self.cur_output_id += add_output

        return layers

    def forward(self, x):
        if self.early_stop:
            return self.early_exit(x)
        outputs = []
        if self.use_rpf:
            fwd = self.init_conv[0](x)
            fwd = self.rp_forward(x, fwd, self.init_conv[1])
            fwd = self.init_conv[2](fwd)
        else:
            fwd = self.init_conv(x)
        for layer in self.layers:
            fwd, is_output, output = layer(fwd)
            if is_output:
                outputs.append(output)
        fwd = self.end_layers(fwd)
        outputs.append(fwd)

        if self.need_info:
            return outputs
        else:
            return nn.functional.softmax(fwd, dim = 1)

    # takes a single input
    def early_exit(self, x):
        device = next(self.parameters()).device
        batch_size = x.shape[0]
        result_prob = torch.zeros((batch_size,self.num_classes), dtype=torch.float).to(device)
        stop_at = torch.zeros(batch_size).to(device)
        stopped = torch.tensor([0]*batch_size, dtype=torch.bool).to(device)
        if self.use_rpf:
            fwd = self.init_conv[0](x)
            fwd = self.rp_forward(x, fwd, self.init_conv[1])
            fwd = self.init_conv[2](fwd)
        else:
            fwd = self.init_conv(x)
        output_id = 0
        for layer in self.layers:
            fwd, is_output, output = layer(fwd)

            if is_output:
                softmax = nn.functional.softmax(output, dim=1)
                confidence = torch.max(softmax, dim=1)
                stop_index = (confidence.values > self.confidence_threshold).view(-1) & ~stopped
                result_prob[stop_index] = softmax[stop_index]
                stop_at[stop_index] = output_id
                stopped[stop_index] = True
                output_id += is_output

        output = self.end_layers(fwd)
        softmax = nn.functional.softmax(output, dim=1)
        confidence = torch.max(softmax, dim=1)
        result_prob[~stopped] = softmax[~stopped]
        stop_at[stop_index] = -1
        if self.need_info:
            return result_prob, stop_at
        else:
            return result_prob

    def random_rp_matrix(self):
        param = next(self.init_conv[0].parameters())
        kernel_size = param.data.size()[-1]
        param.data = torch.normal(mean=0.0, std=1/kernel_size, size=param.data.size()).to('cuda')

    def rp_forward(self, x, out, kernel):
        rp_out = kernel(x)
        if out is None:
            return rp_out
        else:
            out = torch.cat([out, rp_out], dim=1)
            return out
