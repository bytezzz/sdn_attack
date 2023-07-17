import torch
import math

import torch.nn as nn
import numpy as np

import aux_funcs as af
import model_funcs as mf

class ConvBlockWOutput(nn.Module):
    def __init__(self, conv_params, output_params):
        super(ConvBlockWOutput, self).__init__()
        input_channels = conv_params[0]
        output_channels = conv_params[1]
        max_pool_size = conv_params[2]
        batch_norm = conv_params[3]

        add_output = output_params[0]
        num_classes = output_params[1]
        input_size = output_params[2]
        self.output_id = output_params[3]

        self.depth = 1


        conv_layers = []
        conv_layers.append(nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3,padding=1, stride=1))

        if batch_norm:
            conv_layers.append(nn.BatchNorm2d(output_channels))

        conv_layers.append(nn.ReLU())

        if max_pool_size > 1:
            conv_layers.append(nn.MaxPool2d(kernel_size=max_pool_size))

        self.layers = nn.Sequential(*conv_layers)


        if add_output:
            self.output = af.InternalClassifier(input_size, output_channels, num_classes)
            self.no_output = False

        else:
            self.output = nn.Sequential()
            self.forward = self.only_forward
            self.no_output = True


    def forward(self, x):
        fwd = self.layers(x)
        return fwd, 1, self.output(fwd)

    def only_output(self, x):
        fwd = self.layers(x)
        return self.output(fwd)

    def only_forward(self, x):
        fwd = self.layers(x)
        return fwd, 0, None

class FcBlockWOutput(nn.Module):
    def __init__(self, fc_params, output_params, flatten=False):
        super(FcBlockWOutput, self).__init__()
        input_size = fc_params[0]
        output_size = fc_params[1]

        add_output = output_params[0]
        num_classes = output_params[1]
        self.output_id = output_params[2]
        self.depth = 1

        fc_layers = []

        if flatten:
            fc_layers.append(af.Flatten())

        fc_layers.append(nn.Linear(input_size, output_size))
        fc_layers.append(nn.ReLU())
        fc_layers.append(nn.Dropout(0.5))
        self.layers = nn.Sequential(*fc_layers)

        if add_output:
            self.output = nn.Linear(output_size, num_classes)
            self.no_output = False
        else:
            self.output = nn.Sequential()
            self.forward = self.only_forward
            self.no_output = True

    def forward(self, x):
        fwd = self.layers(x)
        return fwd, 1, self.output(fwd)

    def only_output(self, x):
        fwd = self.layers(x)
        return self.output(fwd)

    def only_forward(self, x):
        return self.layers(x), 0, None

class VGG_SDN(nn.Module):
    def __init__(self, params):
        super(VGG_SDN, self).__init__()
        # read necessary parameters
        self.input_size = int(params['input_size'])
        self.num_classes = int(params['num_classes'])
        self.conv_channels = params['conv_channels'] # the first element is input dimension
        self.fc_layer_sizes = params['fc_layers']
        self.output_layer = -1

        # read or assign defaults to the rest
        self.max_pool_sizes = params['max_pool_sizes']
        self.conv_batch_norm = params['conv_batch_norm']
        self.augment_training = params['augment_training']
        self.init_weights = params['init_weights']
        self.add_output = params['add_ic']

        self.train_func = mf.sdn_train
        self.test_func = mf.sdn_test
        self.num_output = sum(self.add_output) + 1

        self.init_conv = nn.Sequential() # just for compatibility with other models
        self.layers = nn.ModuleList()
        self.init_depth = 0

        self.init_rpf_channels = params['init_rpf_pannel']
        self.use_rpf = params['use_rpf']
        self.in_channels = 16

        self.early_stop = False
        self.need_info = False

        init_conv = []
        if self.use_rpf:
            init_conv.append(nn.Conv2d(3, self.in_channels - self.init_rpf_channels, kernel_size=3, stride=1, padding=1, bias=False))
            init_conv.append(nn.Conv2d(3, self.init_rpf_channels, kernel_size=3, stride=1, padding=1, bias=False))
        else:
            init_conv.append(nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False))

        init_conv.append(nn.BatchNorm2d(self.in_channels))
        init_conv.append(nn.ReLU())

        self.init_conv = nn.Sequential(*init_conv) # just for compatibility with other models

        # add conv layers
        # We changed input_channels
        input_channel = self.in_channels
        cur_input_size = self.input_size
        output_id = 0
        for layer_id, channel in enumerate(self.conv_channels):
            if self.max_pool_sizes[layer_id] == 2:
                cur_input_size = int(cur_input_size/2)
            conv_params =  (input_channel, channel, self.max_pool_sizes[layer_id], self.conv_batch_norm)
            add_output = self.add_output[layer_id]
            output_params = (add_output, self.num_classes, cur_input_size, output_id)
            self.layers.append(ConvBlockWOutput(conv_params, output_params))
            input_channel = channel
            output_id += add_output

        fc_input_size = cur_input_size*cur_input_size*self.conv_channels[-1]

        for layer_id, width in enumerate(self.fc_layer_sizes[:-1]):
            fc_params = (fc_input_size, width)
            flatten = False
            if layer_id == 0:
                flatten = True

            add_output = self.add_output[layer_id + len(self.conv_channels)]
            output_params = (add_output, self.num_classes, output_id)
            self.layers.append(FcBlockWOutput(fc_params, output_params, flatten=flatten))
            fc_input_size = width
            output_id += add_output

        end_layers = []
        end_layers.append(nn.Linear(fc_input_size, self.fc_layer_sizes[-1]))
        end_layers.append(nn.Dropout(0.5))
        end_layers.append(nn.Linear(self.fc_layer_sizes[-1], self.num_classes))
        self.end_layers = nn.Sequential(*end_layers)

        if self.init_weights:
            self.initialize_weights()

    def initialize_weights(self):
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
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

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
