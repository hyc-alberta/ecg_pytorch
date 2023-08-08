import torch
import torch.nn as nn


def get_num_filters_at_index(index, **params):
    return 2**int(index / params["conv_increase_channels_at"]) \
        * params["conv_num_filters_start"]


def zeropad_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 3
    shape[2] *= 2
    return tuple(shape)


class ecgNet(nn.Module):
    def __init__(
            self,
            **params
    ):
        super(ecgNet, self).__init__()
        self.initialBlock = nn.Sequential(
            nn.Conv1d(1, 32, params['conv_filter_length'], stride=1, padding='same'),
            nn.BatchNorm1d(params['conv_num_filters_start']),
            nn.ReLU()
        )
        hiddenLayers = []
        for i in range(len(params["conv_subsample_lengths"])):
            hiddenLayers.append(resnetBlock(i, **params))
        self. hiddenLayers = nn.Sequential(*hiddenLayers)
        self.outputBlock1 = nn.Sequential(
            nn.BatchNorm1d((2**3)*params['conv_num_filters_start']),
            nn.ReLU(),
        )
        self.outputBlock2 = nn.Sequential(
            nn.Linear(
            params['conv_num_filters_start']*2**3,
            params['num_categories']),
        )
        for m in self.modules():
            m = m.to(params['device'])
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = self.initialBlock(x)
        x = self.hiddenLayers(x)
        x = self.outputBlock1(x)
        x = x.reshape(x.shape[0], -1)
        x = self.outputBlock2(x)
        return x


class resnetBlock(nn.Module):
    def __init__(
            self,
            index,
            **params
    ):
        super(resnetBlock, self).__init__()
        self.index = index
        in_channels = get_num_filters_at_index(index-1, **params) if index > 0\
            else params["conv_num_filters_start"]
        out_channels = get_num_filters_at_index(index, **params)
        subsample_length = params['conv_subsample_lengths'][index]
        self.zero_pad = (index % params["conv_increase_channels_at"]) == 0 \
                   and index > 0
        self.shortcut = nn.MaxPool1d(subsample_length)
        if index > 0:
            self.block0 = nn.Sequential(
                nn.BatchNorm1d(in_channels),
                nn.ReLU(),
                nn.Dropout1d()
            )
        self.block1 = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                params['conv_filter_length'],
                subsample_length,
                padding='same' if subsample_length == 1 \
                    else int(params['conv_filter_length']/2-1)
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(out_channels),
            nn.Dropout(p=0.2),
            nn.Conv1d(
                out_channels,
                out_channels,
                params['conv_filter_length'],
                stride=1,
                padding='same')
        )

    def forward(self, x):
        if self.index == 0:
            x0 = x
            x = self.block1(x)
            x = self.shortcut(x0)+x
        else:
            x0 = x
            x = self.block0(x)
            x = self.block1(x)
            x0 = self.shortcut(x0)
            if self.zero_pad is True:
                x0 = torch.concatenate([x0, torch.zeros_like(x0)], dim=1)
            x = x+x0
        return x







