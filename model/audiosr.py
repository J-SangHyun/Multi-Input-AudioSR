# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class DilatedLayer(nn.Module):
    def __init__(self, hidden, dilation):
        super(DilatedLayer, self).__init__()
        self.dil_conv = nn.Conv1d(hidden, hidden, kernel_size=(3,), padding=dilation, dilation=dilation)
        self.local_conv = nn.Conv1d(hidden, hidden, kernel_size=(3,), padding=1)
        self.direct_conv = nn.Conv1d(hidden, hidden, kernel_size=(1,))
        self.gate_conv = nn.Conv1d(hidden, hidden, kernel_size=(3,), padding=1)
        self.filter_conv = nn.Conv1d(hidden, hidden, kernel_size=(3,), padding=1)
        self.proj_conv = nn.Conv1d(hidden, hidden, kernel_size=(1,))

    def forward(self, x):
        y = self.dil_conv(x) + self.local_conv(x) + self.direct_conv(x)
        g = self.gate_conv(y)
        f = self.filter_conv(y)
        y = torch.sigmoid(g) * torch.tanh(f)
        y = self.proj_conv(y)
        return x + y, y


class DilatedBlock(nn.Module):
    def __init__(self, hidden_channel, n_layers):
        super(DilatedBlock, self).__init__()
        self.layers = nn.ModuleList([DilatedLayer(hidden_channel, 3**i) for i in range(n_layers)])

    def forward(self, x):
        y = 0
        for layer in self.layers:
            x, skip = layer(x)
            y = skip + y
        return y


class AudioSR(nn.Module):
    def __init__(self, rate):
        super(AudioSR, self).__init__()
        wave_channels = 192
        spec_channels = 32
        n_blocks = 2

        self.name = 'AudioSR'
        self.rate = rate
        self.n_blocks = n_blocks
        self.wave_channels = wave_channels
        self.spec_channels = spec_channels
        self.wave_encoder = WaveEncoder(n_blocks=2, hidden_channels=wave_channels)
        self.spec_encoder = SpecEncoder(n_layers=4, hidden_channels=spec_channels, input_channels=513)

        overall_channels = wave_channels + spec_channels
        self.dilated_blocks = nn.ModuleList([DilatedBlock(overall_channels, 6) for _ in range(n_blocks)])

        shuffle_channels = (overall_channels // rate) * rate
        self.post_conv1 = nn.Conv1d(overall_channels, shuffle_channels, kernel_size=(3,), padding=1)
        self.shuffler = PixelShuffle1D(rate)
        self.post_conv2 = nn.Conv1d(shuffle_channels//rate, shuffle_channels//rate, kernel_size=(3,), padding=1)
        self.post_conv3 = nn.Conv1d(shuffle_channels//rate, shuffle_channels//rate, kernel_size=(3,), padding=1)
        self.post_conv4 = nn.Conv1d(shuffle_channels//rate, 1, kernel_size=(3,), padding=1)

    def forward(self, lr_wave, lr_spec):
        B, L_wave = lr_wave.size()
        B, C_spec, L_spec = lr_spec.size()
        lr_wave = lr_wave.unsqueeze(1)

        wave_feature = self.wave_encoder(lr_wave)
        spec_feature = self.spec_encoder(lr_spec)
        spec_feature = spec_feature.unsqueeze(1)
        scale_factor = L_wave / L_spec
        spec_feature = F.interpolate(spec_feature, scale_factor=(1, scale_factor), mode='bilinear',
                                     align_corners=True, recompute_scale_factor=True)
        spec_feature = spec_feature.squeeze(1)
        x = torch.concat([wave_feature, spec_feature], dim=1)

        for i in range(self.n_blocks):
            x = self.dilated_blocks[i](x)
        x = F.silu(x)
        x = F.silu(self.post_conv1(x))
        x = self.shuffler(x)
        x = F.silu(self.post_conv2(x))
        x = F.silu(self.post_conv3(x))
        x = self.post_conv4(x)
        sr_wave = x.squeeze(1)
        return sr_wave


class WaveEncoder(nn.Module):
    def __init__(self, n_blocks, hidden_channels):
        super(WaveEncoder, self).__init__()
        self.n_blocks = n_blocks
        self.pre_conv = nn.Conv1d(1, hidden_channels, kernel_size=(3,), padding=1)
        self.dilated_blocks = nn.ModuleList([DilatedBlock(hidden_channels, 6) for _ in range(n_blocks)])
        self.post_conv = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=(3,), padding=1)

    def forward(self, lr_wave):
        x = self.pre_conv(lr_wave)
        x = F.silu(x)

        for i in range(self.n_blocks):
            x = self.dilated_blocks[i](x)
        x = F.silu(x)
        x = self.post_conv(x)
        return x


class SpecEncoder(nn.Module):
    def __init__(self, n_layers, hidden_channels, input_channels=513):
        super(SpecEncoder, self).__init__()
        self.n_layers = n_layers
        self.pre_conv = nn.Conv1d(input_channels, hidden_channels, kernel_size=(3,), padding=1)
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=(3,), padding=1)
            for _ in range(n_layers)])
        self.post_conv = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=(3,), padding=1)

    def forward(self, lr_spec):
        x = self.pre_conv(lr_spec)
        x = F.silu(x)

        for i in range(self.n_layers):
            x = self.convs[i](x)
            x = F.silu(x)
        x = self.post_conv(x)
        return x


class PixelShuffle1D(torch.nn.Module):
    def __init__(self, upscale_factor):
        super(PixelShuffle1D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        batch_size = x.shape[0]
        short_channel_len = x.shape[1]
        short_width = x.shape[2]

        long_channel_len = short_channel_len // self.upscale_factor
        long_width = self.upscale_factor * short_width

        x = x.contiguous().view([batch_size, self.upscale_factor, long_channel_len, short_width])
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, long_channel_len, long_width)
        return x
