import math
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm
from librosa.filters import mel as librosa_mel_fn


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


LRELU_SLOPE = 0.1


class ResBlock1(torch.nn.Module):
    def __init__(
        self,
        channels,
        kernel_size=3,
        dilation=(1, 3, 5),
        norm_type: Literal["weight", "spectral"] = "weight"
    ):
        norm = dict(weight=weight_norm, spectral=spectral_norm)[norm_type]
        super(ResBlock1, self).__init__()
        self.convs1 = nn.ModuleList(
            [
                norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            ]
        )
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x


class ResBlock2(torch.nn.Module):
    def __init__(
        self,
        channels,
        kernel_size=3,
        dilation=(1, 3),
        norm_type: Literal["weight", "spectral"] = "weight"
    ):
        super(ResBlock2, self).__init__()
        norm = dict(weight=weight_norm, spectral=spectral_norm)[norm_type]
        self.convs = nn.ModuleList(
            [
                norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
            ]
        )
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x


class AddSkipConn(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.add = torch.nn.quantized.FloatFunctional()

    def forward(self, x):
        return self.add.add(x, self.net(x))


class ConcatSkipConn(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return torch.cat([x, self.net(x)], 1)


def build_block(
        inner_width,
        block_depth,
        mode: Literal["unet_k3_2d", "waveunet_k5"],
        norm
):
    if mode == "unet_k3_2d":
        return nn.Sequential(
            *[
                AddSkipConn(
                    nn.Sequential(
                        nn.LeakyReLU(),
                        norm(
                            nn.Conv2d(
                                inner_width,
                                inner_width,
                                3,
                                padding=1,
                                bias=True,
                            )
                        ),
                    )
                )
                for _ in range(block_depth)
            ]
        )
    elif mode == "waveunet_k5":
        return nn.Sequential(
            *[
                AddSkipConn(
                    nn.Sequential(
                        nn.LeakyReLU(),
                        norm(
                            nn.Conv1d(
                                inner_width,
                                inner_width,
                                5,
                                padding=2,
                                bias=True,
                            )
                        ),
                    )
                )
                for _ in range(block_depth)
            ]
        )
    else:
        raise NotImplementedError


class MultiScaleResnet(nn.Module):
    def __init__(
        self,
        block_widths,
        block_depth,
        in_width=1,
        out_width=1,
        downsampling_conv=True,
        upsampling_conv=True,
        concat_skipconn=True,
        scale_factor=4,
        mode: Literal["waveunet_k5"] = "waveunet_k5",
        norm_type: Literal["weight", "spectral", "id"] = "id"
    ):
        super().__init__()
        norm = dict(
            weight=weight_norm, spectral=spectral_norm, id=lambda x: x
        )[norm_type]
        self.in_width = in_width
        self.out_dims = out_width
        net = build_block(block_widths[-1], block_depth, mode, norm)
        for i in range(len(block_widths) - 1):
            width = block_widths[-2 - i]
            inner_width = block_widths[-1 - i]
            if downsampling_conv:
                downsampling = norm(nn.Conv1d(
                    width, inner_width, scale_factor, scale_factor, 0
                ))
            else:
                downsampling = nn.Sequential(
                    nn.AvgPool1d(scale_factor, scale_factor),
                    norm(nn.Conv1d(width, inner_width, 1)),
                )
            if upsampling_conv:
                upsampling = nn.Sequential(
                    nn.Upsample(scale_factor=scale_factor),
                    norm(nn.Conv1d(inner_width, width, 1)),
                )
            else:
                upsampling = norm(nn.ConvTranspose1d(
                    inner_width, width, scale_factor, scale_factor, 0
                ))
            net = nn.Sequential(
                build_block(width, block_depth, mode, norm),
                downsampling,
                net,
                upsampling,
                build_block(width, block_depth, mode, norm),
            )
            if concat_skipconn:
                net = nn.Sequential(
                    ConcatSkipConn(net),
                    norm(nn.Conv1d(width * 2, width, 1)),
                )
            else:
                net = AddSkipConn(net)
        self.net = nn.Sequential(
            norm(nn.Conv1d(in_width, block_widths[0], 5, padding=2)),
            net,
            norm(nn.Conv1d(block_widths[0], out_width, 5, padding=2)),
        )

    def forward(self, x):
        assert x.shape[1] == self.in_width, (
            "%d-dimensional condition is assumed" % self.in_width
        )
        return self.net(x)


class MultiScaleResnet2d(nn.Module):
    def __init__(
        self,
        block_widths,
        block_depth,
        in_width=1,
        out_width=1,
        downsampling_conv=True,
        upsampling_conv=True,
        concat_skipconn=True,
        scale_factor=4,
        mode: Literal["unet_k3_2d"] = "unet_k3_2d",
        norm_type: Literal["weight", "spectral", "id"] = "id",
    ):
        super().__init__()
        self.in_width = in_width
        self.out_dims = out_width
        norm = dict(
            weight=weight_norm, spectral=spectral_norm, id=lambda x: x
        )[norm_type]
        net = build_block(block_widths[-1], block_depth, mode, norm)
        for i in range(len(block_widths) - 1):
            width = block_widths[-2 - i]
            inner_width = block_widths[-1 - i]
            if downsampling_conv:
                downsampling = norm(
                    nn.Conv2d(
                        width, inner_width, scale_factor, scale_factor, 0
                    )
                )
            else:
                downsampling = nn.Sequential(
                    nn.AvgPool2d(scale_factor, scale_factor),
                    norm(
                        nn.Conv2d(width, inner_width, 1),
                    ),
                )
            if upsampling_conv:
                upsampling = nn.Sequential(
                    nn.Upsample(scale_factor=scale_factor),
                    norm(nn.Conv2d(inner_width, width, 1)),
                )
            else:
                upsampling = norm(
                    nn.ConvTranspose2d(
                        inner_width, width, scale_factor, scale_factor, 0
                    )
                )
            net = nn.Sequential(
                build_block(width, block_depth, mode, norm),
                downsampling,
                net,
                upsampling,
                build_block(width, block_depth, mode, norm),
            )
            if concat_skipconn:
                net = nn.Sequential(
                    ConcatSkipConn(net),
                    norm(nn.Conv2d(width * 2, width, 1)),
                )
            else:
                net = AddSkipConn(net)
        self.net = nn.Sequential(
            norm(nn.Conv2d(in_width, block_widths[0], 3, padding=1)),
            net,
            norm(nn.Conv2d(block_widths[0], out_width, 3, padding=1)),
        )

    def forward(self, x):
        assert x.shape[1] == self.in_width, (
            "%d-dimensional condition is assumed" % self.in_width
        )
        # padding to across spectral dimension to be divisible by 16
        # (max depth assumed to be 4)
        pad = 16 - x.shape[-2] % 16
        shape = x.shape
        padding = torch.zeros((shape[0], shape[1], pad, shape[3])).to(x)
        x1 = torch.cat((x, padding), dim=-2)
        return self.net(x1)[:, :, : x.shape[2]]


class HiFiGeneratorBackbone(torch.nn.Module):
    def __init__(
            self,
            resblock="2",
            upsample_rates=(8, 8, 2, 2),
            upsample_kernel_sizes=(16, 16, 4, 4),
            upsample_initial_channel=128,
            resblock_kernel_sizes=(3, 7, 11),
            resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
            conv_pre_kernel_size=1,
            input_channels=80,
            norm_type: Literal["weight", "spectral"] = "weight",
    ):
        super().__init__()
        self.norm = dict(weight=weight_norm, spectral=spectral_norm)[norm_type]
        self.norm_type = norm_type
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.make_conv_pre(
            input_channels,
            upsample_initial_channel,
            conv_pre_kernel_size
        )

        self.ups = None
        self.resblocks = None
        self.out_channels = self.make_resblocks(
            resblock,
            upsample_rates,
            upsample_kernel_sizes,
            upsample_initial_channel,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
        )

    def make_conv_pre(self, input_channels, upsample_initial_channel, kernel_size):
        assert kernel_size % 2 == 1
        self.conv_pre = self.norm(
            nn.Conv1d(
                input_channels, upsample_initial_channel, kernel_size, 1, padding=kernel_size // 2
            )
        )
        self.conv_pre.apply(init_weights)

    def make_resblocks(
        self,
        resblock,
        upsample_rates,
        upsample_kernel_sizes,
        upsample_initial_channel,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
    ):
        resblock = (
            ResBlock1 if resblock == "1" else ResBlock2
        )

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                self.norm(
                    nn.ConvTranspose1d(
                        upsample_initial_channel // (2 ** i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        ch = None
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(
                    resblock(ch, k, d, norm_type=self.norm_type)
                )
        self.ups.apply(init_weights)
        return ch

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        return x


class SpectralUNet(nn.Module):
    def __init__(
            self,
            block_widths=(8, 16, 24, 32, 64),
            block_depth=5,
            positional_encoding=True,
            norm_type: Literal["weight", "spectral"] = "weight",
    ):
        super().__init__()
        self.positional_encoding = positional_encoding
        self.norm_type = norm_type
        norm = dict(weight=weight_norm, spectral=spectral_norm)[norm_type]

        self.learnable_mel2linspec = norm(nn.Conv1d(80, 513, 1))

        in_width = int(positional_encoding) + 2

        # out_width could be 1 and self.post_conv_2d could be not used here,
        # but both were left for backward compatibility
        # with the other hypotheses we tested
        out_width = block_widths[0]

        self.net = MultiScaleResnet2d(
            block_widths,
            block_depth,
            scale_factor=2,
            in_width=in_width,
            out_width=out_width,
            norm_type=norm_type,
        )

        self.post_conv_2d = nn.Sequential(
            norm(nn.Conv2d(out_width, 1, 1, padding=0)),
        )

        self.post_conv_1d = nn.Sequential(
            norm(nn.Conv1d(513, 128, 1, 1, padding=0)),
        )

        self.mel2lin = None
        self.calculate_mel2lin_matrix()

    def calculate_mel2lin_matrix(self):
        mel_np = librosa_mel_fn(
            sr=16000, n_fft=1024, n_mels=80, fmin=0, fmax=8000
        )
        slices = [
            (np.where(row)[0].min(), np.where(row)[0].max() + 1)
            for row in mel_np
        ]
        slices = [x[0] for x in slices] + [slices[-1][1]]
        mel2lin = np.zeros([81, 513])
        for i, x1, x2 in zip(range(80), slices[:-1], slices[1:]):
            mel2lin[i, x1: x2 + 1] = np.linspace(1, 0, x2 - x1 + 1)
            mel2lin[i + 1, x1: x2 + 1] = np.linspace(0, 1, x2 - x1 + 1)
        mel2lin = mel2lin[1:]
        mel2lin = torch.from_numpy(mel2lin.T).float()
        self.mel2lin = mel2lin

    def mel2linspec(self, mel):
        print('mel2lin', self.mel2lin.shape)
        # torch.Size([513, 80])
        return torch.matmul(self.mel2lin.to(mel), mel)

    def forward(self, mel):
        print('initial', mel.shape)
        # initial torch.Size([4, 80, 512])

        linspec_approx = self.mel2linspec(mel)


        print('linspec_approx', linspec_approx.shape)
        # linspec_approx torch.Size([4, 513, 512])

        linspec_conv_approx = self.learnable_mel2linspec(mel)
        print('linspec_conv_approx', linspec_conv_approx.shape)
        # linspec_conv_approx torch.Size([4, 513, 512])

        linspec_conv_approx = linspec_conv_approx.view(
            mel.shape[0], 1, -1, mel.shape[2]
        )
        print('linspec_conv_approx', linspec_conv_approx.shape)
        # linspec_conv_approx torch.Size([4, 1, 513, 512])


        net_input = linspec_approx.view(
            linspec_approx.shape[0], 1, -1, linspec_approx.shape[2]
        )
        print('net_input', net_input.shape)
        # net_input torch.Size([4, 1, 513, 512])

        if self.positional_encoding:
            pos_enc = torch.linspace(0, 1, 513)[..., None].expand(
                net_input.shape[0], 1, net_input.shape[2], net_input.shape[3]
            )
            print('pos_enc', pos_enc.shape)
            # pos_enc torch.Size([4, 1, 513, 512])

            net_input = torch.cat((net_input, pos_enc.to(net_input)), dim=1)
            print('net_input', net_input.shape)
            # net_input torch.Size([4, 2, 513, 512])

        net_input = torch.cat((net_input, linspec_conv_approx), dim=1)
        print('net input', net_input.shape)
        #net input torch.Size([4, 3, 513, 512]) - right

        out = self.net(net_input)
        print('net output', out.shape)
        # net output torch.Size([4, 8, 513, 512]) -right


        out = self.post_conv_2d(out).squeeze(1)
        print('post_conv_2d out', out.shape)
        # post_conv_2d out torch.Size([4, 513, 512]) -right

        out = self.post_conv_1d(out)
        print('post_conv_1d out', out.shape)
        # post_conv_1d out torch.Size([4, 128, 512]) -right


        return out


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_output)  # Residual connection
        x = self.norm(x)  # Normalization
        return x


class SpectralAttentionUNet(nn.Module):
    def __init__(
            self,
            embed_dim=256,
            num_heads=8,
            depth=2,
            positional_encoding=True,
    ):
        super().__init__()
        self.positional_encoding = positional_encoding
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depth = depth

        # self.learnable_mel2linspec = nn.Linear(80, 513)  # Linear projection from mel to embedding space
        norm = dict(weight=weight_norm, spectral=spectral_norm)['weight']
        self.learnable_mel2linspec = norm(nn.Conv1d(80, 513, 1))


        # Define attention-based encoder-decoder layers
        self.encoder = nn.ModuleList([
            AttentionBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.decoder = nn.ModuleList([
            AttentionBlock(embed_dim, num_heads) for _ in range(depth)
        ])

        # self.output_layer = nn.Sequential(
            # nn.Linear(256, 128),  # Reduce embedding dimension to final output
        # )

        self.post_conv_1d = nn.Sequential(
            norm(nn.Conv1d(self.embed_dim, 128, 1, 1, padding=0)),
        )
        self.mel2lin = None
        self.calculate_mel2lin_matrix()

    def calculate_mel2lin_matrix(self):
        mel_np = librosa_mel_fn(
            sr=16000, n_fft=1024, n_mels=80, fmin=0, fmax=8000
        )
        slices = [
            (np.where(row)[0].min(), np.where(row)[0].max() + 1)
            for row in mel_np
        ]
        slices = [x[0] for x in slices] + [slices[-1][1]]
        mel2lin = np.zeros([81, 513])
        for i, x1, x2 in zip(range(80), slices[:-1], slices[1:]):
            mel2lin[i, x1: x2 + 1] = np.linspace(1, 0, x2 - x1 + 1)
            mel2lin[i + 1, x1: x2 + 1] = np.linspace(0, 1, x2 - x1 + 1)
        mel2lin = mel2lin[1:]
        mel2lin = torch.from_numpy(mel2lin.T).float()
        self.mel2lin = mel2lin
        

    def mel2linspec(self, mel):
        print('self.mel2lin', self.mel2lin.shape)
        # torch.Size([513, 80])
        return torch.matmul(self.mel2lin.to(mel), mel)

    def forward(self, mel):
        print('initial shape', mel.shape)
        # initial shape torch.Size([4, 80, 256])
        linspec_approx = self.mel2linspec(mel)
        print('linspec_approx', linspec_approx.shape)
        # torch.Size([4, 513, 256])

        linspec_conv_approx = self.learnable_mel2linspec(mel)
        linspec_conv_approx = linspec_conv_approx.view(
            mel.shape[0], 1, -1, mel.shape[2]
        )

        net_input = linspec_approx.view(
            linspec_approx.shape[0], 1, -1, linspec_approx.shape[2]
        )
        if self.positional_encoding:
            pos_enc = torch.linspace(0, 1, 513)[..., None].expand(
                net_input.shape[0], 1, net_input.shape[2], net_input.shape[3]
            )
            net_input = torch.cat((net_input, pos_enc.to(net_input)), dim=1)
        net_input = torch.cat((net_input, linspec_conv_approx), dim=1)
        print('net_input', net_input.shape)
        #net_input torch.Size([4, 3, 513, 256])

        
        batch_size, num_channels, freq_bins, seq_len = net_input.shape
        
        net_input = net_input.view(batch_size, seq_len, num_channels * freq_bins)
        print('net_input after view', net_input.shape)
        # torch.Size([4, 256, 1539])
        if net_input.size(-1) != self.embed_dim:
            net_input = nn.Linear(net_input.size(-1), self.embed_dim)(net_input)
            print('net_input after linear', net_input.shape)
            # torch.Size([4, 256, 512])
        linspec_embed = net_input
    
        
        for layer in self.encoder:
            linspec_embed = layer(linspec_embed)
        for layer in self.decoder:
            linspec_embed = layer(linspec_embed)
        print('linspec_embed', linspec_embed.shape)
        # torch.Size([4, 256, 512])
        # out = self.output_layer(linspec_embed.transpose(1, 2)).transpose(1, 2)
        out = self.post_conv_1d(linspec_embed.transpose(1, 2))
        print('out', out.shape)

        return out


class SpectralMaskNet(nn.Module):
    def __init__(
        self,
        in_ch=8,
        act="softplus",
        block_widths=(8, 12, 24, 32),
        block_depth=1,
        norm_type: Literal["weight", "spectral", "id"] = "id"
    ):
        super().__init__()
        self.net = MultiScaleResnet2d(
            block_widths,
            block_depth,
            scale_factor=2,
            in_width=in_ch,
            out_width=in_ch,
            norm_type=norm_type
        )
        if act == "softplus":
            self.act = nn.Softplus()
        else:
            self.act = nn.ReLU()

    def forward(self, x):
        n_fft = 1024
        win_length = n_fft
        hop_length = n_fft // 4
        f_hat = torch.stft(
            x.view(x.shape[0] * x.shape[1], -1),
            n_fft=n_fft,
            center=True,
            hop_length=hop_length,
            window=torch.hann_window(
                window_length=win_length, device=x.device
            ),
            return_complex=False,
        )

        f = (f_hat[:, 1:, 1:].pow(2).sum(-1) + 1e-9).sqrt()

        padding = (
            int(math.ceil(f.shape[-1] / 8.0)) * 8 - f.shape[-1]
        )  # (2**(int(math.ceil(math.log2(f.shape[-1])))) - f.shape[-1]) // 2
        padding_right = padding // 2
        padding_left = padding - padding_right
        f = torch.nn.functional.pad(f, (padding_left, padding_right))

        mult_factor = self.act(
            self.net(f.view(x.shape[0], -1, f.shape[1], f.shape[2]))
        )  # [..., padding_left:-padding_right]
        if padding_right != 0:
            mult_factor = mult_factor[..., padding_left:-padding_right]
        else:
            mult_factor = mult_factor[..., padding_left:]

        mult_factor = mult_factor.reshape(
            (
                mult_factor.shape[0] * mult_factor.shape[1],
                mult_factor.shape[2],
                mult_factor.shape[3],
            )
        )[..., None]

        one_padded_mult_factor = torch.ones_like(f_hat)
        one_padded_mult_factor[:, 1:, 1:] *= mult_factor

        f_hat = torch.view_as_complex(f_hat * one_padded_mult_factor)
        y = torch.istft(
            f_hat,
            n_fft=n_fft,
            hop_length=hop_length,
            window=torch.hann_window(
                window_length=win_length, device=x.device
            ),
        )
        return y.view(x.shape[0], x.shape[1], -1)
