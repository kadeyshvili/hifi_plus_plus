from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm
from librosa.filters import mel as librosa_mel_fn
import numpy as np
import src.utils.upsampling_utils as upsampling_utils
import librosa



mel_basis = {}
hann_window = {}

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def stft(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False,):
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).unsqueeze(0)

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels,
                             fmin=fmin, fmax=fmax)
        mel_basis[str(fmax) + "_" + str(y.device)] = (
            torch.from_numpy(mel).float().to(y.device)
        )
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    freq_and_time  = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[str(y.device)],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    return torch.abs(freq_and_time)



def mel_spectrogram(
    y,
    n_fft,
    num_mels,
    sampling_rate,
    hop_size,
    win_size,
    fmin,
    fmax,
    center=False,
    return_mel_and_spec=False,
):
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).unsqueeze(0)

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels,
                             fmin=fmin, fmax=fmax)
        mel_basis[str(fmax) + "_" + str(y.device)] = (
            torch.from_numpy(mel).float().to(y.device)
        )
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[str(y.device)],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)

    mel = torch.matmul(mel_basis[str(fmax) + "_" + str(y.device)], spec)
    mel = spectral_normalize_torch(mel)
    result = mel.squeeze()

    if return_mel_and_spec:
        spec = spectral_normalize_torch(spec)
        return result, spec
    else:
        return result


def closest_power_of_two(n):
    return 1 << (n - 1).bit_length()



class HiFiPlusGenerator(torch.nn.Module):
    def __init__(
        self,
        hifi_resblock="1",
        hifi_upsample_rates=(8, 8, 2, 2),
        hifi_upsample_kernel_sizes=(16, 16, 4, 4),
        hifi_upsample_initial_channel=128,
        hifi_resblock_kernel_sizes=(3, 7, 11),
        hifi_resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        hifi_input_channels=128,
        hifi_conv_pre_kernel_size=1,

        upsample_block_rates=[2, 2],
        upsample_init_channels = 1,
        upsample_block_kernel_sizes=[4, 4], 


        residual_channels=64,
        bsft_channels=64,

        use_spectralunet=True,
        spectralunet_block_widths=(8, 16, 24, 32, 64),
        spectralunet_block_depth=5,
        spectralunet_positional_encoding=True,

        use_waveunet=True,
        waveunet_block_widths=(10, 20, 40, 80),
        waveunet_block_depth=4,

        use_spectralmasknet=True,
        spectralmasknet_block_widths=(8, 12, 24, 32),
        spectralmasknet_block_depth=4,

        norm_type: Literal["weight", "spectral"] = "weight",
        use_skip_connect=True,
        waveunet_before_spectralmasknet=True,
    ):
        super().__init__()
        self.norm = dict(weight=weight_norm, spectral=spectral_norm)[norm_type]
        self.norm_type = norm_type

        self.use_spectralunet = use_spectralunet
        self.use_waveunet = use_waveunet
        self.use_spectralmasknet = use_spectralmasknet

        self.use_skip_connect = use_skip_connect
        self.waveunet_before_spectralmasknet = waveunet_before_spectralmasknet
        self.upsampling_block1 = upsampling_utils.UpsampleTwice(upsample_init_channels, upsample_block_rates, upsample_block_kernel_sizes)
        self.upsampling_block2 = upsampling_utils.UpsampleTwice(upsample_init_channels, upsample_block_rates, upsample_block_kernel_sizes)
        self.nw_block1 = upsampling_utils.NUWaveBlock(residual_channels, bsft_channels)
        self.nw_block2 = upsampling_utils.NUWaveBlock(residual_channels, bsft_channels)

        self.hifi = upsampling_utils.HiFiUpsampling(
            resblock=hifi_resblock,
            upsample_initial_channel=hifi_upsample_initial_channel,
            resblock_kernel_sizes=hifi_resblock_kernel_sizes,
            resblock_dilation_sizes=hifi_resblock_dilation_sizes,
            input_channels=hifi_input_channels,
            conv_pre_kernel_size=hifi_conv_pre_kernel_size,
            norm_type=norm_type,
        )
        ch = self.hifi.out_channels

        if self.use_spectralunet:
            self.spectralunet2 = upsampling_utils.SpectralUNet3(
                block_widths=spectralunet_block_widths,
                block_depth=spectralunet_block_depth,
                positional_encoding=spectralunet_positional_encoding,
                norm_type=norm_type,
            )

        if self.use_waveunet:
            self.waveunet = upsampling_utils.MultiScaleResnet(
                waveunet_block_widths,
                waveunet_block_depth,
                mode="waveunet_k5",
                out_width=ch,
                in_width=ch,
                norm_type=norm_type
            )

        if self.use_spectralmasknet:
            self.spectralmasknet = upsampling_utils.SpectralMaskNet(
                in_ch=ch,
                block_widths=spectralmasknet_block_widths,
                block_depth=spectralmasknet_block_depth,
                norm_type=norm_type
            )

        self.waveunet_skip_connect = None
        self.spectralmasknet_skip_connect = None
        if self.use_skip_connect:
            self.make_waveunet_skip_connect(ch)
            self.make_spectralmasknet_skip_connect(ch)

        self.conv_post = None
        self.make_conv_post(ch)

    def make_waveunet_skip_connect(self, ch):
        self.waveunet_skip_connect = self.norm(nn.Conv1d(ch, ch, 1, 1))
        self.waveunet_skip_connect.weight.data = torch.eye(ch, ch).unsqueeze(-1)
        self.waveunet_skip_connect.bias.data.fill_(0.0)

    def make_spectralmasknet_skip_connect(self, ch):
        self.spectralmasknet_skip_connect = self.norm(nn.Conv1d(ch, ch, 1, 1))
        self.spectralmasknet_skip_connect.weight.data = torch.eye(ch, ch).unsqueeze(-1)
        self.spectralmasknet_skip_connect.bias.data.fill_(0.0)

    def make_conv_post(self, ch):
        self.conv_post = self.norm(nn.Conv1d(ch, 1, 7, 1, padding=3))
        self.conv_post.apply(upsampling_utils.init_weights)

    def apply_spectralunet2(self, x_reference):
        if self.use_spectralunet:
            orig_length = x_reference.shape[-1]
            pad_size = (
                closest_power_of_two(orig_length) - orig_length
            )
            if pad_size > 0:
                x = torch.nn.functional.pad(x_reference, (0, pad_size))
            else:
                x = x_reference
        
            x_mag = self.spectralunet2(x)
            x_mag = x_mag[..., :orig_length]
        else:
            x = x_reference.squeeze(1)
        return x_mag

    def apply_waveunet(self, x):
        x_a = x
        x = self.waveunet(x_a)
        if self.use_skip_connect:
            x += self.waveunet_skip_connect(x_a)
        return x

    def apply_spectralmasknet(self, x):
        x_a = x
        x = self.spectralmasknet(x)
        if self.use_skip_connect:
            x += self.spectralmasknet_skip_connect(x_a)
        return x

    def forward(self, x_reference):
        x = self.apply_spectralunet2(x_reference)
        x = self.hifi(x)
        if self.use_waveunet and self.waveunet_before_spectralmasknet:
            x = self.apply_waveunet(x)
        if self.use_spectralmasknet:
            x = self.apply_spectralmasknet(x)
        if self.use_waveunet and not self.waveunet_before_spectralmasknet:
            x = self.apply_waveunet(x)

        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

class A2AHiFiPlusGeneratorV4(HiFiPlusGenerator):
    def __init__(
        self,
        hifi_resblock="1",
        hifi_upsample_rates=(8, 8, 2, 2),
        hifi_upsample_kernel_sizes=(16, 16, 4, 4),
        hifi_upsample_initial_channel=128,
        hifi_resblock_kernel_sizes=(3, 7, 11),
        hifi_resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        hifi_input_channels=128,
        hifi_conv_pre_kernel_size=1,

        upsample_init_channels = 1,
        upsample_block_rates=[2, 2],
        upsample_block_kernel_sizes=[4, 4], 

        use_spectralunet=True,
        spectralunet_block_widths=(8, 16, 24, 32, 64),
        spectralunet_block_depth=5,
        spectralunet_positional_encoding=True,

        use_waveunet=True,
        waveunet_block_widths=(10, 20, 40, 80),
        waveunet_block_depth=4,

        use_spectralmasknet=True,
        spectralmasknet_block_widths=(8, 12, 24, 32),
        spectralmasknet_block_depth=4,

        norm_type: Literal["weight", "spectral"] = "weight",
        use_skip_connect=True,
        waveunet_before_spectralmasknet=True,

        waveunet_input: Literal["waveform", "hifi", "both"] = "both",
    ):
        super().__init__(
            hifi_resblock=hifi_resblock,
            hifi_upsample_rates=hifi_upsample_rates,
            hifi_upsample_kernel_sizes=hifi_upsample_kernel_sizes,
            hifi_upsample_initial_channel=hifi_upsample_initial_channel,
            hifi_resblock_kernel_sizes=hifi_resblock_kernel_sizes,
            hifi_resblock_dilation_sizes=hifi_resblock_dilation_sizes,
            hifi_input_channels=hifi_input_channels,
            hifi_conv_pre_kernel_size=hifi_conv_pre_kernel_size,

            upsample_init_channels=upsample_init_channels,
            upsample_block_rates=upsample_block_rates,
            upsample_block_kernel_sizes=upsample_block_kernel_sizes,

            use_spectralunet=use_spectralunet,
            spectralunet_block_widths=spectralunet_block_widths,
            spectralunet_block_depth=spectralunet_block_depth,
            spectralunet_positional_encoding=spectralunet_positional_encoding,

            use_waveunet=use_waveunet,
            waveunet_block_widths=waveunet_block_widths,
            waveunet_block_depth=waveunet_block_depth,

            use_spectralmasknet=use_spectralmasknet,
            spectralmasknet_block_widths=spectralmasknet_block_widths,
            spectralmasknet_block_depth=spectralmasknet_block_depth,

            norm_type=norm_type,
            use_skip_connect=use_skip_connect,
            waveunet_before_spectralmasknet=waveunet_before_spectralmasknet,
        )

        self.waveunet_input = waveunet_input

        self.waveunet_conv_pre = None
        if self.waveunet_input == "waveform":
            self.waveunet_conv_pre = weight_norm(
                nn.Conv1d(
                    1, self.hifi.out_channels, 1
                )
            )
        elif self.waveunet_input == "both":
            self.waveunet_conv_pre = weight_norm(
                nn.Conv1d(
                    1 + self.hifi.out_channels, self.hifi.out_channels, 1
                )
            )
        
    @staticmethod
    def get_melspec(x):
        shape = x.shape
        x = x.view(shape[0] * shape[1], shape[2])
        x = mel_spectrogram(x, 1024, 80, 16000, 256, 1024, 0, 8000)
        x = x.view(shape[0], -1, x.shape[-1])
        return x
    

    @staticmethod
    def get_stft(x):
        shape = x.shape
        x = x.view(shape[0] * shape[1], shape[2])
        x = stft(x, 1024, 80, 2000, 256, 1024, 0, 2000)
        x = x.view(shape[0], -1, x.shape[-1])
        return x
    
    
    
    def apply_waveunet_a2a(self, x, x_reference):
        if self.waveunet_input == "waveform":
            x_a = self.waveunet_conv_pre(x_reference)
        elif self.waveunet_input == "both":
            x_a = torch.cat([x, x_reference], 1)
            x_a = self.waveunet_conv_pre(x_a)
        elif self.waveunet_input == "hifi":
            x_a = x
        else:
            raise ValueError
        x = self.waveunet(x_a)
        if self.use_skip_connect:
            x += self.waveunet_skip_connect(x_a)
        return x

    def forward(self, x, x_reference, initial_sr, target_sr):
        initial_x = x.clone()
        batch_size = x.shape[0]

        current_size = initial_x.shape[-1]
        target_size = (target_sr // initial_sr) * current_size
        closest_size = ((current_size + 1023) // 1024) * 1024
        pad_size =  closest_size - current_size
        padded_x = torch.nn.functional.pad(initial_x, (0, pad_size))
        padded_reference = torch.nn.functional.pad(x_reference, (0, pad_size * (target_sr // initial_sr))).to(x.device)
        resampled_once = []
        for i in range(batch_size):
            x_single = padded_x[i].cpu().numpy()
            x_resampled_once = librosa.resample(
                x_single, orig_sr=initial_sr, target_sr=target_sr // 2, res_type="polyphase"
            )

            target_length_once = x_single.shape[-1] * (target_sr // 2 // initial_sr)
            if len(x_resampled_once) > target_length_once:
                x_resampled_once = x_resampled_once[:target_length_once]
            
            resampled_once.append(x_resampled_once)
        

        x_half_resempled = np.stack(resampled_once)
        x_half_resempled = torch.tensor(x_half_resempled, dtype=padded_x.dtype).to(x.device)

        upsampled_x = self.upsampling_block1(padded_x)


        if initial_sr==4000 and target_sr==16000:

            highcut = initial_sr // 2
            nyq = 0.5 * target_sr // 2
            hi = highcut / nyq
            fft_size = 1024 // 2 + 1
            band4_8 = torch.zeros(fft_size, dtype=torch.float)
            band4_8[:int(hi * fft_size)] = 1
            band4_8 = band4_8.unsqueeze(0).unsqueeze(0) 
            band4_8 = band4_8.repeat(batch_size, 2, 1).to(upsampled_x.device)
            x_4_8 = self.nw_block1(upsampled_x, x_half_resempled, band4_8)


            upsampled_x_4 = self.upsampling_block2(x_4_8)
            highcut = initial_sr // 2 * 2
            nyq = 0.5 * target_sr
            hi = highcut / nyq
            fft_size = 1024 // 2 + 1
            band8_16 = torch.zeros(fft_size, dtype=torch.float)
            band8_16[:int(hi * fft_size)] = 1
            band8_16 = band8_16.unsqueeze(0).unsqueeze(0) 
            band8_16 = band8_16.repeat(batch_size, 2, 1).to(upsampled_x.device)
            x_8_16 = self.nw_block2(upsampled_x_4, padded_reference, band8_16)

        elif initial_sr==4000 and target_sr==8000:
            highcut = initial_sr // 2
            nyq = 0.5 * target_sr // 2
            hi = highcut / nyq
            fft_size = 1024 // 2 + 1
            band4_8 = torch.zeros(fft_size, dtype=torch.float)
            band4_8[:int(hi * fft_size)] = 1
            band4_8 = band4_8.unsqueeze(0).unsqueeze(0) 
            band4_8 = band4_8.repeat(batch_size, 2, 1).to(upsampled_x.device)
            x_8_16 = self.nw_block1(upsampled_x, padded_reference, band4_8)

        elif initial_sr==8000 and target_sr==16000:
            highcut = initial_sr // 2 * 2
            nyq = 0.5 * target_sr
            hi = highcut / nyq
            fft_size = 1024 // 2 + 1
            band8_16 = torch.zeros(fft_size, dtype=torch.float)
            band8_16[:int(hi * fft_size)] = 1
            band8_16 = band8_16.unsqueeze(0).unsqueeze(0) 
            band8_16 = band8_16.repeat(batch_size, 2, 1).to(upsampled_x.device)
            x_8_16 = self.nw_block2(upsampled_x, padded_reference, band8_16)


        x = self.get_stft(x_8_16)
        x = torch.abs(x)

        x = self.apply_spectralunet2(x)
        x = self.hifi(x)
        
        if self.use_waveunet and self.waveunet_before_spectralmasknet:
            x = self.apply_waveunet_a2a(x, padded_reference)
        if self.use_spectralmasknet:
            x = self.apply_spectralmasknet(x)
        if self.use_waveunet and not self.waveunet_before_spectralmasknet:
            x = self.apply_waveunet_a2a(x, padded_reference)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x[..., :target_size]