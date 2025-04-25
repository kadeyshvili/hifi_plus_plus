from pathlib import Path
from torch.utils.data import Dataset
import torchaudio
import random
from tqdm import tqdm
from src.datasets.base_dataset import BaseDataset
import os
import librosa
import torch
import scipy
import numpy as np
from librosa.util import normalize


from src.model.melspec import MelSpectrogramConfig, MelSpectrogram


# class HiFiGanDataset(BaseDataset):
#     def __init__(self, data_path, limit=None, max_len=22528, **kwargs):
#         data_path = Path(data_path)
#         self.mel_creator = MelSpectrogram(MelSpectrogramConfig())
#         self.wavs_and_paths = []
#         self.max_len = max_len
#         for file_path in tqdm((data_path).iterdir(), desc='Loading files'):
#             wav, sr = torchaudio.load(file_path)
#             if sr != 22050:
#                 wav = torchaudio.functional.resample(wav, sr, 22050)
#             wav = wav[0:1, :]
#             path = file_path
#             self.wavs_and_paths.append({'wav' : wav, 'path' : path})
#         if limit is not None:
#             self.wavs_and_paths = self.wavs_and_paths[:limit]

#     def __len__(self):
#         return len(self.wavs_and_paths)

#     def __getitem__(self, idx):
#         wav = self.wavs_and_paths[idx]['wav']
#         path = self.wavs_and_paths[idx]['path']
#         if self.max_len is not None:
#             start = random.randint(0,  wav.shape[-1] - self.max_len)
#             wav = wav[:, start : start + self.max_len]
#         melspec = self.mel_creator(wav.detach()).squeeze(0)
#         return {"wav": wav, 'path' : path, 'melspec' : melspec}


def get_dataset_filelist(dataset_split_file, input_wavs_dir):
    with open(dataset_split_file, "r", encoding="utf-8") as fi:
        files = [
            os.path.join(input_wavs_dir, fn)
            for fn in fi.read().split("\n")
            if len(fn) > 0
        ]
    return files



def low_pass_filter(audio: np.ndarray, max_freq,
                    lp_type="default", orig_sr=16000):
    if lp_type == "default":
        tmp = librosa.resample(
            audio, orig_sr=orig_sr, target_sr=max_freq * 2, res_type="polyphase"
        )
    elif lp_type == "decimate":
        sub = orig_sr / (max_freq * 2)
        assert int(sub) == sub
        tmp = scipy.signal.decimate(audio, int(sub))
    else:
        raise NotImplementedError
    # soxr_hq is faster and better than polyphase,
    # but requires additional libraries installed
    # the speed difference is only 4 times, we can live with that
    tmp = librosa.resample(tmp, orig_sr=max_freq * 2, target_sr=16000, res_type="polyphase")
    return tmp[: audio.size]



def split_audios(audios, segment_size, split):
    audios = [torch.FloatTensor(audio).unsqueeze(0) for audio in audios]
    if split:
        if audios[0].size(1) >= segment_size:
            max_audio_start = audios[0].size(1) - segment_size
            audio_start = random.randint(0, max_audio_start)
            audios = [
                audio[:, audio_start : audio_start + segment_size]
                for audio in audios
            ]
        else:
            audios = [
                torch.nn.functional.pad(
                    audio,
                    (0, segment_size - audio.size(1)),
                    "constant",
                )
                for audio in audios
            ]
    audios = [audio.squeeze(0).numpy() for audio in audios]
    return audios

class VCTKDataset(BaseDataset):
    def __init__(
        self,
        dataset_split_file,
        vctk_wavs_dir,
        segment_size=8192,
        sampling_rate=16000,
        split=True,
        shuffle=False,
        device=None,
        input_freq=None,
        lowpass="default",
    ):
        self.audio_files = get_dataset_filelist(dataset_split_file,
                                                vctk_wavs_dir)
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.device = device
        self.input_freq = input_freq
        self.lowpass = lowpass
        self.clean_wavs_dir = vctk_wavs_dir
        self.mel_creator = MelSpectrogram(MelSpectrogramConfig())

    def __getitem__(self, index):
        vctk_fn = self.audio_files[index]

        vctk_audio = librosa.load(
            vctk_fn,
            sr=self.sampling_rate,
            res_type="polyphase",
        )[0]
        (vctk_audio, ) = split_audios([vctk_audio],
                                      self.segment_size, self.split)

        lp_inp = low_pass_filter(
            vctk_audio, self.input_freq,
            lp_type=self.lowpass, orig_sr=self.sampling_rate
        )
        input_audio = normalize(lp_inp)[None] * 0.95
        assert input_audio.shape[1] == vctk_audio.size

        input_audio = torch.FloatTensor(input_audio)
        # audio = torch.FloatTensor(normalize(vctk_audio) * 0.95)
        # audio = audio.unsqueeze(0)
        # input_audio = torch.nn.functional.pad(input_audio, (0, self.segment_size - input_audio.size(1)), 'constant')
        # audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')
        melspec = self.mel_creator(input_audio.detach()).squeeze(0)

        return {"wav": input_audio, 'path' : vctk_fn, 'melspec' : melspec}
        # return input_audio, audio

    def __len__(self):
        return len(self.audio_files)


