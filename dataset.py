# -*- coding: utf-8 -*-
import random
import torch
import torch.nn.functional as F
import librosa
import torchaudio.transforms as T
from glob import glob
from torch.utils.data import Dataset


class VCTK092Dataset(Dataset):
    def __init__(self, dtype, mode, lr=16000, hr=48000):
        super(VCTK092Dataset, self).__init__()
        self.hr, self.lr = hr, lr
        self.rate = hr // lr
        self.dtype = dtype
        self.mode = mode
        self.file_names = []

        assert dtype in ['single', 'multi']
        assert mode in ['train', 'val', 'test']

        if dtype == 'single':
            if mode == 'train':
                self.file_names += glob('dataset/VCTK-Corpus-0.92/wav48_silence_trimmed/p225/*mic1.flac')[:-16]
            elif mode == 'val':
                self.file_names += glob('dataset/VCTK-Corpus-0.92/wav48_silence_trimmed/p225/*mic1.flac')[-16:-8]
            elif mode == 'test':
                self.file_names += glob('dataset/VCTK-Corpus-0.92/wav48_silence_trimmed/p225/*mic1.flac')[-8:]

        elif dtype == 'multi':
            dir_names = glob('dataset/VCTK-Corpus-0.92/wav48_silence_trimmed/*/')
            dir_names = list(filter(lambda x: 'p280' not in x and 'p315' not in x, dir_names))
            if mode == 'train':
                for directory in dir_names[:-16]:
                    self.file_names += glob(f'{directory}*mic1.flac')
            elif mode == 'val':
                for directory in dir_names[-16:-8]:
                    self.file_names += glob(f'{directory}*mic1.flac')
            elif mode == 'test':
                for directory in dir_names[-8:]:
                    self.file_names += glob(f'{directory}*mic1.flac')

        n_fft = 1024
        win_length = None
        hop_length = 256

        self.spec = T.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            power=2.0,
        )

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        raw_wave_hr = librosa.load(file_name, sr=self.hr)[0]

        res_type = random.choice(['linear', 'sinc_best', 'soxr_qq', 'soxr_hq']) if self.mode == 'train' else 'sinc_best'
        raw_wave_lr = librosa.resample(y=raw_wave_hr, orig_sr=self.hr, target_sr=self.lr, res_type=res_type)

        if self.mode == 'train':
            max_length = 15000 // self.rate
            lr_length = len(raw_wave_lr)
            start = random.randrange(max(1, lr_length - max_length))

            wave_lr = torch.Tensor(raw_wave_lr[start:start + max_length])
            wave_lr = F.pad(wave_lr, (0, max(0, max_length - len(wave_lr))), 'constant', 0)
            wave_hr = torch.Tensor(raw_wave_hr[self.rate * start:self.rate * (start + max_length)])
            wave_hr = F.pad(wave_hr, (0, max(0, self.rate * max_length - len(wave_hr))), 'constant', 0)

            # add noise
            noise_std, noise_mean = 0.05, 0.0
            wave_lr += torch.randn(wave_lr.size()) * noise_std + noise_mean

        else:
            wave_lr = torch.Tensor(raw_wave_lr)
            wave_hr = torch.Tensor(raw_wave_hr)

        spec_lr = self.spec(wave_lr)
        return wave_lr, wave_hr, spec_lr

    def __len__(self):
        return len(self.file_names)
