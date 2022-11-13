# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from glob import glob

import soundfile as sf

from dataset import VCTK092Dataset
from model.audiosr import AudioSR
from utils import SNR, LSD

device = 'cuda' if torch.cuda.is_available() else 'cpu'

config_files = glob('./config/*.json')
print("------- Configuration --------")
for i in range(len(config_files)):
    print(f'{i}. {os.path.basename(config_files[i])}')
config_file = config_files[int(input("Choose Configuration: "))]

with open(config_file, 'r') as f:
    config = json.load(f)

LR_sample_rate = config['LR_sample_rate']
HR_sample_rate = config['HR_sample_rate']
rate = HR_sample_rate // LR_sample_rate

dtype = config['dtype']
test_dataset = VCTK092Dataset(dtype=dtype, mode='test', lr=LR_sample_rate, hr=HR_sample_rate)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1)

root = Path('./')
audio_dir = root / 'audio' / dtype / f'x{rate}'
audio_dir.mkdir(parents=True, exist_ok=True)

model = AudioSR(rate).to(device)

ckpt_root = root / 'model' / dtype / f'x{rate}'
best_path = ckpt_root / 'best.pth'
best_val_loss = np.inf
last_epoch = 0
if os.path.exists(best_path):
    ckpt = torch.load(best_path, map_location=torch.device(device))
    model.load_state_dict(ckpt['model'])
    last_epoch = ckpt['epoch']
    best_val_loss = ckpt['val_loss']
    print('Best checkpoint is loaded.')
    print(f'Best Epoch: {ckpt["epoch"]} |',
          f'Best Avg Train Loss: {ckpt["train_loss"]} |',
          f'Best Avg Val Loss: {ckpt["val_loss"]}')
else:
    print('No checkpoint is found.')

model.eval()
idx = 0
snr = 0.
lsd = 0.
with torch.no_grad():
    for lr_wave, hr_wave, lr_spec in test_loader:
        lr_wave = lr_wave.to(device)
        hr_wave = hr_wave.to(device)
        lr_spec = lr_spec.to(device)
        sr_wave = model(lr_wave, lr_spec)[:, :len(hr_wave[0])]

        snr += SNR(sr_wave, hr_wave)
        lsd += LSD(sr_wave, hr_wave)

        lr = lr_wave.to('cpu')
        hr = hr_wave.to('cpu')
        sr = sr_wave.to('cpu')

        if idx < 4:
            lr_path = audio_dir / f'lr{idx}.wav'
            hr_path = audio_dir / f'hr{idx}.wav'
            sr_path = audio_dir / f'sr{idx}.wav'

            sf.write(f'{lr_path}', lr[0].detach().numpy(), samplerate=LR_sample_rate, format='wav', subtype='PCM_16')
            sf.write(f'{hr_path}', hr[0].detach().numpy(), samplerate=HR_sample_rate, format='wav', subtype='PCM_16')
            sf.write(f'{sr_path}', sr[0].detach().numpy(), samplerate=HR_sample_rate, format='wav', subtype='PCM_16')
        idx += 1

print(f'SNR: {float(snr) / len(test_dataset):.3g}')
print(f'LSD: {float(lsd) / len(test_dataset):.3g}')
