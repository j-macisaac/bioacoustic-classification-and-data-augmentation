# -*- coding: utf-8 -*-
"""
Created on Tue May 20 18:26:12 2025

@author: jennifer.macisaac
"""

import os, glob
import shutil
import random

base_path = "D:/3_ref_library/2_sliced_wav/3_all_species/"

train_path = f"{base_path}Dataset/train/"
val_path = f"{base_path}Dataset/val/"
test_path = f"{base_path}Dataset/test/"

species = os.listdir(train_path)

new_path = f"{base_path}Dataset_corrected/"
os.makedirs(new_path, exist_ok=True)

for sp in species:
    os.makedirs(f"{new_path}train/{sp}", exist_ok=True)
    files = glob.glob(
        f"{train_path}{sp}/*.wav") + glob.glob(
            f"{val_path}{sp}/*.wav") + glob.glob(f"{test_path}{sp}/*.wav") 
    for file in files:
        shutil.copy(file, f"{new_path}train/{sp}")

for sp in species:
    os.makedirs(f"{new_path}val/{sp}", exist_ok=True)
    train_files = glob.glob(f"{new_path}train/{sp}/*.wav")
    val_dir = f"{new_path}val/{sp}"
    samples = random.sample(train_files, 50)
    for sample in samples:
        shutil.move(sample, val_dir)

for sp in species:
    os.makedirs(f"{new_path}test/{sp}")
    train_files = glob.glob(f"{new_path}train/{sp}/*.wav")
    test_dir = f"{new_path}test/{sp}"
    samples = random.sample(train_files, 100)
    for sample in samples:
        shutil.move(sample, test_dir)
        
for sp in species:
    train = len(os.listdir(f"{new_path}/train/{sp}"))
    val = len(os.listdir(f"{new_path}/val/{sp}"))
    test = len(os.listdir(f"{new_path}/test/{sp}"))
    #aug = len(os.listdir(f"{files_loc}/aug/{sp}"))
    print(f"No. {sp} files: train = {train}, val = {val}, test = {test}")
    #print(f"{sp}: train - {train}, aug - {aug} total = {(train + aug)}"

## augmenting files:
    

import math
import numpy as np
import librosa as lb
import soundfile as sf
from audiomentations import BandStopFilter, Shift, TimeMask

# variables 

aug_effects =['tshift', 'ts_noise', 'ts_bandsf', 'ts_tmask', 'ts_bsf_noise']

# data paths

train_dir = f"{new_path}/train/"
species = os.listdir(train_dir)


# augmentation functions

class augmentation:
        
    def noise_ran():
        noise_path = f"{base_path}/Bg_noise_2s/"
        noise_files = os.listdir(noise_path)
        ran_file = noise_files[np.random.randint(0, len(noise_files))]
        noise, sr = lb.load(noise_path + ran_file, sr=256000)
        return noise
        
        
    def noise(audio, sr):
        noise_clip = augmentation.noise_ran()
    
        if len(noise_clip) >= len(audio):
            max_start = len(noise_clip) - len(audio)
            start_ = np.random.randint(0, max_start + 1) if max_start > 0 else 0
            ns_slice = noise_clip[start_: start_ + len(audio)]
        else:
            # If noise is shorter, repeat it to cover the length of the audio
            repeats = int(np.ceil(len(audio) / len(noise_clip)))
            ns_slice = np.tile(noise_clip, repeats)[:len(audio)]
    
        call = (
            audio * np.random.uniform(0.8, 1.2)
            + ns_slice * np.random.uniform(0, 0.1)
        )
        return call

 
    
    
    def tshift(audio, sr):
        timeshift = Shift(min_fraction=-0.25, max_fraction=0.25, 
                          rollover=True, p=1)
        call = timeshift(audio, sr)
        effect = "tshift"
        return call, effect
    
    
    def ts_tmask(audio, sr):
        timeshift = Shift(min_fraction=-0.25, max_fraction=0.25, 
                          rollover=True, p=1)
        ts_call = timeshift(audio, sr)
        tmask = TimeMask(min_band_part=0.02, max_band_part=0.05, 
                         fade=False, p=1)
        call = tmask(ts_call, sr)
        effect = "ts_tmask"
        return call, effect
    
    
    def ts_bandsf(audio, sr):
        timeshift = Shift(min_fraction=-0.25, max_fraction=0.25, 
                          rollover=True, p=1)
        ts_call = timeshift(audio, sr)
        bandsf = BandStopFilter(min_center_freq=1000, max_center_freq=100000,
            min_bandwidth_fraction=0.5, max_bandwidth_fraction=1, p=1)
        call = bandsf(ts_call, sr)
        effect = "ts_bandsf"
        return call, effect    
    
    def ts_noise(audio, sr):
        timeshift = Shift(min_fraction=-0.25, max_fraction=0.25, 
                          rollover=True, p=1)
        ts_call = timeshift(audio, sr)
        call = augmentation.noise(ts_call, sr)
        effect = "ts_noise"
        return call, effect
    
    
    def ts_bsf_noise(audio, sr):
        timeshift = Shift(min_fraction=-0.25, max_fraction=0.25, 
                          rollover=True, p=1)
        ts_call = timeshift(audio, sr)
        ts_noise_call = augmentation.noise(ts_call, sr)
        bandsf = BandStopFilter(min_center_freq=1000, max_center_freq=100000,
            min_bandwidth_fraction=0.5, max_bandwidth_fraction=1, p=1)
        call = bandsf(ts_noise_call, sr)
        effect = "ts_bsf_noise"
        return call, effect


# create new directory and augment files to create dataset 
# with 4000 examples per species
        
for sp in species:
    files = os.listdir(f"{train_dir}/{sp}/")
    if len(files) < 3750:
        save_loc = f"{new_path}/aug/{sp}/"
        if os.path.exists(save_loc) is False:
            os.makedirs(save_loc)
        no_aug = 3750 - len(files)
        rpt = math.ceil(no_aug / len(files))
        
        for file in files:
            fname = file.rsplit('.', 1)[0]
            audio, sr = lb.load(f"{train_dir}/{sp}/{file}", sr=256000)
            audio = audio[:sr]
            for i in range(rpt):
                if len(os.listdir(f"{save_loc}/")) + len(files) < 3750:  
                    print(f"processing......{file}")
                    for func in random.sample(aug_effects, 1):
                        func_ = getattr(augmentation, f"{func}")
                        call, effect = func_(audio, sr)
                        file = f"{save_loc}/{fname}_{effect}_{i}.wav"
                        if os.path.exists(file) is False:
                           sf.write(file=file, data=call, samplerate=sr)