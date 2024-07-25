import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
import librosa
import numpy as np
from text import text_to_sequence
import collections
from scipy import signal
import torch as t
import math


class LJDatasets(Dataset):
    """LJSpeech dataset."""

    def __init__(self, csv_file, root_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the wavs.

        """
        self.landmarks_frame = pd.read_csv(csv_file, sep='|', header=None)
        self.root_dir = root_dir

    def load_wav(self, filename):
        return librosa.load(filename, sr=22050)

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        wav_name = os.path.join(
            self.root_dir, self.landmarks_frame.iloc[idx, 0]) + '.wav'
        
        wav = self.load_wav(os.path.join(
            self.root_dir, "wavs", self.landmarks_frame.iloc[idx, 0]) + '.wav')[0]
        text = self.landmarks_frame.iloc[idx, 1]

        sample = {'text': text, 'wav': wav}

        return sample

def get_param_size(model):
    params = 0
    for p in model.parameters():
        tmp = 1
        for x in p.size():
            tmp *= x
        params += tmp
    return params

def get_eval_dataset():
    data_path = "/workspace/kickstart-tts/FastSpeech/data/LJSpeech-1.1"
    return LJDatasets(os.path.join(data_path, 'val.csv'), data_path)