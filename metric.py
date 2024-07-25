import torch
import numpy as np
import torch.functional as F
from icecream import ic
import whisper
import torchaudio
import jiwer
from torchaudio.pipelines import SQUIM_SUBJECTIVE

def wer(model, text_gt, audio_pred):
    # whisper
    audio_pred = torchaudio.transforms.Resample(22050, 16000)(audio_pred.cpu())
    audio_pred.cpu().numpy().astype(np.int16).flatten().astype(np.float32) / 32768.0
    text_pred = model.transcribe(audio_pred)
    text_pred = text_pred['text']
    ic(text_pred, text_gt)
    transformation = jiwer.Compose([
        jiwer.ExpandCommonEnglishContractions(),
        jiwer.RemovePunctuation(),
        jiwer.ToLowerCase(),
        jiwer.RemoveWhiteSpace(replace_by_space=True),
        jiwer.RemoveMultipleSpaces(),
        jiwer.ReduceToListOfListOfWords(word_delimiter=" ")
    ])
    wer = jiwer.wer([text_gt], text_pred, truth_transform=transformation, hypothesis_transform=transformation)
    # ic(wer)
    return wer

def mos(model, audio_gt, audio_pred):
    # SQUIM
    audio_gt = torchaudio.transforms.Resample(22050, 16000)(audio_gt.cpu()).cuda()
    audio_pred = torchaudio.transforms.Resample(22050, 16000)(audio_pred.cpu()).cuda().unsqueeze(0)
    mos = model(audio_pred, audio_gt)[0].detach().cpu().numpy()
    # ic(mos)
    return mos
