import torch
import torch.nn as nn
import argparse
import numpy as np
import random
import time
import shutil
import os

import hparams as hp
import audio
import utils
import dataset
import text
import model as M
import waveglow

from eval_dataset import get_eval_dataset, DataLoader
import whisper
from torchaudio.pipelines import SQUIM_SUBJECTIVE
from metric import wer, mos

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_DNN(num):
    checkpoint_path = "checkpoint_" + str(num) + ".pth.tar"
    model = nn.DataParallel(M.FastSpeech()).to(device)
    model.load_state_dict(torch.load(os.path.join(hp.checkpoint_path,
                                                  checkpoint_path))['model'])
    model.eval()
    return model


def synthesis(model, text, alpha=1.0):
    text = np.array(text)
    text = np.stack([text])
    src_pos = np.array([i+1 for i in range(text.shape[1])])
    src_pos = np.stack([src_pos])
    sequence = torch.from_numpy(text).cuda().long()
    src_pos = torch.from_numpy(src_pos).cuda().long()

    with torch.no_grad():
        mel, mel_postnet = model.module.forward(sequence, src_pos, alpha=alpha)
    return mel_postnet[0].cpu().transpose(0, 1), mel_postnet.contiguous().transpose(1, 2), mel[0].cpu().transpose(0, 1)


# def get_data():
#     test1 = "I am very happy to see you again!"
#     test2 = "Durian model is a very good speech synthesis!"
#     test3 = "When I was twenty, I fell in love with a girl."
#     test4 = "I remove attention module in decoder and use average pooling to implement predicting r frames at once"
#     test5 = "You can not improve your past, but you can improve your future. Once time is wasted, life is wasted."
#     test6 = "Death comes to all, but great achievements raise a monument which shall endure until the sun grows old."
#     data_list = list()
#     data_list.append(text.text_to_sequence(test1, hp.text_cleaners))
#     data_list.append(text.text_to_sequence(test2, hp.text_cleaners))
#     data_list.append(text.text_to_sequence(test3, hp.text_cleaners))
#     data_list.append(text.text_to_sequence(test4, hp.text_cleaners))
#     data_list.append(text.text_to_sequence(test5, hp.text_cleaners))
#     data_list.append(text.text_to_sequence(test6, hp.text_cleaners))
#     return data_list

def draw_mel(mel_ori, mel):
    mel_ori = mel_ori.cpu().detach().numpy()
    mel = mel.cpu().detach().numpy()

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(mel_ori, aspect='auto', interpolation='none', origin='lower')
    plt.colorbar()
    plt.title("Original Mel")

    plt.subplot(1, 2, 2)
    plt.imshow(mel, aspect='auto', interpolation='none', origin='lower')
    plt.colorbar()
    plt.title("Predicted Mel")

    plt.tight_layout()
    plt.savefig("test_mel.png")
    

if __name__ == "__main__":
    # Test
    WaveGlow = utils.get_WaveGlow()
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, default=201000)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--ckpt_path", type=str, default=hp.checkpoint_path, required=False)
    parser.add_argument("--save_path", type=str, default=hp.save_path, required=False)
    args = parser.parse_args()

    # print("use griffin-lim and waveglow")
    print("use waveglow")
    model = get_DNN(args.step)
    num_param = utils.get_param_num(model)
    print('Number of TTS Parameters:', num_param)
    eval_dataset = get_eval_dataset()
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False, drop_last=True, num_workers=0)
    
    wer_tot = 0
    wer_gt_tot = 0
    mos_tot = 0
    time_tot = 0

    whisper_model = whisper.load_model("base.en").cuda()
    squim_subjective_model = SQUIM_SUBJECTIVE.get_model().cuda()

    for i, data in enumerate(eval_dataloader):
        wav_gt = data['wav'].cuda()
        text_gt = data['text'][0]
        phn = text.text_to_sequence(text_gt, hp.text_cleaners)

        s_t = time.perf_counter()
        mel, mel_cuda, mel_original = synthesis(model, phn, args.alpha)
        e_t = time.perf_counter()
        draw_mel(mel_original, mel)
        wav_pred = waveglow.inference.inference_direct(mel_cuda, WaveGlow)
        wav_pred = torch.Tensor(wav_pred).type(torch.float32) / 32768.0
        

        wer_value = wer(whisper_model, text_gt, wav_pred)
        wer_gt_value = wer(whisper_model, text_gt, wav_gt.squeeze())
        mos_value = mos(squim_subjective_model, wav_gt, wav_pred)
        time_value = e_t - s_t
        wer_tot += wer_value
        wer_gt_tot += wer_gt_value
        mos_tot += mos_value
        time_tot += time_value
        print(f"idx: {i}, WER: {wer_value}, WER GT: {wer_gt_value}, MOS: {mos_value}, Time: {time_value}")

    print(f"WER: {wer_tot / len(eval_dataloader)}, WER GT: {wer_gt_tot / len(eval_dataloader)}, MOS: {mos_tot / len(eval_dataloader)}, Time: {time_tot / len(eval_dataloader)}")