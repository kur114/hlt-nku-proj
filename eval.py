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


def get_data():
    test1 = "I am very happy to see you again!"
    test2 = "Durian model is a very good speech synthesis!"
    test3 = "When I was twenty, I fell in love with a girl."
    test4 = "I remove attention module in decoder and use average pooling to implement predicting r frames at once"
    test5 = "You can not improve your past, but you can improve your future. Once time is wasted, life is wasted."
    test6 = "Death comes to all, but great achievements raise a monument which shall endure until the sun grows old."
    data_list = list()
    data_list.append(text.text_to_sequence(test1, hp.text_cleaners))
    data_list.append(text.text_to_sequence(test2, hp.text_cleaners))
    data_list.append(text.text_to_sequence(test3, hp.text_cleaners))
    data_list.append(text.text_to_sequence(test4, hp.text_cleaners))
    data_list.append(text.text_to_sequence(test5, hp.text_cleaners))
    data_list.append(text.text_to_sequence(test6, hp.text_cleaners))
    return data_list

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
    plt.savefig(args.save_path+"/"+str(args.step)+"_"+str(i)+"_mel.png")
    

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
    data_list = get_data()
    for i, phn in enumerate(data_list):
        mel, mel_cuda, mel_original = synthesis(model, phn, args.alpha)
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)
        # draw mel_original & mel
        draw_mel(mel_original, mel)
        # audio.tools.inv_mel_spec(
        #     mel, args.save_path+"/"+str(args.step)+"_"+str(i)+".wav")
        # print("Done Griffin-Lim", i + 1)
        waveglow.inference.inference(
            mel_cuda, WaveGlow,
            args.save_path+"/"+str(args.step)+"_"+str(i)+"_waveglow.wav")
        # print("Done WaveGlow", i + 1)
        print("Done", i + 1)

    s_t = time.perf_counter()
    for i in range(100):
        for _, phn in enumerate(data_list):
            _, _, _ = synthesis(model, phn, args.alpha)
        print(i)
    e_t = time.perf_counter()
    print((e_t - s_t) / 100.)
