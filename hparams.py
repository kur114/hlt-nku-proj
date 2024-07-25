# Mel
num_mels = 80
text_cleaners = ['english_cleaners']

# FastSpeech
vocab_size = 300
max_seq_len = 3000

encoder_dim = 256
encoder_n_layer = 4
encoder_head = 2
encoder_conv1d_filter_size = 1024

decoder_dim = 256
decoder_n_layer = 4
decoder_head = 2
decoder_conv1d_filter_size = 1024

fft_conv1d_kernel = (9, 1)
fft_conv1d_padding = (4, 0)

duration_predictor_filter_size = 256
duration_predictor_kernel_size = 3
dropout = 0.1

conformer_conv_kernel_size = 31
block_type = 'FFCBlock' # ['FFTBlock', 'FFCBlock']

postnet_type = 'UNet' # ['CBHG', 'UNet']
unet_max_seq_len = 1024

# Train
root_path = "backup"
exp_name = "unet"
checkpoint_path = f"./{root_path}/{exp_name}/model_new"
logger_path = f"./{root_path}/{exp_name}/logger"
save_path = f"./{root_path}/{exp_name}/results"
# checkpoint_path = "./model_new"
# logger_path = "./logger"
# save_path = "./results"
# mel_ground_truth = "./mels"
alignment_path = "./alignments"

batch_size = 64
epochs = 2000
n_warm_up_step = 4000

learning_rate = 1e-3
weight_decay = 1e-6
grad_clip_thresh = 1.0
decay_step = [500000, 1000000, 2000000]

save_step = 3000
log_step = 5
clear_Time = 20

batch_expand_size = 32
