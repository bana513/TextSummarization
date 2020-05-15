import torch

from summarization import UsedBertTokens
import matplotlib


class Config:
    on_server = False

    # Load model
    load_state = None  # Set None for new model

    # Tensorboard info
    model_name = "test_lstm_enc"

    if on_server:
        device = torch.device("cuda:1")
        data_path = "/userhome/student/bial/remotepycharm/data/text_summarization/"  # remote server path
        batch_size = 24
        matplotlib.use('Agg')
    else:
        device = torch.device("cuda:0")
        data_path = "D:/Data/text_summarization/"  # local path
        batch_size = 3

    # Parameters:
    lr = 5e-3
    num_epochs = 400
    num_warmup_steps = 10 #100
    max_len = 512
    vocab_size = decoder_token_num = None
    grad_clip = .5

    decoder_hidden_dim = 64 #128
    encoder_dim = 64 #768
    attention_dim = 64

    PAD_ID, UNK_ID, CLS_ID, SEP_ID, MASK_ID, S_ID, T_ID = 0, 1, 2, 3, 4, 5, 6 # 0, 100, 101, 102, 103, 104, 105

    def __init__(self):
        if Config.device.type == 'cuda':
            torch.cuda.set_device(self.device)

        UsedBertTokens(Config.data_path)  # Load token id translator
        Config.vocab_size = Config.decoder_token_num = len(UsedBertTokens.get_instance().token_ids)
