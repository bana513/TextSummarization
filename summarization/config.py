import torch

from summarization.tokenizer import UsedBertTokens


class Config:
    device = torch.device("cuda:0")
    data_path = "D:/Data/text_summarization/"
    # data_path = "/userhome/student/bial/remotepycharm/data/text_summarization/"

    # Parameters:
    batch_size = 24
    lr = 1e-3
    num_epochs = 10
    num_warmup_steps = 100
    max_len = 512
    vocab_size = decoder_token_num = None
    grad_clip = .5

    decoder_hidden_dim = 128
    encoder_dim = 768
    attention_dim = 64

    PAD_ID, UNK_ID, CLS_ID, SEP_ID, MASK_ID, S_ID, T_ID = 0, 1, 2, 3, 4, 5, 6 # 0, 100, 101, 102, 103, 104, 105

    def __init__(self):
        if Config.device.type == 'cuda':
            torch.cuda.set_device(self.device)

        UsedBertTokens(Config.data_path)
        Config.vocab_size = Config.decoder_token_num = len(UsedBertTokens.get_instance().token_ids)
