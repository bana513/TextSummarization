import torch


class Config:
    device = torch.device("cuda:0")
    data_path = "D:/Data/text_summarization/"

    # Parameters:
    batch_size = 8
    lr = 1e-3
    max_grad_norm = 1.0
    num_epochs = 10
    num_warmup_steps = 100
    max_len = 512
    vocab_size = 119547
    clip = .5

    decoder_hidden_dim = 128
    encoder_dim = embed_dim = 768
    attention_dim = 64
    decoder_token_num = 119547

    PAD_ID, CLS_ID, SEP_ID, MASK_ID = 0, 101, 102, 103

    def __init__(self):
        if Config.device.type == 'cuda':
            torch.cuda.set_device(self.device)
