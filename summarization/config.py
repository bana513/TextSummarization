import torch


class Config:
    device = torch.device("cuda:0")
    data_path = "D:/Data/text_summarization/"

    max_len = 512
    vocab_size = 119547
    clip = .5
    batch_size = 32
    warmup = -1

    def __init__(self):
        if Config.device.type == 'cuda':
            torch.cuda.set_device(self.device)
