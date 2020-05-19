import torch

from summarization import UsedBertTokens
import matplotlib


class Config:
    on_server = False

    # Load model
    load_state = 20  # Set None for new model

    # Tensorboard info
    model_name = "bert_lstm_dropout_4000"

    if on_server:
        device = torch.device("cuda:1")
        data_path = "/userhome/student/bial/remotepycharm/data/text_summarization/"  # remote server path
        batch_size = 24
        matplotlib.use('Agg')
    else:
        device = torch.device("cuda:0")
        data_path = "D:/Data/text_summarization/"  # local path
        batch_size = 16

    # Parameters:
    lr = 3e-3
    num_epochs = 30
    num_warmup_steps = 1000
    max_len = 512
    vocab_size = decoder_token_num = None
    grad_clip = .5
    max_content_len = 128

    decoder_dim = 512
    encoder_dim = 768
    attention_dim = 128
    embedding_dim = 768
    value_dim = 512

    # Dropouts
    q_drop = 0.2
    k_drop = 0.3
    v_drop = 0.3
    lstm_out_drop = 0.2

    # input_drop = 0.1
    # output_drop = 0.1
    # hidden_drop = 0.1
    # weight_drop = 0.1

    PAD_ID, UNK_ID, CLS_ID, SEP_ID, MASK_ID, S_ID, T_ID = 0, 1, 2, 3, 4, 5, 6 # 0, 100, 101, 102, 103, 104, 105

    # token_id_list = "used_token_ids_4000.pkl"
    tokenized_data_file = "hvg_tokenized_shrink_4000.pkl"

    def __init__(self):
        if Config.device.type == 'cuda':
            torch.cuda.set_device(self.device)

        UsedBertTokens(Config.data_path)  # Load token id translator
        Config.vocab_size = Config.decoder_token_num = len(UsedBertTokens.get_instance().token_ids)
