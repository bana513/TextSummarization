import numpy as np
import pickle

from transformers import BertTokenizer


class UsedBertTokens:
    """Singleton class"""

    __instance = None

    @staticmethod
    def get_instance():
        """ Static access method. """
        if UsedBertTokens.__instance is None:
            UsedBertTokens()
        return UsedBertTokens.__instance

    def __init__(self, data_path):
        """ Virtually private constructor. """
        if UsedBertTokens.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            UsedBertTokens.__instance = self
            with open(data_path + "used_token_ids.pickle", 'rb') as f:
                self.token_ids = pickle.load(f)


class SmartTokenizer:
    def __init__(self, UNK_ID=100):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)

        token_ids = UsedBertTokens.get_instance().token_ids
        self.decode_matrix = np.array(token_ids)

        self.encode_matrix = np.ones(self.tokenizer.vocab_size, dtype=int) * UNK_ID
        for i, token_id in enumerate(token_ids):
            self.encode_matrix[token_id] = i

    def decode(self, token_ids):
        return self.tokenizer.decode(self.decode_matrix[token_ids])

    def encode(self, text):
        return self.encode_matrix[self.tokenizer.encode(text)].tolist()

    def translate(self, token_ids):
        return self.encode_matrix[token_ids]

    def translate_back(self, token_ids):
        return self.decode_matrix[token_ids]
