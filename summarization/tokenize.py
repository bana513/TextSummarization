from summarization.config import Config
from transformers import BertTokenizer
from tqdm import tqdm
import pickle
import pandas as pd

from summarization.tokenizer import SmartTokenizer

if __name__ == "__main__":
    Config()
    df = pd.read_csv(Config.data_path + "hvg.csv")

    # tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
    tokenizer = SmartTokenizer()

    summaries = []
    contents = []

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        if pd.isna(row.content) or pd.isna(row.description): continue

        content = tokenizer.encode(row.content)
        if 20 < len(content) <= 512:
            summary = tokenizer.encode(row.description)

            if 8 < len(summary) <= len(content) // 2:
                summaries.append(summary)
                contents.append(content)

    print(len(summaries), len(contents))

    with open(Config.data_path + "hvg_tokenized_shrink.pickle", 'wb') as f:
        pickle.dump((contents, summaries), f)

