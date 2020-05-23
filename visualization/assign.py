from langdetect import detect 
from transformers import BertTokenizer
import pandas as pd

def assign_lang(x):
    try:
        res = detect(x)
    except Exception as e:
        res = "unk"
    return res

def contains_ekezet(x):
    if 'á' in x:
        return True
    elif 'é' in x:
        return True
    elif 'í' in x:
        return True
    elif 'ó' in x:
        return True
    elif 'ú' in x:
        return True
    elif 'ü' in x:
        return True
    elif 'ü' in x:
        return True
    elif 'ö' in x:
        return True
    elif 'ő' in x:
        return True
    return False

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)

def tokenize(x):
    content, summary = x
    
    if pd.isna(content) or pd.isna(summary): return None, None

    content = tokenizer.encode(content)
    if 20 < len(content) <= 512:
        summary = tokenizer.encode(summary)

        if 8 < len(summary) <= len(content) // 2:
            return content, summary
    return None, None