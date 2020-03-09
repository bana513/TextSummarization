import torch
from torch.nn import NLLLoss
from torch.optim.adamw import AdamW
from transformers import get_linear_schedule_with_warmup, BertTokenizer

from summarization.config import Config
from summarization.data import read_dataset, split_dataset, get_data_loader
from summarization.evaluation import evaluate_and_show_attention, test_text
from summarization.model import BertSummarizer
from summarization.progress_bar import ProgressBar

if __name__ == '__main__':
    Config()
    device = Config.device

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)

    contents, summaries = read_dataset(Config.data_path + "hvg_tokenized.pickle")
    train_contents, train_summaries, valid_contents, valid_summaries = split_dataset(contents, summaries, .9)
    train_loader = get_data_loader(train_contents, train_summaries, train_set=True)
    valid_loader = get_data_loader(valid_contents, valid_summaries, train_set=False)
    print(f"# samples: {len(contents)}, # train batches: {len(train_loader)}")

    model = BertSummarizer().to(device)
    print("Total params: " + str(sum(p.numel() for p in model.parameters())))
    print("Trainable params: " + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print(model)

    optimizer = AdamW(model.parameters(), lr=Config.lr)

    num_training_steps = Config.num_epochs * len(train_loader) // Config.batch_size
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=Config.num_warmup_steps,
                                                num_training_steps=num_training_steps)

    criterion = NLLLoss(ignore_index=Config.PAD_ID)

    progress_bar = ProgressBar(len(train_loader))

    for epoch in range(Config.num_epochs):
        progress_bar.start()
        for batch in train_loader:
            model.train()
            preds, attns = model(batch)  # batch = (content_tensors, content_lengths, summary_tensors, summary_lengths)

            target = batch[2]
            loss = criterion(preds.permute(0, 2, 1), target.to(Config.device))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           Config.max_grad_norm)

            optimizer.step()
            scheduler.step()

            if progress_bar.count % 50 == 0:
                evaluate_and_show_attention(model, test_text, tokenizer,
                                            iteration=epoch+progress_bar.count/len(train_loader))

            progress_bar.update(loss=loss.item())
            progress_bar.progress()

        print(f"\rEpoch: {epoch + 1}\tLoss: {progress_bar.loss}")
