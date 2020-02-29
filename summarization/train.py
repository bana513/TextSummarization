import torch
from torch.nn import NLLLoss
from summarization.config import Config
from transformers import get_linear_schedule_with_warmup
from summarization.data import SummarizationDataset, collate
from summarization.model import BertSummarizer
from summarization.progress_bar import ProgressBar
from summarization.sampler import NoisySortedBatchSampler
from torch.utils.data import DataLoader
from torch.optim.adamw import AdamW
from tqdm import tqdm


if __name__ == '__main__':
    Config()
    device = Config.device

    train_dataset = SummarizationDataset(Config.data_path + "hvg_tokenized.pickle", Config.batch_size)
    train_sampler = NoisySortedBatchSampler(train_dataset, batch_size=Config.batch_size, drop_last=True,
                                            shuffle=True, sort_key_noise=0.02)
    train_loader = DataLoader(train_dataset,
                              collate_fn=collate,
                              num_workers=5,
                              batch_sampler=train_sampler)
    print("Number of batches:", len(train_loader))

    model = BertSummarizer().to(device)
    print("Total params: " + str(sum(p.numel() for p in model.parameters())))
    print("Trainable params: " + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print(model)

    optimizer = AdamW(model.parameters(), lr=Config.lr)

    num_training_steps = Config.num_epochs * len(train_loader) // Config.batch_size
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=Config.num_warmup_steps,
                                                num_training_steps=num_training_steps)  # PyTorch scheduler

    criterion = NLLLoss(ignore_index=Config.PAD_ID)

    progress_bar = ProgressBar(len(train_loader))

    running_loss = None

    for epoch in range(Config.num_epochs):
        progress_bar.start()
        for batch in train_loader:

            preds, attns = model(batch)  # batch = (content_tensors, content_lengths, summary_tensors, summary_lengths)

            target = batch[2]
            loss = criterion(preds.permute(0,2,1), target.to(Config.device))

            if running_loss is None:
                running_loss = loss.item()
            running_loss = .95 * running_loss + .05 * loss.item()
            print(f"Loss: {running_loss}", end="")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           Config.max_grad_norm)

            optimizer.step()
            scheduler.step()

            progress_bar.update()
            progress_bar.progress()
        print(f"\rEpoch: {epoch + 1}, loss: {running_loss}")
        progress_bar.stop()


