import torch
from torch.nn import NLLLoss
from torch.optim.adamw import AdamW
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup, BertTokenizer

from summarization import Config
from summarization import read_dataset, split_dataset, get_data_loader
from summarization import evaluate_and_show_attention, test_text, validate_model
from summarization import BertSummarizer
from summarization import ProgressBar
from summarization import SmartTokenizer
from summarization import save_model

if __name__ == '__main__':
    Config()
    device = Config.device

    tokenizer = SmartTokenizer()

    contents, summaries = read_dataset(Config.data_path + "hvg_tokenized_shrink.pickle")
    train_contents, train_summaries, valid_contents, valid_summaries = split_dataset(contents, summaries, .95)
    train_loader = get_data_loader(train_contents, train_summaries, train_set=True)
    valid_loader = get_data_loader(valid_contents, valid_summaries, train_set=False)
    print(f"# samples: {len(contents)}, # train batches: {len(train_loader)}")

    model = BertSummarizer().to(device)
    print("Total params: " + str(sum(p.numel() for p in model.parameters())))
    print("Trainable params: " + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print(model)

    optimizer = AdamW(model.parameters(), lr=Config.lr)

    epoch_steps = len(train_loader)
    num_training_steps = Config.num_epochs * epoch_steps
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=Config.num_warmup_steps,
                                                num_training_steps=num_training_steps)

    criterion = NLLLoss(ignore_index=Config.PAD_ID)

    progress_bar = ProgressBar(len(train_loader))

    # Tensorboard
    summary_writer = SummaryWriter(comment=Config.model_name)
    summary_writer.add_hparams({k: str(v) for k, v in Config.__dict__.items() if not k.startswith('__')}, {})
    summary_writer.add_hparams({"epoch_steps": epoch_steps}, {})

    for epoch in range(Config.num_epochs):
        model.train()
        progress_bar.start()

        for batch in train_loader:
            # Forward step
            preds, attns = model(batch)  # batch = (content_tensors, content_lengths, summary_tensors, summary_lengths)

            # Calculate loss
            target = batch[2]
            loss = criterion(preds.permute(0, 2, 1), target.to(Config.device))

            # Calculate gradients
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           Config.max_grad_norm)

            optimizer.step()

            # Warmup scheduler
            scheduler.step()

            # Update progress bar
            progress_bar.update(loss=loss.item())
            progress_bar.progress()

            # Show attention plot
            if progress_bar.count % 50 == 0:
                evaluate_and_show_attention(model, test_text, tokenizer,
                                            iteration=epoch+progress_bar.count/epoch_steps,
                                            to_file=True)
                model.train()

            # Update tensorboard
            if progress_bar.count % 10 == 0:
                counter = epoch * progress_bar.total_items + progress_bar.count
                summary_writer.add_scalar(f'train_loss', progress_bar.loss, counter)

        # Validate model every epoch
        val_loss, val_acc = validate_model(model=model,
                                           criterion=criterion,
                                           valid_loader=valid_loader,
                                           tokenizer=tokenizer,
                                           summary_writer=summary_writer,
                                           step=(epoch+1)*epoch_steps,
                                           verbose=True)

        print(f"\rEpoch: {epoch + 1}\tTrain loss: {progress_bar.loss:2.3f}" +
              f"\tValid loss: {val_loss:2.3f}\tValid accuracy: {val_acc*100:2.4f}%")

        # Save model
        save_model(model.decoder, optimizer, epoch)
