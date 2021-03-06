import pickle

import torch
from torch.nn import NLLLoss
from torch.optim.adamw import AdamW
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup

from summarization import Config, load_model, read_dataset, split_dataset, get_data_loader, \
    evaluate_and_show_attention, test_text, test_text2, validate_model, BertSummarizer, ProgressBar,\
    SmartTokenizer, save_model

if __name__ == '__main__':
    Config()
    device = Config.device

    tokenizer = SmartTokenizer()

    contents, summaries = read_dataset(Config.data_path + Config.tokenized_data_file)
    train_contents, train_summaries, valid_contents, valid_summaries = split_dataset(contents, summaries, .97)
    train_contents, train_summaries, valid_tf_contents, valid_tf_summaries = split_dataset(train_contents, train_summaries, .97)

    train_loader = get_data_loader(train_contents, train_summaries, train_set=True)
    valid_tf_loader = get_data_loader(valid_tf_contents, valid_tf_summaries, train_set=False)
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

    if Config.token_weight_file is not None:
        with open(Config.data_path + Config.token_weight_file, 'rb') as f:
            token_weights = pickle.load(f)
        token_weights = torch.Tensor(token_weights).to(Config.device)
    else:
        token_weights = None
    criterion = NLLLoss(ignore_index=Config.PAD_ID, weight=token_weights)

    progress_bar = ProgressBar(len(train_loader), ema=0)

    # Tensorboard
    summary_writer = SummaryWriter(comment=Config.model_name)
    summary_writer.add_hparams({k: str(v) for k, v in Config.__dict__.items() if not k.startswith('__')}, {})
    summary_writer.add_hparams({"epoch_steps": epoch_steps}, {})

    if Config.load_state is not None:
        load_model(model.decoder, optimizer, Config.load_state)

    running_acc = None
    adaptive_summary_len = 3
    increase_sum_len = 0

    for epoch in range(0 if Config.load_state is None else Config.load_state,
                       Config.num_epochs):
        model.train()
        progress_bar.start()

        for batch in train_loader:

            content_tensors, content_lengths, summary_tensors, summary_lengths = batch
            summary_tensors = summary_tensors[:, :adaptive_summary_len]
            summary_lengths.clamp_(0, 3)

            # Forward step
            preds, attns = model(batch)  # batch = (content_tensors, content_lengths, summary_tensors, summary_lengths)

            # Calculate loss
            target = batch[2].to(Config.device)
            target[target == Config.UNK_ID] = Config.PAD_ID
            loss = criterion(preds.permute(0, 2, 1), target)

            # Calculate accuracy
            pred_id = preds.argmax(2)
            acc = (pred_id == target).float().mean().cpu()
            running_acc = running_acc * .95 + acc * 0.05 if running_acc is not None else acc
            if running_acc > .1:
                increase_sum_len += 1
                if increase_sum_len == 10:
                    adaptive_summary_len += 1
            else:
                increase_sum_len = 0

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
            if progress_bar.count % 200 == 0:
                evaluate_and_show_attention(model, test_text, tokenizer,
                                            iteration=epoch + progress_bar.count / epoch_steps)
                evaluate_and_show_attention(model, test_text2, tokenizer,
                                            iteration=epoch + progress_bar.count / epoch_steps)
                model.train()

            if progress_bar.count % 2000 == 0:
                # Validate model every epoch
                val_loss, val_acc, tf_loss, tf_acc, rouge1, rouge2, rougel = validate_model(
                    model=model,
                    criterion=criterion,
                    valid_loader=valid_loader,
                    valid_tf_loader=valid_tf_loader,
                    tokenizer=tokenizer,
                    summary_writer=summary_writer,
                    epoch=epoch+progress_bar.count/epoch_steps,
                    epoch_steps=epoch_steps,
                    verbose=True)

                print(f"\rEpoch: {epoch + 1}" +
                      f"\tTrain loss: {progress_bar.loss:2.3f}" +
                      f"\tValid loss: {val_loss:2.3f}\t" +
                      f"\tValid tf loss: {tf_loss:2.3f}\t")

                model.train()

            # Update tensorboard
            if progress_bar.count % 10 == 0:
                try:
                    counter = epoch * progress_bar.total_items + progress_bar.count
                    summary_writer.add_scalars('Loss', {
                        'train': progress_bar.loss,
                    }, counter)
                    summary_writer.add_scalars('Acc', {
                        'train': running_acc,
                    }, counter)
                    summary_writer.add_scalars('Summary len', {
                        'train': adaptive_summary_len,
                    }, counter)
                except Exception as e:
                    print("Summary writer error")

        # Validate model every epoch
        val_loss, val_acc, tf_loss, tf_acc, rouge1, rouge2, rougel = validate_model(
            model=model,
            criterion=criterion,
            valid_loader=valid_loader,
            valid_tf_loader=valid_tf_loader,
            tokenizer=tokenizer,
            summary_writer=summary_writer,
            epoch=epoch+1,
            epoch_steps=epoch_steps,
            verbose=True)

        print(f"\rEpoch: {epoch + 1}" +
              f"\tTrain loss: {progress_bar.loss:2.3f}" +
              f"\tValid loss: {val_loss:2.3f}\t" +
              f"\tValid tf loss: {tf_loss:2.3f}\t")

        # Save model
        if (epoch+1) % 1 == 0:
            try:
                save_model(model.decoder, optimizer, epoch+1)
            except OSError:
                print("Error while saving. Skipping...")
