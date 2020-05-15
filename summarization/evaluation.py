import torch
import matplotlib.pyplot as plt
from matplotlib import ticker
from summarization import Config
from summarization import ProgressBar
from rouge import Rouge
import numpy as np

test_text = "Egy ember életét vesztette, amikor két személygépkocsi ütközött az 1-es főúton Bicskénél közölte a Fejér Megyei Rendőr-főkapitányság szóvivője. A balesetben hárman sérültek meg, egy ember olyan súlyosan, hogy a helyszínen meghalt. A rendőrség a helyszínelés idejére teljesen lezárta az érintett útszakaszt, a forgalmat Bicske belterületén keresztül terelik el."


def validate(model, criterion, loader, tokenizer, tf=False, step=None, verbose=True):
    if verbose: print()
    model.eval()

    progress_bar = ProgressBar(len(loader))
    progress_bar.start()
    total_acc, total_loss, n = 0, 0, 0
    total_rouge1, total_rouge2, total_rougel = 0, 0, 0

    with torch.no_grad():
        for batch in loader:
            padded_contents, _, padded_summaries, _ = batch
            max_len = padded_summaries.shape[1]

            preds, _ = model(batch, teacher_forcing_ratio=1 if tf else 0, max_len=max_len)
            loss = criterion(preds.permute(0, 2, 1), padded_summaries.to(Config.device))

            refs, hyps = [], []

            if not tf:
                for pred, summary in zip(preds, padded_summaries):
                    pred = pred.argmax(1).detach().cpu()
                    eos_index = (pred == Config.SEP_ID).nonzero()
                    if len(eos_index) > 0:
                        eos_index = eos_index[0].item()
                        pred = pred[:eos_index + 1]

                    eos_index = (summary == Config.SEP_ID).nonzero()
                    if len(eos_index) > 0:
                        eos_index = eos_index[0].item()
                        summary = summary[:eos_index + 1]

                    ref = tokenizer.decode(summary)
                    try:
                        hyp = tokenizer.decode(pred)
                    except TypeError as e:
                        hyp = "[CLS] [SEP]"

                    refs.append(ref)
                    hyps.append(hyp)


                rouge1, rouge2, rougel = calc_rouge(hyps, refs)
                total_rouge1 += rouge1
                total_rouge2 += rouge2
                total_rougel += rougel

            total_loss += loss.item()
            n += 1
            progress_bar.update()

        if verbose:
            for batch in loader:
                padded_contents, content_sizes, padded_summaries, summary_sizes = batch

                preds, attentions = model(batch, teacher_forcing_ratio=0, max_len=max_len)

                output = preds.argmax(2).detach().cpu()

                # Print first few examples from first batch
                for i in range(min(output.shape[0], 3)):
                    output_i = output[i]
                    eos_index = (output_i == Config.SEP_ID).nonzero()
                    if len(eos_index) > 0:
                        eos_index = eos_index[0].item()
                        output_i = output_i[:eos_index + 1]

                    print("\nContent:\n" + tokenizer.decode(padded_contents[i][:content_sizes[i]].tolist()))
                    print("Summary:\n" + tokenizer.decode(padded_summaries[i][:summary_sizes[i]].tolist()))
                    print("Prediction:\n" + tokenizer.decode(output_i.tolist()))
                break

            evaluate_and_show_attention(model, test_text, tokenizer,
                                        step if step is not None else 0)

    loss, acc = total_loss / n, total_acc / n
    rouge1, rouge2, rougel = total_rouge1/n, total_rouge2/n, total_rougel/n
    return loss, acc, rouge1, rouge2, rougel


def validate_model(model, criterion, valid_loader, valid_tf_loader, tokenizer, summary_writer, step=None, verbose=True):
    if verbose: print()
    model.eval()

    # Validate on one part with teacher forcing too see predicted summaries are getting better
    print("Validating with TF ...")
    tf_loss, tf_acc, _, _, _ = validate(model, criterion, valid_tf_loader, tokenizer,
                                        tf=True, step=step, verbose=verbose)

    # Validate on the real validation set without teacher forcing to get real results without cheating
    print("Validating...")
    loss, acc, rouge1, rouge2, rougel = validate(model, criterion, valid_loader, tokenizer,
                                                 tf=False, step=step, verbose=False)

    if summary_writer is not None and step is not None:
        summary_writer.add_scalars('Loss', {
            'valid_tf': tf_loss,
            'valid': loss,
        }, step)
        summary_writer.add_scalars('Rouge', {
            'Rouge/rouge-1': rouge1,
            'Rouge/rouge-2': rouge2,
            'Rouge/rouge-L': rougel
        }, step)

    return loss, acc, tf_loss, tf_acc, rouge1, rouge2, rougel


def evaluate(model, input_text, tokenizer, max_len=100):
    model.eval()
    with torch.no_grad():
        content = torch.LongTensor(tokenizer.encode(input_text)).unsqueeze(0)
        content_size = torch.LongTensor([content.shape[1]])

        output, attention = model((content, content_size, None, None),
                                  teacher_forcing_ratio=0.0, max_len=max_len)

        attention = attention.squeeze(0).detach().cpu()

        output = output.argmax(2).detach().cpu()
        output = output.squeeze(0)

        eos_index = (output == Config.SEP_ID).nonzero()
        if len(eos_index) > 0:
            eos_index = eos_index[0].item()
            output = output[:eos_index+1]
            attention = attention[:eos_index+1]

        output_words = [tokenizer.decode([w]) for w in output.tolist()]
    return output_words, attention


def show_attention(input_sentence, output_words, attentions, iteration=0, to_file=Config.on_server):
    # Set up figure with colorbar
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)

    fig.suptitle(f"Epoch: {iteration:.2f}")
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)
    # Set up axes
    # Weird matplotlib bug - not showing first word -> shift
    ax.set_xticklabels(["[CLS]"]+input_sentence, rotation=45)
    ax.set_yticklabels(["[CLS]"]+output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    if to_file:
        plt.savefig(f"{Config.data_path}plots/{Config.model_name}_{iteration:01.4f}.jpg")
    else:
        plt.show()


def evaluate_and_show_attention(model, input_text, tokenizer, iteration=0, to_file=Config.on_server):
    output_words, attentions = evaluate(model, input_text, tokenizer)
    encoded_input = [tokenizer.decode([w]) for w in tokenizer.encode(input_text)]
    print('\ninput = ', ' '.join(encoded_input))
    print('output =', ' '.join(output_words))
    show_attention(encoded_input, output_words, attentions, iteration, to_file)


def calc_rouge(hyps, refs):
    try:
        rouge = Rouge()
        scores = rouge.get_scores(hyps, refs, avg=True)
        rouge1 = scores["rouge-1"]["f"]
        rouge2 = scores["rouge-2"]["f"]
        rougel = scores["rouge-l"]["f"]
        return rouge1, rouge2, rougel
    except Exception as e:
        return 0, 0, 0
