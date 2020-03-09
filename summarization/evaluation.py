import torch
import matplotlib.pyplot as plt
from matplotlib import ticker

from summarization.config import Config
from summarization.progress_bar import ProgressBar

test_text = "Egy ember életét vesztette, amikor két személygépkocsi ütközött az 1-es főúton Bicskénél közölte a Fejér Megyei Rendőr-főkapitányság szóvivője. A balesetben hárman sérültek meg, egy ember olyan súlyosan, hogy a helyszínen meghalt. A rendőrség a helyszínelés idejére teljesen lezárta az érintett útszakaszt, a forgalmat Bicske belterületén keresztül terelik el."


def validate_model(model, criterion, valid_loader, encoder_processor, decoder_processor, loss_printer, summary_writer):
    print("Validating...")
    model.eval()

    progress_bar = ProgressBar(len(valid_loader))
    progress_bar.start()
    total_acc, total_loss, n = 0, 0, 0
    with torch.no_grad():
        for data in valid_loader:
            padded_contents, content_sizes, padded_summaries, _ = data
            padded_contents = padded_contents.to(Config.device)
            padded_summaries = padded_summaries.to(Config.device)
            max_len = padded_summaries.shape[1]
            output, _ = model(padded_contents, content_sizes, padded_summaries, teacher_forcing_ratio=1.0, max_len=max_len)
            loss = criterion(output.view(-1, Config.decoder_token_num), padded_summaries.view(-1))

            total_loss += loss.item()
            n += 1
            progress_bar.update()

        for data in valid_loader:
            padded_contents, content_sizes, padded_summaries, summary_sizes = data

            padded_contents = padded_contents.to(Config.device)

            output, attentions = model(padded_contents, content_sizes, None, teacher_forcing_ratio=0.0)
            output = output.argmax(2).detach().cpu()
            for i in range(min(output.shape[0], 10)):
                print("\nContent:\n" + encoder_processor.decode(padded_contents[i][:content_sizes[i]].tolist()))
                print("Summary:\n" + decoder_processor.decode(padded_summaries[i][:summary_sizes[i]].tolist()))
                print("Prediction:\n" + decoder_processor.decode(output[i].tolist()))
            break

        evaluate_and_show_attention(model,
                                    test_text,
                                    encoder_processor, decoder_processor)

    loss, acc = total_loss / n, total_acc / n


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


def show_attention(input_sentence, output_words, attentions, iteration=0):
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
    plt.show()


def evaluate_and_show_attention(model, input_text, tokenizer, iteration=0):
    output_words, attentions = evaluate(model, input_text, tokenizer)
    encoded_input = [tokenizer.decode([w]) for w in tokenizer.encode(input_text)]
    print('\ninput = ', ' '.join(encoded_input))
    print('output =', ' '.join(output_words))
    show_attention(encoded_input, output_words, attentions, iteration)
