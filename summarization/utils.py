import torch
from summarization import Config


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths)
    ids = torch.arange(0, max_len, device=lengths.device).long()
    return ids >= lengths.unsqueeze(1)


def save_model(model, optimizer, epoch):
    print("Saving model...")
    torch.save(model.state_dict(), f"{Config.data_path}/models/{Config.model_name}_{str(epoch)}.pth")
    torch.save(optimizer.state_dict(), f"{Config.data_path}/models/{Config.model_name}_{str(epoch)}_optimizer_state.pth")

def load_model(model, optimizer, epoch):
    print("Loading model...")
    model.load_state_dict(torch.load(f"{Config.data_path}/models/{Config.model_name}_{str(epoch)}.pth"))
    optimizer.load_state_dict(torch.load(f"{Config.data_path}/models/{Config.model_name}_{str(epoch)}_optimizer_state.pth"))
