import torch


def load_checkpoint(model, filename, map_location="cpu", strict=False):
    checkpoint = torch.load(filename, map_location=map_location)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    return model.load_state_dict(state_dict, strict=strict)
