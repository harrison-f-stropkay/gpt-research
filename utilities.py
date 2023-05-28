# utility functions

import numpy as np
import torch


def load_cls_tensor(prompt_number, cap):
    return torch.load(f"cls_tensors/cls_prompt{prompt_number:03d}_cap{cap:03d}.pt")

def get_labels():
    return np.concatenate((np.zeros(20, dtype=int), np.ones(20, dtype=int)))