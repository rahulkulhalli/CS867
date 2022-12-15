"""
Trains a character-level language model using the GPT model.

Credits: Andrej Karpathy's minGPT implementation (https://github.com/karpathy/minGPT)
"""

import os
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset

from models.mingpt.model import GPT
from models.mingpt.trainer import Trainer
from models.mingpt.utils import set_seed, setup_logging, CfgNode as CN

# -----------------------------------------------------------------------------

def get_config():

    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = './models/'

    # data
    C.data = CharDataset.get_default_config()

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-mini'

    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster
    C.max_iters = 20000

    return C

# -----------------------------------------------------------------------------

class CharDataset(Dataset):
    """
    Emits batches of characters
    """

    @staticmethod
    def get_default_config():
        C = CN()
        C.block_size = 128
        return C

    def __init__(self, config, data):
        self.config = config

        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.vocab_size = vocab_size
        self.data = data

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.config.block_size

    def __len__(self):
        return len(self.data) - self.config.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.config.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        # return as tensors
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

# -----------------------------------------------------------------------------

def sample(model, seed_string, stoi, itos, device):
    model.eval()
    with torch.no_grad():
        # sample from the model...
        x = torch.tensor([stoi[s] for s in seed_string], dtype=torch.long)[None,...].to(device)
        y = model.generate(x, 500, temperature=1.0, do_sample=True, top_k=10)[0]
        generated_str = ''.join([itos[int(i)] for i in y])
        return generated_str


def save_model(model, **kwargs):
    config = {
        "state_dict": model.state_dict(),
        "stoi": kwargs['stoi'],
        "itos": kwargs['itos'],
        "train_history": kwargs['train_hx']
    }

    model_name_or_path = kwargs['mpath']
    with open(model_name_or_path, 'wb') as f:
        torch.save(config, f)

    print("Saved model checkpoint and history.")


if __name__ == '__main__':

    domain_inp = input("Select the domain (Metamorphosis [m], Crime and Punishment [c], Dracula [d], Frankenstein [f]): ")
    if domain_inp != "" and domain_inp.lower() in ["m", "c", "d", "f"]:
        domain = domain_inp
    else:
        # let's default to CrimeAndPunishment.
        domain = "c"

    model_mapper = {
        "c": Path("models/gpt_weights/model_iter8000_CrimeAndPunishment.pt"),
        "d": Path("models/gpt_weights/model_iter34000_Dracula.pt"),
        "m": Path("models/gpt_weights/model_iter9000_Kafka.pt"),
        "f": Path("models/gpt_weights/minGPT_23000_Frankenstein.pt")
    }

    with open(model_mapper[domain], 'rb') as f:
        ckpt_config = torch.load(f, map_location='cpu')

    # # get default config and overrides from the command line, if any
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    setup_logging(config)
    set_seed(config.system.seed)

    # # construct the model
    config.model.vocab_size = len(ckpt_config['stoi'])
    config.model.block_size = 128
    model = GPT(config.model)

    # load the weights and set it to eval mode
    model.load_state_dict(ckpt_config["state_dict"])
    model.eval()
    print("Model weights loaded")

    device = torch.device('cpu')
    seed_string = input("Start writing from here: ")

    # If the user doesn't add whitespace, add manually.
    if seed_string[-1] != " ":
        seed_string += " "

    output = sample(model, seed_string, ckpt_config['stoi'], ckpt_config['itos'], device)
    print(50*'=')
    print(f"Model output: {output}")
    print(50*'=')
