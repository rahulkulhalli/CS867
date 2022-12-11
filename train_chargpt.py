"""
Trains a character-level language model.

Credits: Andrej Karpathy's minGPT GitHub implementation (https://github.com/karpathy/minGPT)
"""

import os
import sys

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

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
    with open(model_name_or_path, 'wb'):
        torch.save(config)

    print("Saved model checkpoint and history.")


if __name__ == '__main__':

    # get default config and overrides from the command line, if any
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    # print(config)
    corpus_name = "CrimeAndPunishment"
    setup_logging(config)
    set_seed(config.system.seed)

    # construct the training dataset
    text = open('data/c_and_p.txt', 'r').read() # don't worry we won't run out of file handles
    train_dataset = CharDataset(config.data, text)

    # construct the model
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    model = GPT(config.model)

    # construct the trainer object
    trainer = Trainer(config.trainer, model, train_dataset)

    # iteration callback
    def batch_end_callback(trainer):

        if trainer.iter_num % 10 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

        if trainer.iter_num % 500 == 0:
            # evaluate both the train and test score
            print(sample(model, "Alas, ", trainer.stoi, trainer.itos, trainer.device))

            # save the latest model
            ckpt_path = os.path.join(config.system.work_dir, f"model_iter{trainer.iter_num}_{corpus_name}.pt")
            save_model(
                model,
                mpath=ckpt_path,
                stoi=trainer.stoi,
                itos=trainer.itos,
                train_hx=
            )

            # revert model to training mode
            model.train()

    trainer.set_callback('on_batch_end', batch_end_callback)

    # run the optimization
    trainer.run()
