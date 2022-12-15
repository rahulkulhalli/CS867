from src.utils.make_lstm_dataset import DatasetForLSTM
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from src.utils.text_utils import TextUtils
from models.lstm import SimpleLSTM
from pathlib import Path


def generate_next_char(model, char, ix2char, char2ix, h=None):
    """
    Given a character, predict the next character.
    """

    model.eval()

    with torch.no_grad():
        x = F.one_hot(torch.tensor(char2ix[char]), num_classes=len(char2ix)).float().reshape((1, 1, -1))

        if not h:
            h = model.init_hidden(batch_size=1)

        h = tuple([_h.data for _h in h])

        out, h = model(x, h)

        probs = F.softmax(out.detach(), dim=-1).squeeze()

        arg = torch.multinomial(probs, num_samples=1, replacement=True).item()
        # print(arg)

        return ix2char[arg], h


def sample(net, seed_string, ix2char, char2ix, max_generation_len=1000):
    # First, convert all characters in the seed string to their encodings.
    # We warm-up the hidden state of the LSTM by passing the input string.
    outputs = []

    h = net.init_hidden(batch_size=1)
    for char in seed_string:
        next_char, h = generate_next_char(net, char, ix2char, char2ix, h)

    # Only the (n+1)th character of `seed_string` is relevant.
    outputs.append(next_char)

    # autoregressively generate the next character conditioned on the previous character.
    for ix in range(max_generation_len):
        next_char, h = generate_next_char(net, outputs[-1], ix2char, char2ix, h)
        outputs.append(next_char)

    return ''.join(outputs)


if __name__ == "__main__":

    domain_inp = input("Select the domain (Metamorphosis [m], Crime and Punishment [c], Dracula [d], Frankenstein [f]): ")
    if domain_inp != "" and domain_inp.lower() in ["m", "c", "d", "f"]:
        domain = domain_inp
    else:
        # let's default to CrimeAndPunishment.
        domain = "c"

    model_mapper = {
        "c": Path("models/lstm_weights/lstm_final_CrimeAndPunishment.pt"),
        "d": Path("models/lstm_weights/lstm_final_Dracula.pt"),
        "m": Path("models/lstm_weights/lstm_final_Kafka.pt"),
        "f": Path("models/lstm_weights/lstm_final_Frankenstein.pt")
    }

    with open(model_mapper[domain], 'rb') as f:
        config = torch.load(f, map_location='cpu')

    # batch_first is set to True.
    model = SimpleLSTM (
        vocab_size=len(config['ix2char']),
        n_hidden=config['n_hidden'],
        embedding_dims=64
    )

    # load the weights.
    model.load_state_dict(config['state_dict'])
    print("Model weights loaded!")
    model.eval()

    seed_string = input("Start writing from here: ")

    # If the user doesn't add whitespace, add manually.
    if seed_string[-1] != " ":
        seed_string += " "

    output_string = sample(model, seed_string, config['ix2char'], config['char2ix'])
    print(50*'=')
    print("Generated text")
    print(50*'=')
    print(seed_string + output_string)
