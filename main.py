import argparse
from pathlib import Path
import os
from src.utils import io_utils
from src.utils.text_utils import TextUtils
from models.markov_model import MarkovModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("domain_size", metavar="s", type=int, help="Select the size of the domain")
    parser.add_argument("controller", metavar="p", help="Choose controller [0: Human, 1: Baseline AI, 2: Language Model]", default=1, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    # text utility helper.
    file_path = Path("data/c_and_p.txt")
    txt_utils = TextUtils(file_path, compute_counts=True)

    # our character level Markov Model. We also allow smoothing/temperature to induce some fuzziness in sampling.
    markov_model = MarkovModel(txt_utils)
    text, nll = markov_model.random_chain("i", 500)
    print(f"{text=}\n\n{nll=}")
