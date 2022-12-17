import argparse
from pathlib import Path
import os
from src.utils.text_utils import TextUtils
from models.markov_model import MarkovModel


def get_markov_ouput(data_path: Path, seed_string: str, model_type: str = 'char'):
    txt_utils = TextUtils(data_path, compute_counts=True, model_type='char', get_from_cache=True)

    if seed_string[-1] == " ":
        last_char = seed_string[-2]
    else:
        last_char = seed_string[-1]

    # our character level Markov Model. We also allow smoothing/temperature to induce some fuzziness in sampling.
    markov_model = MarkovModel(txt_utils)
    text, nll = markov_model.random_chain(last_char, 500)

    return seed_string + text
