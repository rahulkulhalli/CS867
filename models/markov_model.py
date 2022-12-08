import string
import re
import torch
import os
from src.utils.text_utils import TextUtils


class MarkovModel:
    def __init__(self, txt_utils: TextUtils):
        self._text_utils = txt_utils
        self._char2ix = self._text_utils.char2ix
        self._ix2char = self._text_utils.ix2char
        self._P = self._text_utils.get_P_table()

    def get_next_most_char(self, char):
        preprocessed_char = self._text_utils._preprocess(char)
        if preprocessed_char in self._char2ix.keys():
            # pick the row index.
            row_ix = self._char2ix[preprocessed_char]
            return self._ix2char[self._P[row_ix, :].argmax()]

    def sample_next_most_likely(self, char, smoothing=1):
        preprocessed_char = self._text_utils._preprocess(char)
        if preprocessed_char in self._char2ix.keys():

            # pick the row index.
            row_ix = self._char2ix[preprocessed_char]

            # Pick the row. We smooth to add fake counts to the row and avoid zeros.
            row = self._P[row_ix] + smoothing

            # normalize the vector to probability distributions.
            row_norm = row/row.sum()

            # sample the index from the distribution.
            sampled_ix = torch.multinomial(row_norm, num_samples=1, replacement=True).item()

            # return char at index.
            return self._ix2char[sampled_ix], torch.log(row_norm[sampled_ix]).item()


    def random_chain(self, seed_char, chain_length=500):
        preprocessed_char = self._text_utils._preprocess(seed_char)
        nll = 0.

        if preprocessed_char in self._char2ix.keys():
            next_char, prob = self.sample_next_most_likely(seed_char)
            nll += prob
            chars = [seed_char, next_char]

            for _ in range(chain_length-1):
                next_char, prob = self.sample_next_most_likely(next_char)
                nll += prob

                chars.append(next_char)

        return "".join(chars), -(nll/chain_length)
