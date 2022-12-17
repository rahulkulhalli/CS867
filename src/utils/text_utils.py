import os
import re
import torch
import string
from pathlib import Path


class TextUtils:
    def __init__(
        self,
        fpath: Path,
        compute_counts: bool = False,
        model_type: str = 'char',
        do_cache: bool = False,
        get_from_cache: bool = False
    ):

        self._path = fpath

        with open(fpath, "r") as f:
            data = f.read()

        self.cleaned_text = self._preprocess(data)
        self._model_type = model_type
        self._cache_ptable = do_cache
        self._retrieve_cached = get_from_cache

        self._words_list = [x.replace(".", "") for x in self.cleaned_text.split() if x!=''] + ['.']
        self._unique_char_list = sorted(set(self.cleaned_text))

        if self._model_type == 'char':
            self.char2ix = {v:k for (k,v) in enumerate(self._unique_char_list)}
            self.ix2char = {v:k for (k,v) in self.char2ix.items()}
            self.word2ix = None
            self.ix2word = None
        else:
            self.char2ix = None
            self.ix2char = None
            self.word2ix = {v:k for (k,v) in enumerate(set(self._words_list))}
            self.ix2word = {v:k for (k,v) in self.word2ix.items()}

        # print("Unique words: ", len(set(self._words_list)))
        # print("Unique chars: ", len(set(self._unique_char_list)))

        if compute_counts:
            # time-consuming stuff here.
            self._P = self._create_ptable()
        else:
            self._P = None

    def get_cleaned_text(self):
        return self.cleaned_text

    def _preprocess(self, data):
        data_str = data.lower()
        # data_str = re.sub(pattern="\s+", repl=" ", string=data_str, flags=re.MULTILINE).strip()

        # don't remove the period.
        # data_str = re.sub('[!"\\#\\$%\\&\'\\(\\)\\*\\+,\\-/:;<=>\\?@\\[\\\\\\]\\^_`\\{\\|\\}\\~]+', repl="", string=data_str, flags=re.MULTILINE)

        # some more ugly punctuation
        # data_str = data_str.replace("“", "").replace("”", "")

        # lastly, remove the 2+ periods.
        # least....he -> least he
        # data_str = re.sub("\\.{2,}", repl=" ", string=data_str)

        return data_str

    def _create_ptable(self):

        cache_path = Path("models/markov_weights/")
        filename = self._path.name.replace(".txt", ".pt")

        if self._retrieve_cached and filename in os.listdir(cache_path):
            with open(cache_path / filename, "rb") as f:
                p = torch.load(f)
        else:
            num_dims = len(self.char2ix.keys())

            p = torch.zeros((num_dims, num_dims))

            for (c1, c2) in zip(self.cleaned_text, self.cleaned_text[1:]):
                p[self.char2ix[c1], self.char2ix[c2]] += 1.

        if self._cache_ptable:
            # Determine which data is being read. I know hardcoding the path is bad but I'll need
            # to re-write the whole class to organize everything properly.
            with open(cache_path / filename, "wb") as f:
                torch.save(p, f)

            print(f"Cached PTable at {cache_path / filename}")

        return p

    def preprocess_live_input(self, input_stream: str):
        return self._preprocess(input_stream)

    def get_P_table(self):
        return self._P
