"""_Generate data for the LSTM model by creating appropriate windowed inputs and corresponding outputs._
"""

import torch
from torch.utils.data import Dataset
from torch.nn.functional import one_hot
from src.utils.text_utils import TextUtils


class DatasetForLSTM(Dataset):
    def __init__(self, txt_utils: TextUtils, window_size: int, model_type: str,
                 stride: int = 1, mode: str = 'train',
                 test_data: str = None
                 ) -> None:
        super().__init__()

        self._text_helper = txt_utils
        self._model_type = model_type
        self._window_size = window_size
        self._stride = stride
        self._mode = mode

        if self._mode == 'test':
            self._text_helper.cleaned_text = self._text_helper._preprocess(test_data)

        self.data = self._make_data()


    def _make_data(self):

        """_summary_

        x = i have to really understand this implementation; i keep forgetting it and that's not good.
        y = have to really understand this implementation; i keep forgetting it and that's not good.<EOS>


        Returns:
            _type_: _description_
        """

        pairs = []

        # some sort of switching here for word and char.
        if self._model_type == 'char':
            words = self._text_helper.cleaned_text
        else:
            words = self._text_helper._words_list

        for ix in range(0, len(words)-self._window_size-1, self._window_size):

            inp = words[ix:ix+self._window_size]
            output = words[ix+1:ix+self._window_size+1]

            inp = [self._text_helper.char2ix[c] for c in inp]
            output = [self._text_helper.char2ix[c] for c in output]

            pairs.append((inp, output))

        if self._mode == 'train':
            return pairs[:int(0.8 * len(pairs))]
        elif self._mode == 'val':
            return pairs[int(0.8 * len(pairs)):]
        else:
            return pairs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        # no need to convert to OHE. Instead, directly index into the embedding matrix.
        x, y = self.data[index]

        # print(x_window, target)

        x_tensor = one_hot(torch.tensor(x), num_classes=len(self._text_helper.char2ix)).float()
        y_tensor = torch.tensor(y)

        return x_tensor, y_tensor.long()
