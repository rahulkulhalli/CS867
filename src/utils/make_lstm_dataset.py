"""_Generate data for the LSTM model by creating appropriate windowed inputs and corresponding outputs._
"""

import torch
from torch.utils.data import Dataset
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
            self._text_helper.cleaned_text = test_data
        
        self.data = self._make_data()
    
    
    def _make_data(self):
        
        pairs = []
        
        # some sort of switching here for word and char.
        if self._model_type == 'char':    
            words = self._text_helper.cleaned_text
        else:
            words = self._text_helper._words_list
        
        for ix in range(len(words)-self._window_size):
            inp = words[ix:ix+self._window_size]
            output = words[ix+self._window_size]
            
            pairs.append((inp, output))
            ix += self._stride
        
        if ix < len(words):
            buffer = ['.' for _ in range(self._window_size)]
            for i, remaining in enumerate(words[ix:]):
                buffer[i] = remaining
            
            pairs.append((buffer, '.'))
            
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
        x_window, target = self.data[index]
        
        # print(x_window, target)
        
        if len(x_window) == self._window_size:
            
            if self._model_type == 'char':
                x_tensor = torch.tensor([self._text_helper.char2ix[x] for x in x_window])
                y_tensor = torch.tensor(self._text_helper.char2ix[target])
            else:
                x_tensor = torch.tensor([self._text_helper.word2ix[x] for x in x_window])
                y_tensor = torch.tensor(self._text_helper.word2ix[target])
            
            return x_tensor.long(), y_tensor.long()
