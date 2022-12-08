import torch
from torch.utils.data import Dataset
from torch.nn.functional import one_hot


class TextDataset(Dataset):
    def __init__(self, txt_helper):
        self._helper = txt_helper
        self.data = self._make_data()

    def _make_data(self):
        text = self._helper.cleaned_text
        data = []
        for (c1, c2) in zip(text, text[1:]):
            data.append((self._helper.char2ix[c1], self._helper.char2ix[c2]))

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ix):
        x, y = self.data[ix]

        x_tensor = torch.tensor(x)
        y_tensor = torch.tensor(y)

        x_onehot = one_hot(x_tensor, num_classes=len(self._helper.char2ix)).float()

        return x_onehot, y_tensor.long()
