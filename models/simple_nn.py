import torch.nn as nn
import torch.nn.functional as F


class SimpleNN(nn.Module):
    def __init__(self, vocab_size):
        super(SimpleNN, self).__init__()
        # (1, 27) (27, 27) = (1, 27)
        self.hidden = nn.Linear(vocab_size, 128, bias=False)
        self.linear1 = nn.Linear(128, vocab_size, bias=True)

    def forward(self, x):
        z = F.relu(self.hidden(x))
        # return logits.
        return self.linear1(z)
