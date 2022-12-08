import torch.nn as nn


class SimpleNN(nn.Module):
    def __init__(self, vocab_size):
        super(SimpleNN, self).__init__()
        # (1, 27) (27, 27) = (1, 27)
        self.linear1 = nn.Linear(vocab_size, vocab_size, bias=False)

    def forward(self, x):
        # return logits.
        return self.linear1(x)
