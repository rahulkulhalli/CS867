import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size: int, n_hidden: int = 8, embedding_dims: int = 8):
        super(SimpleLSTM, self).__init__()

        self.vocab_size = vocab_size
        self.n_hidden = n_hidden
        # self._embedding_dims = embedding_dims

        # embedding matrix will be (vocab, expected_dims). We fine-tune this during backprop too.
        # self.embedding = torch.randn((self.vocab_size, self._embedding_dims), requires_grad=True)
        self.lstm = nn.LSTM(input_size=self.vocab_size, hidden_size=self.n_hidden, num_layers=3, bias=True, batch_first=True, dropout=0.1)
        self.dpt = nn.Dropout(p=0.2)
        self.hidden2vocab = nn.Linear(self.n_hidden, vocab_size)

    def init_hidden(self, num_layers: int, batch_size: int):
        # create two zero vectors.
        # the dims are (1*num_layers, batch_size, num_hidden)
        h0 = torch.zeros((num_layers, batch_size, self.n_hidden), requires_grad=True)
        c0 = torch.zeros((num_layers, batch_size, self.n_hidden), requires_grad=True)
        return h0, c0

    def forward(self, x, hidden):
        # emb_x = self.embedding[x]
        out, (h_new, c_new) = self.lstm(x, hidden)

        # pass it through dropout.
        out = self.dpt(out)

        # Output will be (batch_size, num_layers, hidden_size)
        out = out.contiguous().view(out.size(0)*a.size(1), self.n_hidden)

        # pass it through the linear layer.
        out = self.hidden2vocab(out)

        # print(out.size())

        return out, (h_new, c_new)
