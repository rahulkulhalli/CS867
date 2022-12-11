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
        self.lstm = nn.LSTM(input_size=self.vocab_size, hidden_size=self.n_hidden, num_layers=1, bias=True, batch_first=True)
        self.hidden2hidden = nn.Linear(self.n_hidden, self.n_hidden)
        self.dpt = nn.Dropout(p=0.2)
        self.hidden2vocab = nn.Linear(self.n_hidden, vocab_size)

    def init_hidden(self, batch_size: int):
        # create two zero vectors.
        # the dims are (1*num_layers, batch_size, num_hidden)
        h0 = torch.zeros((1, batch_size, self.n_hidden), requires_grad=True)
        c0 = torch.zeros((1, batch_size, self.n_hidden), requires_grad=True)
        return h0, c0

    def forward(self, x, hidden):
        # emb_x = self.embedding[x]
        out, (h_new, c_new) = self.lstm(x, hidden)
        out = out.contiguous().view(-1, self.n_hidden)
        out = self.dpt(out)
        out = self.hidden2hidden(out)
        out = self.dpt(out)
        out = self.hidden2vocab(out)

        # print(out.size())

        return out, (h_new, c_new)
