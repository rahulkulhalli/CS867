import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size: int, n_hidden: int = 8, embedding_dims: int = 8):
        super(SimpleLSTM, self).__init__()
        
        self.vocab_size = vocab_size
        self.n_hidden = n_hidden
        self._embedding_dims = embedding_dims
        
        # embedding matrix will be (vocab, expected_dims). We fine-tune this during backprop too.
        self.embedding = torch.randn((self.vocab_size, self._embedding_dims), requires_grad=True)
        self.lstm = nn.LSTM(input_size=self._embedding_dims, hidden_size=self.n_hidden, num_layers=1, bias=False, batch_first=True)
        self.hidden2vocab = nn.Linear(self.n_hidden, vocab_size)
    
    def init_hidden(self, batch_size: int):
        # create two zero vectors.
        # the dims are (1*num_layers, batch_size, num_hidden)
        h0 = torch.randn((1, batch_size, self.n_hidden), requires_grad=True)
        c0 = torch.randn((1, batch_size, self.n_hidden), requires_grad=True)
        return h0, c0
        
    def forward(self, x, hidden):
        emb_x = self.embedding[x]                                   # (b, seq) -> (b, seq, features)
        out, (h_new, c_new) = self.lstm(emb_x, hidden)
        
        # out[:, -1, :] == h[0, :, :]
        last_timestep_output = out[:, -1, :].view(-1, self.n_hidden)
        
        linear = self.hidden2vocab(last_timestep_output)
        
        return linear, (h_new, c_new)
