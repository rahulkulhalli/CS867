from src.utils.make_lstm_dataset import DatasetForLSTM
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from src.utils.text_utils import TextUtils
from models.lstm import SimpleLSTM
from pathlib import Path


if __name__ == "__main__":

    batch_size = 8
    n_epochs = 5
    window_size = 4
    n_hidden_features = 128
    n_embedding_dims = 64
    save_every = 2

    txt_utils = TextUtils(Path("data/c_and_p.txt"), compute_counts=False, model_type='char')

    train_dataset = DatasetForLSTM(txt_utils, window_size=window_size, model_type='char', stride=1, mode='train')
    test_dataset = DatasetForLSTM(txt_utils, window_size=window_size, model_type='char', stride=1, mode='val')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    TRAIN = True

    # batch_first is set to True.
    model = SimpleLSTM (
        vocab_size=len(txt_utils.char2ix),
        n_hidden=n_hidden_features,
        embedding_dims=n_embedding_dims
    )

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    if TRAIN:

        mean_loss = []
        for epoch_ix in range(1, n_epochs+1):
            if epoch_ix > 1:
                # reduce lr by 20%.
                optimizer.param_groups[0]['lr'] *= 0.95

            h = model.init_hidden(batch_size=batch_size)

            for iter_ix, (x, y) in enumerate(train_loader):

                h = tuple([e.data for e in h])

                optimizer.zero_grad()

                # (out -> (b, seq_len, n_features))
                logits, h = model(x, h)

                seq_loss = torch.tensor(0., requires_grad=True)

                for i in range(window_size):
                    seq_loss = seq_loss + criterion(logits[:, i, :], y[:, i])

                seq_loss.backward()

                mean_loss.append(seq_loss.detach().item())

                if (iter_ix+1) % 5000 == 0:
                    print(f"epoch: {epoch_ix}, iteration: {iter_ix+1}, mean loss: {torch.tensor(mean_loss).mean()}")

                optimizer.step()

            if epoch_ix % save_every == 0:
                print("Saving LSTM model.")
                torch.save(model.state_dict(), Path(f"models/lstm_{epoch_ix}.pt"))

        # save the model.
        torch.save(model.state_dict(), Path("models/lstm_final.pth"))

        print(50*'+')
        print("TESTING PERFORMANCE...")
        print(50*'+')

        mean_loss = []
        with torch.no_grad():
            for iter_ix, (x, y) in enumerate(test_loader):
                h = model.init_hidden(batch_size=batch_size)

                # print(x.shape, y.shape)

                h = tuple([e.data for e in h])

                # (out -> (b, seq_len, n_features))
                logits, h = model(x, h)

                out = F.log_softmax(logits, dim=1)

                loss = criterion(out, y)

                mean_loss.append(loss.detach().item())

                if (iter_ix+1) % 250 == 0:
                    print(f"epoch: {epoch_ix}, iteration: {iter_ix+1}, mean loss: {torch.tensor(mean_loss).mean()}")
