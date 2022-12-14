from src.utils.make_lstm_dataset import DatasetForLSTM
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from src.utils.text_utils import TextUtils
from models.lstm import SimpleLSTM
from pathlib import Path

def save_model(model_name_or_path: Path, **kwargs):
    config = {
        "state_dict": kwargs['state_dict'],
        "n_hidden": kwargs['n_hidden_features'],
        "tokens": kwargs['unique_char_list'],
        "ix2char": kwargs['ix2char'],
        "char2ix": kwargs['char2ix'],
        "train_history": kwargs['train_history'],
        "val_history": kwargs['val_history']
    }

    with open(model_name_or_path, "wb") as f:
        torch.save(config, f)


def save_plots(plot_name_or_path, train_hx, val_hx):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(nrows=1, ncols=1)

    ax.plot(train_hx, 'b-')
    ax.plot(val_hx, 'r-')

    plt.legend(['Training loss', 'Validation loss'])
    plt.xlabel('Iterations')
    plt.ylabel('CE Loss')

    plt.savefig(plot_name_or_path)
    print("Plots saved.")


if __name__ == "__main__":

    batch_size = 8
    n_epochs = 4
    window_size = 50
    n_hidden_features = 256
    n_embedding_dims = 64
    save_every = 2

    domain = "CrimeAndPunishment"

    txt_utils = TextUtils(Path("data/c_and_p.txt"), compute_counts=False, model_type='char')

    train_dataset = DatasetForLSTM(txt_utils, window_size=window_size, model_type='char', stride=1, mode='train')
    test_dataset = DatasetForLSTM(txt_utils, window_size=window_size, model_type='char', stride=1, mode='val')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # batch_first is set to True.
    model = SimpleLSTM (
        vocab_size=len(txt_utils.char2ix),
        n_hidden=n_hidden_features,
        embedding_dims=n_embedding_dims
    )

    print(f"Total model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    train_history, val_history = list(), list()
    mean_loss = list()
    for epoch_ix in range(1, n_epochs+1):

        model.train()

        print("=================== Training ===================")

        if (epoch_ix % 2) == 0:
            # reduce lr by 5%.
            optimizer.param_groups[0]['lr'] *= 0.95

        h = model.init_hidden(batch_size=batch_size)

        for iter_ix, (x, y) in enumerate(train_loader):

            h = tuple([e.data for e in h])

            optimizer.zero_grad()

            # (out -> (b, seq_len, n_features))
            logits, h = model(x, h)

            loss = criterion(logits, y.view(batch_size * window_size))
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 5.)

            mean_loss.append(loss.detach().item())
            train_history.append(loss.detach().item())

            if (iter_ix+1) % 1000 == 0:
                print(f"epoch: {epoch_ix}, iteration: {iter_ix+1}, mean train loss: {torch.tensor(mean_loss).mean()}")

            optimizer.step()

        if epoch_ix % save_every == 0:
            print("Saving LSTM model.")
            save_model(
                Path(f"models/lstm_weights/lstm_{domain}_{epoch_ix}.pt"),
                state_dict=model.state_dict(),
                n_hidden_features=n_hidden_features,
                unique_char_list=txt_utils._unique_char_list,
                ix2char=txt_utils.ix2char,
                char2ix=txt_utils.char2ix,
                train_history=train_history,
                val_history=val_history
            )

        print("=================== Validation ===================")

        model.eval()

        mean_val_loss = list()
        with torch.no_grad():
            val_h = model.init_hidden(batch_size=batch_size)

            for iter_ix, (x, y) in enumerate(test_loader):

                val_h = tuple([e.data for e in val_h])

                # (out -> (b, seq_len, n_features))
                logits, val_h = model(x, val_h)

                val_loss = criterion(logits, y.view(batch_size * window_size))

                mean_val_loss.append(val_loss.detach().item())
                val_history.append(val_loss.detach().item())

                if (iter_ix+1) % 1000 == 0:
                    print(f"epoch: {epoch_ix}, iteration: {iter_ix+1}, mean val loss: {torch.tensor(mean_val_loss).mean()}")

    # save final model
    print("Saving final model..")
    save_model(
        Path(f"models/lstm_weights/lstm_final_{domain}.pt"),
        state_dict=model.state_dict(),
        n_hidden_features=n_hidden_features,
        unique_char_list=txt_utils._unique_char_list,
        ix2char=txt_utils.ix2char,
        char2ix=txt_utils.char2ix,
        train_history=train_history,
        val_history=val_history
    )

    # save plots.
    # save_plots(Path("models/lstm_final_plots.png"), train_history, val_history)
