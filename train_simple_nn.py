from pathlib import Path
import torch.optim as optim
import torch.nn as nn
from torch.nn.functional import one_hot

from models.simple_nn import SimpleNN
from src.utils.make_nn_dataset import TextDataset
from torch.utils.data import DataLoader
from src.utils.text_utils import TextUtils
import torch


if __name__ == "__main__":
    text_helper = TextUtils(Path("data/c_and_p.txt"))

    model = SimpleNN(vocab_size=len(text_helper.char2ix))
    dataset = TextDataset(text_helper)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    TRAIN = True

    if TRAIN:
        # initialize training loop.
        for epoch_ix in range(4):
            for iter_ix, (x, y_true) in enumerate(dataloader):

                optimizer.zero_grad()

                y_logits = model(x)

                # softmax across rows.
                # y_probs = torch.softmax(y_logits, dim=1)

                loss = criterion(y_logits, y_true)

                if (iter_ix+1)%1000==0:
                    print(f"{iter_ix=} => {loss.item()}")

                loss.backward()
                optimizer.step()

            model.eval()
            # try sampling.
            seed_char = 'g'
            output_string = [seed_char]
            with torch.no_grad():
                for _ in range(500):
                    input_enc = one_hot(torch.tensor(text_helper.char2ix[seed_char]), num_classes=len(text_helper.char2ix)).float().reshape((1, -1))
                    outputs = model(input_enc)
                    prob_outputs = torch.softmax(outputs, dim=1).detach().squeeze()
                    out_ix = torch.multinomial(prob_outputs, num_samples=1, replacement=True)
                    seed_char = text_helper.ix2char[out_ix.item()]
                    output_string.append(seed_char)

            print(''.join(output_string))

            # set back to train
            model.train()
