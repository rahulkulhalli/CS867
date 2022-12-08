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
    
    txt_utils = TextUtils(Path("data/c_and_p.txt"), compute_counts=False, model_type='word') 
    
    train_dataset = DatasetForLSTM(txt_utils, window_size=window_size, model_type='word', stride=1, mode='train')
    test_dataset = DatasetForLSTM(txt_utils, window_size=window_size, model_type='word', stride=1, mode='val')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    TRAIN = True
    
    # batch_first is set to True.
    model = SimpleLSTM (
        vocab_size=len(txt_utils.word2ix),
        n_hidden=n_hidden_features,
        embedding_dims=n_embedding_dims
    )
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.NLLLoss()
    
    if TRAIN:
        
        mean_loss = []
        for epoch_ix in range(1, n_epochs+1):
            if epoch_ix > 1:
                # reduce lr by 20%.
                optimizer.param_groups[0]['lr'] *= 0.9
                
            h = model.init_hidden(batch_size=batch_size)
            
            for iter_ix, (x, y) in enumerate(train_loader):
                
                h = tuple([e.data for e in h])
            
                optimizer.zero_grad()
                
                # (out -> (b, seq_len, n_features))
                logits, h = model(x, h)
                
                out = F.log_softmax(logits, dim=1)
                
                loss = criterion(out, y)
                
                loss.backward()
                
                mean_loss.append(loss.detach().item())
                
                if (iter_ix+1) % 5000 == 0:
                    print(f"epoch: {epoch_ix}, iteration: {iter_ix+1}, mean loss: {torch.tensor(mean_loss).mean()}")
                        
                optimizer.step()
        
        # save the model.
        torch.save(model.state_dict(), Path("models/lstm.pth"))
        
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
                
                if (iter_ix+1) % 5000 == 0:
                    print(f"epoch: {epoch_ix}, iteration: {iter_ix+1}, mean loss: {torch.tensor(mean_loss).mean()}")
                    
    else:
        load_path = Path("models/lstm.pth")
        if load_path.exists():
            
            model.load_state_dict(torch.load(load_path, map_location='cpu'))
            print("LSTM Model weights loaded.")
            model.eval()
            
            # sample from the model.
            seed = "i wanted to know how to proceed from h"
            preprocessed_seed = txt_utils.preprocess_live_input(seed)
            dataset = DatasetForLSTM(txt_utils, window_size=window_size, stride=1, mode='test', test_data=preprocessed_seed)
            loader = DataLoader(dataset=dataset, shuffle=False, batch_size=1, drop_last=True)
            
            mean_loss = []
            next_chars = []
            with torch.no_grad():
                for iter_ix, (x, y) in enumerate(loader):
                    h = model.init_hidden(batch_size=1)
                    
                    h = tuple([e.data for e in h])
                    
                    # (out -> (b, seq_len, n_features))
                    logits, h = model(x, h)
                    probs = F.softmax(logits, dim=1).detach().squeeze()
                    
                    out = F.log_softmax(logits, dim=1)
                    
                    # sample instead of argmax for a more realistic output.
                    next_chars.append(torch.multinomial(probs, num_samples=1, replacement=True).item())
                    
                    loss = criterion(out, y)
                    
                    mean_loss.append(loss.detach().item())
                    
                    # print(f"mean loss: {torch.tensor(mean_loss).mean()}")
            
            print(''.join([txt_utils.ix2char[ix] for ix in next_chars]))