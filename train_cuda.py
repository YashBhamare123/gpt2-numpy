import cupy as np
import requests
from tqdm import tqdm

import src.cuda.nn as nn
import src.cuda.optim as optim
from src.cuda.loader import DataLoader   
from src.cuda.model import get_gpt2

dataset = requests.get('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt').text

# get unique characters
chars = sorted(list(set(dataset)))

# vocab
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

def encode(s):
    return np.array([stoi[c] for c in s], dtype=np.int32)

def decode(arr):
    return ''.join([itos[i] for i in arr])

dataset = encode(dataset)

train_dl = DataLoader(dataset[1024:10000], 32, 16)
val_dl = DataLoader(dataset[:1024], 16, 16)
test_seq = "Romeo: "
test_arr = encode(test_seq)


def train(model : nn.Pipeline, epochs, train_dl: DataLoader, val_dl : DataLoader, optimizer: optim.Adam):
    cross_entropy = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        for x, labels in tqdm(train_dl):

            labels = np.expand_dims(labels, -1)
            print(labels.shape)

            out = model(x)
            print(out.shape)
            out_loss = cross_entropy.forward(out, labels)

            d_Ly = cross_entropy.backward()
            print(d_Ly.shape)
            model.backward(d_Ly)
            optimizer.step()
            optimizer.zero_grad()
            
            print(f"training loss : {out_loss}")
        
        model.eval()
        loss = 0
        len_val = len(val_dl)
        for x, labels in tqdm(val_dl):
            out = model(x)
            labels = np.expand_dims(labels, -1)
            loss = loss + cross_entropy.forward(out, labels)
            print(f"validation loss {loss / len_val}")

        print(f"Done with epoch {epoch}")



pipe = get_gpt2()
optimizer = optim.Adam(pipe.parameters(), 1e-4, 0.9, 0.98)
epochs = 10

train(pipe, epochs, train_dl, val_dl, optimizer)

