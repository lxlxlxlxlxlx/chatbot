import json
import numpy as np
from nltk_utils import *
from model import NN
from time import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
Y_train = []

for x, y in xy:
    bag = bag_of_words(x, all_words)
    X_train.append(bag)
    label = tags.index(y) # lable
    Y_train.append(label) # Cross entropy loss

X_train = np.array(X_train)
Y_train = np.array(Y_train)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# Hyperparameters
batch_size = 8
learning_rate = 0.001
num_epochs = 1000
hidden_size = 8
output_size = len(tags)
input_size = len(X_train[0])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)

model = NN(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

start_t = time()
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        #forward
        outputs = model(words)
        loss = criterion(outputs, labels)

        #backward and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1)%100 == 0:
        print(f'epoch{epoch+1}/{num_epochs}, loss={loss.item():.4f}')

print(f'final loss, loss={loss.item():.4f}, time cost {(time()-start_t)/60} minutes')

data = {
    "model_state": model.state_dict(),
    "input_size" : input_size,
    "output_size" : output_size,
    "hidden_size" : hidden_size,
    "all_words" : all_words,
    "tags" : tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'traning complete. model saved to {FILE}')