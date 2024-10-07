import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from d2l import torch as d2l
from torch.utils import data
from torch import nn

device = torch.device('cuda')

init_data = pd.read_csv('Excercise_of_DeepLearning_Coursera/BostonHousing.csv')
init_data = init_data.dropna()

X = init_data.iloc[:, :-1].values
Y = init_data.iloc[:, -1].values
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.3, random_state=42)  

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val) 

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).requires_grad_(True)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).requires_grad_(True)
Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32).unsqueeze(1)

device = torch.device('cuda')
X_train_tensor = X_train_tensor.to(device)
Y_train_tensor = Y_train_tensor.to(device)
X_val_tensor = X_val_tensor.to(device)
Y_val_tensor = Y_val_tensor.to(device)

def load_array(init_data_arrays, batch_size, is_train=True):
    init_dataset = data.TensorDataset(*init_data_arrays)
    return data.DataLoader(init_dataset, batch_size, shuffle=is_train)

batch_size = 10
num_epochs = 100
l2_reg = 0.01
num_epochs = 100

init_data_iter = load_array((X_train_tensor, Y_train_tensor), batch_size, is_train=True)

net = nn.Sequential(nn.Linear(X_train_tensor.shape[1], 1)).to(device)
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr=l2_reg)


for epoch in range(num_epochs):
    for X, y in init_data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(X_val_tensor), Y_val_tensor)
    print(f'epoch {epoch + 1}, validation loss {l:f}')

