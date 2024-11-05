import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def feature_init(X, tau):
    tau = 4
    featrures = torch.zeros((X.shape[0]-tau, tau))
    for i in range(featrures.shape[0]):
        featrures[i] = X[i:i+tau]
    labels = X[tau:].reshape((-1, 1))
    return featrures , labels

def train_loader(features,labels,batch_size,train_size,test_size):
    X_train, X_val, Y_train, Y_val = train_test_split(features,labels,test_size=0.3, random_state=42,shuffle=False)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, Y_train),
        batch_size=batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_val, Y_val),
        batch_size=batch_size, shuffle=False)
    return train_loader,val_loader

net = nn.Sequential(
    nn.Linear(4, 128),
    nn.ReLU(),
    nn.Linear(128, 512),
    nn.ReLU(),
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Linear(128, 1)
)

T = 1000
time = torch.arange(1,T+1,dtype=torch.float32)
X = torch.sin(0.01*time) + torch.normal(0, 0.2, (T,))

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

features,labels = feature_init(X, tau=4)
train_loader,val_loader = train_loader(features,labels,batch_size=16,train_size=0.7,test_size=0.3)
    
def train(net, train_loader, val_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = net(features)
            loss = criterion(outputs, labels)
            loss.sum().backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
    return net
train(net, train_loader, val_loader, criterion, optimizer, num_epochs=100)
