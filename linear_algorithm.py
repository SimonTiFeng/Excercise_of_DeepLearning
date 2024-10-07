import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv('Excercise_of_DeepLearning_Coursera/BostonHousing.csv')
data = data.dropna()

X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values
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

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    np.random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num_examples)], device=features.device)
        yield features[batch_indices], labels[batch_indices]


w = torch.nn.Parameter(torch.normal(mean=0, std=0.1, size=(X_train.shape[1], 1), device=device),requires_grad=True)
b = torch.nn.Parameter(torch.zeros(1, device=device),requires_grad=True)


def squared_loss(y_pred, y_true):
    return (y_pred - y_true)**2

def sgd(params, lr, batch_size):
    with torch.no_grad(): 
        for param in params:
            param.data -= lr * param.grad / batch_size
            param.grad.zero_()


learning_rate = 0.1
num_epochs = 1000
batch_size = 32
loss_fn = squared_loss

for epoch in range(num_epochs):
    for X, Y in data_iter(batch_size, X_train_tensor, Y_train_tensor):
        l = loss_fn(torch.matmul(X, w) + b, Y)
        l_sum = l.sum()
        l_sum.backward()
        sgd([w, b], learning_rate, batch_size)
        
    with torch.no_grad():
        train_l = loss_fn(torch.matmul(X_train_tensor, w) + b, Y_train_tensor).mean()
        val_l = loss_fn(torch.matmul(X_val_tensor, w) + b, Y_val_tensor).mean()
    
    if epoch % 100 == 0:
        print(f'epoch {epoch + 1}, train loss: {train_l.item() / X_train.shape[0]:.6f}, val loss: {val_l.item() / X_val.shape[0]:.6f}')

plt.scatter(X_train[:, 5], Y_train, c='b')
plt.plot(X_train[:, 5], (torch.matmul(X_train_tensor, w)+b).cpu().detach().numpy(), c='r')
plt.show()
