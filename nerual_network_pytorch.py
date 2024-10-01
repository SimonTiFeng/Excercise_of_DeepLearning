import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        
        layers = []
        hidden_size = 64  
        
        # append the inpute layer structuring as a linear layer with ReLU activation function   
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        
        for _ in range(7):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_size, 1))
        layers.append(nn.Sigmoid())  

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

data = pd.read_csv('Excercise_of_DeepLearning_Coursera/data.csv')
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
data = data.drop(columns=['id'])
data = data.drop(columns=['Unnamed: 32'], errors='ignore')    
Y = data['diagnosis'].values
X = data.drop(columns=['diagnosis']).values
# 将数据集划分为训练集和验证集
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.3, random_state=42)  

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val) 

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32).unsqueeze(1)

device = torch.device('cuda')
X_train_tensor = X_train_tensor.to(device)
Y_train_tensor = Y_train_tensor.to(device)
X_val_tensor = X_val_tensor.to(device)
Y_val_tensor = Y_val_tensor.to(device)

input_size = X_train_tensor.shape[1]
model = NeuralNetwork(input_size).to(device)
criterion = nn.BCELoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 1000
train_losses = []
val_losses = [] 
f1_scores = []
recall_scores = []
precision_scores = []

for epoch in range(epochs):
    model.train()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, Y_train_tensor)

    l2_lambda = 0.001
    l2_norm = sum(p.pow(2).sum() for p in model.parameters())
    loss += l2_lambda * l2_norm
    
    optimizer.zero_grad()  
    loss.backward()  
    optimizer.step()  
    
    train_losses.append(loss.item())  
    model.eval() 
    with torch.no_grad():
        val_outputs = model(X_val_tensor)  
        val_loss = criterion(val_outputs, Y_val_tensor)  
        val_losses.append(val_loss.item())  
        
        precision = (val_outputs >= 0.5).float().mean().item()
        recall = (val_outputs[Y_val_tensor == 1] >= 0.5).float().mean().item()
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1 = 2 * precision * recall / (precision + recall+ 1e-10)
        f1_scores.append(f1)
    
    if val_losses[epoch] > val_losses[epoch-1] and epoch > 0:
        print(f'Early Stopping at Epoch {epoch}')
        break

    if epoch % 999 == 0:
        print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')

    with torch.no_grad():
        model.eval()  
        y_val_pred = model(X_val_tensor)
        y_val_pred = (y_val_pred >= 0.5).float()

        accuracy = (y_val_pred == Y_val_tensor).float().mean().item()
        print(f'Validation Accuracy: {accuracy:.4f}')

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')  
plt.plot(f1_scores, label='F1 Score')
plt.plot(precision_scores, label='Precision Score')
plt.plot(recall_scores, label='Recall Score')
plt.title('Learning Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()