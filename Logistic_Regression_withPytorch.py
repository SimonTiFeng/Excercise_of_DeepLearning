import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import time

covertype = pd.read_csv('Excercise_of_MachineLearning_Zhouzhihua/covtype.data.gz', compression='gzip')

y = covertype.iloc[:, -1]  
X = covertype.iloc[:, :-1]

X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=0.3, random_state=42)  
scaler = StandardScaler()
Y_val = Y_val -1
Y_train = Y_train -1

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val) 

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train.values, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
Y_val_tensor = torch.tensor(Y_val.values, dtype=torch.long)

device = torch.device('cuda')
X_train_tensor = X_train_tensor.to(device)
Y_train_tensor = Y_train_tensor.to(device)
X_val_tensor = X_val_tensor.to(device)
Y_val_tensor = Y_val_tensor.to(device)

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)

train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

criterion = nn.CrossEntropyLoss()
input_size = X_train_tensor.shape[1]
num_classes = len(y.unique())
model = LogisticRegressionModel(input_size, num_classes).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

epochs = 10
patience = 100
best_loss = float('inf')
patience_counter = 0
train_losses = []
val_losses = []

for epoch in range(epochs):
    start_time = time.time()
    model.train()
    train_loss = 0
    for X_batch, Y_batch in train_loader:
        outputs = model(X_batch)
        loss = criterion(outputs, Y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_losses.append(train_loss / len(train_loader))

    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for X_val_batch, Y_val_batch in val_loader:
            val_outputs = model(X_val_batch)
            val_loss += criterion(val_outputs, Y_val_batch).item()
            _, predicted_classes = torch.max(val_outputs, 1)
            correct += (predicted_classes == Y_val_batch).sum().item()
    
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    accuracy = correct / len(Y_val_tensor)

    if val_loss < best_loss:
        best_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break
    end_time = time.time()
    print(f'Epoch {epoch} time: {end_time - start_time:.2f} seconds')
    print(f'训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}, 准确率: {accuracy:.4f}')

accuracy = correct / len(Y_val_tensor)
print(f'验证准确率: {accuracy:.4f}')

plt.plot(train_losses, label='训练损失')
plt.plot(val_losses, label='验证损失')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
