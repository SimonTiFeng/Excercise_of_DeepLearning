import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import time
from sklearn.metrics import precision_score, recall_score, f1_score


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

class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, block1_layers_num, block2_layers_num, output_dim):
        super(Net, self).__init__()

        block1_layers = []
        block1_layers.append(nn.Linear(input_dim, hidden_dim))
        block1_layers.append(nn.ReLU())
        for _ in range(block1_layers_num):
            block1_layers.append(nn.Linear(hidden_dim, hidden_dim))
            block1_layers.append(nn.ReLU())
        block1_layers.append(nn.Linear(hidden_dim, output_dim))

        block2_layers = []
        block2_layers.append(nn.Linear(input_dim, hidden_dim))
        block2_layers.append(nn.ReLU())
        for _ in range(block2_layers_num):
            block2_layers.append(nn.Linear(hidden_dim, hidden_dim))
            block2_layers.append(nn.ReLU())
        block2_layers.append(nn.Linear(hidden_dim, output_dim))

        self.block1 = nn.Sequential(*block1_layers)
        self.block2 = nn.Sequential(*block2_layers)

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x)
        output = torch.cat((x1, x2), dim=1)
        return output

train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=2000, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2000, shuffle=True)

epochs = 100
num_classes = len(y.unique())
input_dim = X_train_tensor.shape[1]
hidden_dim = 100
block1_layers_num = 2
block2_layers_num = 4
output_dim = 2 * num_classes

criterion = nn.CrossEntropyLoss()
input_size = X_train_tensor.shape[1]
num_classes = len(y.unique())
model = Net(input_dim, hidden_dim, block1_layers_num, block2_layers_num, output_dim).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

def train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, epochs, device):
    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += Y_batch.size(0)
            correct += (predicted == Y_batch).sum().item()

        train_accuracy = 100 * correct / total
        train_losses.append(train_loss / len(train_loader))

        model.eval()
        val_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, Y_batch)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += Y_batch.size(0)
                correct += (predicted == Y_batch).sum().item()

        val_accuracy = 100 * correct / total
        val_losses.append(val_loss / len(val_loader))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, "
              f"Train Acc: {train_accuracy:.2f}%, Val Loss: {val_losses[-1]:.4f}, "
              f"Val Acc: {val_accuracy:.2f}%")

    return train_losses, val_losses

start_time = time.time()
train_losses, val_losses = train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, epochs, device)
print(f"Training completed in {time.time() - start_time:.2f} seconds")

plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss Curves')
plt.show()

model.load_state_dict(torch.load('best_model.pth'))
model.eval()
Y_true, Y_pred = [], []

with torch.no_grad():
    for X_batch, Y_batch in val_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        Y_true.extend(Y_batch.cpu().numpy())
        Y_pred.extend(predicted.cpu().numpy())

precision = precision_score(Y_true, Y_pred, average='weighted')
recall = recall_score(Y_true, Y_pred, average='weighted')
f1 = f1_score(Y_true, Y_pred, average='weighted')

print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")