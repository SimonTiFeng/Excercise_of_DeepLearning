import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import time
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score, precision_recall_fscore_support
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize
import struct
from torchvision import datasets, transforms
from d2l import torch as d2l

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_images = train_dataset.data.numpy()
train_labels = train_dataset.targets.numpy()
test_images = test_dataset.data.numpy()
test_labels = test_dataset.targets.numpy()

train_images = torch.from_numpy(train_images).float()
train_labels = torch.from_numpy(train_labels).long()
test_images = torch.from_numpy(test_images).float()
test_labels = torch.from_numpy(test_labels).long()

device = torch.device('cuda')

train_images = train_images.to(device)
train_labels = train_labels.to(device)
test_images = test_images.to(device)
test_labels = test_labels.to(device)

net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
    nn.AvgPool2d(kernel_size=2, stride=2),  
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.ReLU(),  
    nn.Linear(120, 84), nn.ReLU(),
    nn.Linear(84, 10)
)

net.to(device)

train_dataset = TensorDataset(train_images, train_labels)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = TensorDataset(test_images, test_labels)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

num_epochs = 10  

for epoch in range(num_epochs):
    net.train()  
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.unsqueeze(1).to(device) 
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = net(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

def evaluate_model(model, loader):
    model.eval() 
    correct, total = 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():  
        for images, labels in loader:
            images = images.unsqueeze(1).to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

evaluate_model(net, test_loader)

loss_values = []

for epoch in range(num_epochs):
    net.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.unsqueeze(1).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    loss_values.append(epoch_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

plt.plot(range(1, num_epochs+1), loss_values, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.show()
