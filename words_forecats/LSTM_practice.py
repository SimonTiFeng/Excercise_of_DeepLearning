import corpus_import
from model import LSTMModel
import torch
import torch.nn as nn
import torch.optim as optim

iterator, vocab = corpus_import.corpus_ip(batch_size=2, num_steps=5)

vocab_size = len(vocab)
embedding_dim = 100
hidden_dim = 256

model = LSTMModel(vocab_size, embedding_dim, hidden_dim).to(torch.device('cuda'))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    model.train() 
    
    inputs, targets = next(iterator)

    inputs, targets = inputs.to(torch.device('cuda')), targets.to(torch.device('cuda'))

    optimizer.zero_grad()  

    outputs = model(inputs) 
    outputs = outputs.view(-1, vocab_size)  
    targets = targets.view(-1)  

    loss = criterion(outputs, targets)  

    loss.backward()  
    optimizer.step()  

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

def predict_next_word(model, input_tensor):
    model.eval()  
    with torch.no_grad():
        output = model(input_tensor) 
        output = output[:, -1, :]  
        predicted_index = torch.argmax(output, dim=1)  
    return predicted_index.item()
