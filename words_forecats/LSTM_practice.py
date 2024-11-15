import corpus_import
from model import LSTMModel
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

iterator, vocab = corpus_import.corpus_ip(batch_size=2, num_steps=4)

vocab_size = len(vocab)
embedding_dim = 100
hidden_dim = 256

model = LSTMModel(vocab_size, embedding_dim, hidden_dim).to(torch.device('cuda'))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10000

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

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

def predict_next_word(model, input_tensor,vocab):
    
    reverse_vocab = {v: k for k, v in vocab.items()}

    input_tensor = corpus_import.sentence_to_indices(input_tensor, vocab)
    input_tensor = torch.tensor(input_tensor).unsqueeze(0) 

    if torch.cuda.is_available():
        input_tensor = input_tensor.to(torch.device('cuda'))
        model = model.to(torch.device('cuda')) 
    
    model.eval()  
    with torch.no_grad():
        output = model(input_tensor) 
        outout = output.squeeze(0).to('cpu').numpy()
        output = np.argmax(outout, axis=1)

    output_word = [reverse_vocab[i] for i in output]
    return output_word

input_sentence = ['time', 'of', 'the', 'clock']
input_sentence_str = ' '.join(input_sentence)  
print(predict_next_word(model, input_sentence_str, vocab))
