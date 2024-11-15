import corpus_import
import torch.nn as nn

# Load data
iterator, vocab = corpus_import.corpus_ip(batch_size=2, num_steps=4)

# Define model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)  
        lstm_out, _ = self.lstm(x)  
        out = self.fc(lstm_out) 
        return out
