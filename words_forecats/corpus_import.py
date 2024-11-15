from d2l import torch as d2l
import re
from nltk.tokenize import word_tokenize
from collections import Counter
import random
import torch
import numpy as np


def load_data_to_vocab():
    with open(d2l.download('time_machine'), 'r') as data:
        lines = data.readlines()

    tokens = word_tokenize(' '.join(lines))
    tokens = [token.lower() for token in tokens]
    tokens = [token for token in tokens if token.isalpha()]

    vocab = {"<UNK>": 0} 
    vocab.update({word: idx for idx, word in enumerate(set(tokens), 1)})
    return vocab


def sentence_to_indices(sentence, vocab):
    tokens = word_tokenize(sentence)
    indices = [vocab.get(word, vocab["<UNK>"]) for word in tokens]
    return indices

def seq_data_iter_random(words, batch_size, num_steps, vocab):

    num_examples = (len(words) - 1) // num_steps
    initial_indices = list(range(0, num_examples * num_steps, num_steps))
    random.shuffle(initial_indices)

    while True:
        for i in range(0, len(initial_indices), batch_size):
            initial_indices_per_batch = initial_indices[i:i + batch_size]

            X, Y = [], []
            for idx in initial_indices_per_batch:
                x_sequence = words[idx:idx + num_steps]
                y_sequence = words[idx + 1:idx + num_steps + 1]

                if 'cannot' in x_sequence or 'cannot' in y_sequence:
                    continue

                if len(x_sequence) == num_steps and len(y_sequence) == num_steps:
                    X.append(sentence_to_indices(' '.join(x_sequence), vocab))
                    Y.append(sentence_to_indices(' '.join(y_sequence), vocab))

            if len(X) == batch_size and len(Y) == batch_size:
                yield (torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long))

def corpus_ip(batch_size=2, num_steps=4):
    d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                    '090b5e7e70c295757f55df93cb0a180b9691891a')

    with open(d2l.download('time_machine'), 'r') as data:
        data_lines = data.readlines()
    data_lines = [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in data_lines]

    words = ' '.join(data_lines).split()
    vocab = load_data_to_vocab()
    iterator = seq_data_iter_random(words, batch_size, num_steps, vocab)

    return iterator, vocab

iterator, vocab = corpus_ip(batch_size=2, num_steps=4)

