import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from preprocess import output_train_ints, N_TOKENS, get_label_from_output

SEQ_LENGTH = 50
EMBEDDING_DIM = 100
N_HIDDEN = 50
N_EPOCHS = 25000
RATE = 0.01

samples, labels = output_train_ints(SEQ_LENGTH)


class LSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, out_dim):
        super(LSTM, self).__init__()
        # Should handle one word at a time
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(N_TOKENS, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        embeddings = self.embedding(x)
        lstm_out, hidden = self.lstm(embeddings.view(len(x), 1, -1))
        lstm_linear_pass = self.out_layer(lstm_out.view(len(x), -1))
        lstm_sentence_average = torch.mean(lstm_linear_pass, 0)
        return F.log_softmax(lstm_sentence_average).view(1, 2)


lstm = LSTM(EMBEDDING_DIM, N_HIDDEN, 2)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(lstm.parameters(), lr=RATE)
# Do this on independent samples / words because of varying seq lengths


def train(sample, label):
    hidden = torch.rand((1, N_HIDDEN))
    lstm.hidden_dim = hidden

    lstm.zero_grad()
    output = None

    # for word_idx in sample:
    # output = lstm(torch.tensor(word_idx))
    output = lstm(sample)
    loss = loss_function(output, label)
    loss.backward()
    optimizer.step()

    return output, loss.item()


loss_counter = 0
print_interval = 1000
n_correct = 0

for epoch in range(N_EPOCHS):
    rand_idx = random.randint(0, len(samples) - 1)
    sample = samples[rand_idx]
    label = labels[rand_idx]

    output, loss = train(sample, label)
    loss_counter += loss

    n_correct += 1 if torch.argmax(output[0]).item() == label else 0

    if epoch % print_interval == 0:
        guess = get_label_from_output(output)
        actual = "H" if label == 0 else "M"
        print(f"Model Predicted: {guess}, actual is: {actual}.")
        print(f"Epoch average loss: {loss_counter / print_interval}")
        loss_counter = 0

print(f"Accuracy after training: {n_correct / N_EPOCHS}")
