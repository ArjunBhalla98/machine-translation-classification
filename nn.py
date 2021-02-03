from numpy.core.fromnumeric import argmax
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from preprocess import output_train_ints, N_TOKENS

SEQ_LENGTH = 50
N_EPOCHS = 50000

# Credit: pytorch docs (https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        # self.embedding = nn.Embedding(N_TOKENS, 50)

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden


n_hidden = 50
n_categories = 2
rnn = RNN(1, n_hidden, n_categories)
lr = 0.0001
criterion = nn.NLLLoss()
samples, labels = output_train_ints(SEQ_LENGTH)

# H = 0, M = 1
def train(sample, label):
    hidden = torch.zeros(1, n_hidden)

    rnn.zero_grad()
    output = None

    for i in range(len(sample)):
        output, hidden = rnn(torch.reshape(torch.tensor(sample[i]), (1, 1)), hidden)

    loss = criterion(output, label)
    loss.backward()

    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-lr)

    return output, loss.item()


def get_label_from_output(output):
    return "H" if torch.argmax(output[0]).item() == 0 else "M"


loss_counter = 0
print_interval = 1000
n_correct = 0
m_predicted = 0

for i in range(N_EPOCHS):
    rand_idx = random.randint(0, len(samples) - 1)
    sample = samples[rand_idx]
    label = labels[rand_idx]

    output, loss = train(sample, label)
    loss_counter += loss

    n_correct += 1 if torch.argmax(output[0]).item() == label else 0
    m_predicted += 1 if torch.argmax(output[0]).item() == 1 else 0
    if i % print_interval == 0:
        guess = get_label_from_output(output)
        actual = "H" if label == 0 else "M"
        print(f"Model Predicted: {guess}, actual is: {actual}.")
        print(f"Epoch average loss: {loss_counter / print_interval}")
        loss_counter = 0

print(f"Accuracy after training: {n_correct / N_EPOCHS}")
print(m_predicted)
# print(train(samples[0], labels[0]))
# print(len(samples) == len(labels))