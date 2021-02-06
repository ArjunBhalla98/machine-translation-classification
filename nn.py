from numpy.core.fromnumeric import argmax
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from preprocess import output_train_ints, N_TOKENS, get_label_from_output
from eval import calc_precision, calc_recall, calc_f1

SEQ_LENGTH = 50
N_EPOCHS = 50000

# Credit: pytorch docs (https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.hidden_layer = None
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        self.hidden_layer = hidden
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden


n_hidden = 50
n_categories = 2
rnn = RNN(1, n_hidden, n_categories)
lr = 0.01
criterion = nn.NLLLoss()
samples, scores, labels = output_train_ints()

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


def get_f1(test_samples, test_scores, test_labels):
    true_h = 0
    classified_as_h = 0
    total_h = 0

    true_m = 0
    classified_as_m = 0
    total_m = 0

    for i in range(len(test_samples)):
        sample = test_samples[i]
        score = test_scores[i]
        label = test_labels[i]

        with torch.no_grad():
            model_prediction = None
            hidden = rnn.hidden_layer
            for i in range(len(sample)):
                model_prediction, hidden = rnn(
                    torch.reshape(torch.tensor(sample[i]), (1, 1)), hidden
                )

            model_prediction = torch.argmax(model_prediction[0]).item()

            if label == 0:
                total_h += 1

                if model_prediction == 0:
                    true_h += 1
                    classified_as_h += 1
                else:
                    classified_as_m += 1
            else:
                total_m += 1

                if model_prediction == 1:
                    true_m += 1
                    classified_as_m += 1
                else:
                    classified_as_h += 1

    h_precision = calc_precision(true_h, classified_as_h)
    h_recall = calc_recall(true_h, total_h)
    m_precision = calc_precision(true_m, classified_as_m)
    m_recall = calc_recall(true_m, total_m)

    h_f1 = calc_f1(h_precision, h_recall)
    m_f1 = calc_f1(m_precision, m_recall)

    return h_f1, m_f1


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

test_samples, test_scores, test_labels = output_train_ints(False)

h_f1, m_f1 = get_f1(test_samples, test_scores, test_labels)

print(f"H F1: {h_f1}, M F1: {m_f1}")