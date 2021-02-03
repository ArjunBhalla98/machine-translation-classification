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
N_EPOCHS = 18000
RATE = 0.01

samples, labels = output_train_ints()


class LSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, out_dim):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(N_TOKENS, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        embeddings = self.embedding(x)
        lstm_out, _ = self.lstm(embeddings.view(len(x), 1, -1))
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


# For calculating F1 score:
# Human F1: Precision = true H / everything classified as H,
# Recall = true H / total H seen
def calc_precision(true_pos, all_pos):
    return true_pos / all_pos


def calc_recall(true_pos, total_relevant):
    return true_pos / total_relevant


def calc_f1(precision, recall):
    return 2 * (precision * recall) / (precision + recall)


loss_counter = 0
print_interval = 1000
n_correct = 0


# Train
for epoch in range(N_EPOCHS):
    rand_idx = random.randint(0, len(samples) - 1)
    sample = samples[rand_idx]
    label = labels[rand_idx]

    output, loss = train(sample, label)
    loss_counter += loss

    model_prediction = torch.argmax(output[0]).item()
    n_correct += 1 if model_prediction == label else 0

    if epoch % print_interval == 0:
        guess = get_label_from_output(output)
        actual = "H" if label == 0 else "M"
        print(f"Model Predicted: {guess}, actual is: {actual}.")
        print(f"Epoch average loss: {loss_counter / print_interval}")
        loss_counter = 0

print(f"Accuracy after training: {n_correct / N_EPOCHS}")
# For F1:
true_h = 0
classified_as_h = 0
total_h = 0

true_m = 0
classified_as_m = 0
total_m = 0

test_samples, test_labels = output_train_ints(False)

for i in range(len(test_samples)):
    sample = test_samples[i]
    label = test_labels[i]

    with torch.no_grad():
        model_prediction = torch.argmax(lstm(sample)[0]).item()

        # F1 stuff:
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

print("H precision, M precision:")
print(h_precision, m_precision)

print(f"H F1: {h_f1}, M F1: {m_f1}")
