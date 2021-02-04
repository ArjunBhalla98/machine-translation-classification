import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from preprocess import output_train_ints, N_TOKENS, get_label_from_output

SEQ_LENGTH = 100
VAL_SET_PCT = 0.25
EMBEDDING_DIM = 300
N_HIDDEN = 100
N_EPOCHS = 15000
RATE = 0.01

samples, scores, labels = output_train_ints()
val_set_cutoff = int(VAL_SET_PCT * len(samples))
val_samples = samples[:val_set_cutoff]
val_scores = scores[:val_set_cutoff]
val_labels = labels[:val_set_cutoff]

samples = samples[val_set_cutoff:]
scores = scores[val_set_cutoff:]
labels = labels[val_set_cutoff:]


class LSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, out_dim):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(N_TOKENS, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, score):
        embeddings = self.embedding(x) * score
        lstm_out, _ = self.lstm(embeddings.view(len(x), 1, -1))
        lstm_linear_pass = self.out_layer(lstm_out.view(len(x), -1))
        lstm_sentence_average = torch.mean(lstm_linear_pass, 0)
        return F.log_softmax(lstm_sentence_average).view(1, 2)


lstm = LSTM(EMBEDDING_DIM, N_HIDDEN, 2)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(lstm.parameters(), lr=RATE)
# Do this on independent samples / words because of varying seq lengths


def train(sample, score, label):
    hidden = torch.rand((1, N_HIDDEN))
    lstm.hidden_dim = hidden

    lstm.zero_grad()

    output = lstm(sample, score)
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

# Train
for epoch in range(N_EPOCHS):
    rand_idx = random.randint(0, len(samples) - 1)
    sample = samples[rand_idx]
    score = scores[rand_idx]
    label = labels[rand_idx]

    output, loss = train(sample, score, label)
    loss_counter += loss

    model_prediction = torch.argmax(output[0]).item()
    # n_correct += 1 if model_prediction == label else 0

    if epoch % print_interval == 0:
        guess = get_label_from_output(output)
        actual = "H" if label == 0 else "M"
        print(f"Model Predicted: {guess}, actual is: {actual}.")
        print(f"Epoch {epoch} average loss: {loss_counter / print_interval}\n")
        loss_counter = 0

n_correct = 0

for i in range(len(samples)):
    sample = samples[i]
    score = scores[i]
    label = labels[i]

    with torch.no_grad():
        model_prediction = torch.argmax(lstm(sample, score)[0]).item()

        if label == model_prediction:
            n_correct += 1

print(f"Training Accuracy: {n_correct / len(samples)}\n")

n_correct = 0

for i in range(len(val_samples)):
    sample = val_samples[i]
    score = val_scores[i]
    label = val_labels[i]

    with torch.no_grad():
        model_prediction = torch.argmax(lstm(sample, score)[0]).item()

        if label == model_prediction:
            n_correct += 1

print(f"Validation Accuracy: {n_correct / len(val_samples)}\n")

# For F1:
true_h = 0
classified_as_h = 0
total_h = 0

true_m = 0
classified_as_m = 0
total_m = 0

correct_test = 0

test_samples, test_scores, test_labels = output_train_ints(False)

for i in range(len(test_samples)):
    sample = test_samples[i]
    score = test_scores[i]
    label = test_labels[i]

    with torch.no_grad():
        model_prediction = torch.argmax(lstm(sample, score)[0]).item()

        # F1 stuff:
        if label == 0:
            total_h += 1

            if model_prediction == 0:
                true_h += 1
                classified_as_h += 1
                correct_test += 1
            else:
                classified_as_m += 1
        else:
            total_m += 1

            if model_prediction == 1:
                correct_test += 1
                true_m += 1
                classified_as_m += 1
            else:
                classified_as_h += 1

EPSILON = 0.0000001
h_precision = calc_precision(true_h, classified_as_h) + EPSILON
h_recall = calc_recall(true_h, total_h) + EPSILON
m_precision = calc_precision(true_m, classified_as_m) + EPSILON
m_recall = calc_recall(true_m, total_m) + EPSILON

h_f1 = calc_f1(h_precision, h_recall)
m_f1 = calc_f1(m_precision, m_recall)

print("H precision, M precision:")
print(h_precision, m_precision)
print()

print(f"H F1: {h_f1}, M F1: {m_f1}\n")
print(f"Test Accuracy: {correct_test / len(test_samples)}")
