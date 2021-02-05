import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from preprocess import output_train_ints, N_TOKENS, get_label_from_output

SEQ_LENGTH = 100
VAL_SET_PCT = 0.15
EMBEDDING_DIM = 300
N_HIDDEN = 100
N_EPOCHS = 20000
RATE = 0.1

# Ideas: add dropout? Remove / add more layers from network (doesn't seem to do much)

samples, scores, labels = output_train_ints()
val_set_cutoff = int(VAL_SET_PCT * len(samples))
val_samples = samples[:val_set_cutoff]
val_scores = scores[:val_set_cutoff]
val_labels = labels[:val_set_cutoff]

samples = samples[val_set_cutoff:]
scores = scores[val_set_cutoff:]
labels = labels[val_set_cutoff:]


class Model(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, out_dim):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(N_TOKENS, embedding_dim)
        self.linear = nn.Linear(embedding_dim, out_dim)
        self.dropout = nn.Dropout(p=0.3)
        # self.out_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, score, train=True):
        embeddings = self.embedding(x) * score
        out = self.linear(torch.mean(embeddings, 0))
        if train:
            out = self.dropout(out)
        return F.sigmoid(out).double()
        # model_out, _ = self.model(embeddings.view(len(x), 1, -1))
        # model_linear_pass = self.out_layer(model_out.view(len(x), -1))
        # if train:
        #     model_linear_pass = self.dropout(model_linear_pass)
        # model_sentence_average = torch.mean(model_linear_pass, 0)
        # return F.log_softmax(model_sentence_average).view(1, 2)


model = Model(EMBEDDING_DIM, N_HIDDEN, 1)
loss_function = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=RATE)
# Do this on independent samples / words because of varying seq lengths


def train(sample, score, label):
    hidden = torch.rand((1, N_HIDDEN))
    model.hidden_dim = hidden

    model.zero_grad()

    output = model(sample, score)
    loss = loss_function(output, label.double())
    loss.backward()
    optimizer.step()

    return output, loss.item()


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

    model_prediction = 0 if output <= 0.5 else 1
    # n_correct += 1 if model_prediction == label else 0

    if epoch % print_interval == 0:
        guess = "H" if output <= 0.5 else "M"
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
        model_prediction = 0 if model(sample, score, False) <= 0.5 else 1

        if label == model_prediction:
            n_correct += 1

print(f"Training Accuracy: {n_correct / len(samples)}\n")

n_correct = 0

for i in range(len(val_samples)):
    sample = val_samples[i]
    score = val_scores[i]
    label = val_labels[i]

    with torch.no_grad():
        model_prediction = 0 if model(sample, score, False) <= 0.5 else 1

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
        model_prediction = 0 if model(sample, score, False) <= 0.5 else 1

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
