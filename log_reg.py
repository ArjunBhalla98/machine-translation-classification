import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from preprocess import output_train_ints, N_TOKENS, get_label_from_output
from eval import *

VAL_SET_PCT = 0.15
EMBEDDING_DIM = 300
N_HIDDEN = 100
N_EPOCHS = 70000
RATE = 0.001

samples, scores, labels = output_train_ints()
val_set_cutoff = int(VAL_SET_PCT * len(samples))
val_samples = samples[:val_set_cutoff]
val_scores = scores[:val_set_cutoff]
val_labels = labels[:val_set_cutoff]

samples = samples[val_set_cutoff:]
scores = scores[val_set_cutoff:]
labels = labels[val_set_cutoff:]


class LogisticRegression(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, out_dim):
        super(LogisticRegression, self).__init__()
        self.name = "LogReg"
        self.embedding = nn.Embedding(N_TOKENS, embedding_dim)
        self.linear = nn.Linear(embedding_dim, out_dim)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x, score, train=True):
        embeddings = self.embedding(x) * score
        out = self.linear(torch.mean(embeddings, 0))
        if train:
            out = self.dropout(out)
        return F.sigmoid(out).double()


log_reg = LogisticRegression(EMBEDDING_DIM, N_HIDDEN, 1)
loss_function = nn.BCELoss()
optimizer = optim.SGD(log_reg.parameters(), lr=RATE)

# Do this on independent samples / words because of varying seq lengths
def train(sample, score, label):
    hidden = torch.rand((1, N_HIDDEN))
    log_reg.hidden_dim = hidden

    log_reg.zero_grad()

    output = log_reg(sample, score)
    loss = loss_function(output.double(), label.double())
    loss.backward()
    optimizer.step()

    return output, loss.item()


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

    log_reg_prediction = 0 if output <= 0.5 else 1

    if epoch % print_interval == 0:
        guess = "H" if output <= 0.5 else "M"
        actual = "H" if label == 0 else "M"
        print(f"Model Predicted: {guess}, actual is: {actual}.")
        print(f"Epoch {epoch} average loss: {loss_counter / print_interval}\n")
        loss_counter = 0

        train_acc = get_accuracy(log_reg, samples, scores, labels, loss_function)
        print(f"Training Accuracy: {train_acc}\n")
        val_acc = get_accuracy(
            log_reg, val_samples, val_scores, val_labels, loss_function
        )
        print(f"Validation Accuracy:  {val_acc}")

test_samples, test_scores, test_labels = output_train_ints(False)

h_f1, m_f1 = get_f1(log_reg, test_samples, test_scores, test_labels)

print(f"H F1: {h_f1}, M F1: {m_f1}\n")
