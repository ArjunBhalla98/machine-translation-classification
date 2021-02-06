import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from preprocess import output_train_ints, N_TOKENS, get_label_from_output
from eval import *

VAL_SET_PCT = 0.15
EMBEDDING_DIM = 300
N_HIDDEN = 100
N_EPOCHS = 100
RATE = 0.001

train_accuracy_list = []
val_accuracy_list = []

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
        self.name = "LSTM"
        # Minor hack to help init embedding weights
        torch.manual_seed(15177802706603365271)
        self.embedding = nn.Embedding(N_TOKENS, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.out_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, score, train=True, hidden=None):
        embeddings = self.embedding(x) * score
        lstm_out, _ = self.lstm(embeddings.view(len(x), 1, -1))
        lstm_linear_pass = self.out_layer(lstm_out.view(len(x), -1))
        if train:
            lstm_linear_pass = self.dropout(lstm_linear_pass)
        final_out = torch.mean(lstm_linear_pass, 0)
        return F.log_softmax(final_out).view(1, 2)


lstm = LSTM(EMBEDDING_DIM, N_HIDDEN, 2)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(lstm.parameters(), lr=RATE)

# Do this on independent samples / words because of varying seq lengths
def train(sample, score, label):
    lstm.zero_grad()

    output = lstm(sample, score)
    loss = loss_function(output, label)
    loss.backward()
    optimizer.step()

    return output, loss.item()


loss_counter = 0
print_interval = 1

# Train
for epoch in range(N_EPOCHS):
    for n in range(len(samples)):
        i = random.randint(0, len(samples) - 1)
        sample = samples[i]
        score = scores[i]
        label = labels[i]

        output, loss = train(sample, score, label)
        loss_counter += loss

        model_prediction = torch.argmax(output[0]).item()

        if epoch % print_interval == 0 and n == len(samples) - 1:
            guess = get_label_from_output(output)
            actual = "H" if label == 0 else "M"
            print(f"Model Predicted: {guess}, actual is: {actual}.")
            print(f"Epoch {epoch} average loss: {loss_counter / len(samples)}\n")

            train_acc = get_accuracy(lstm, samples, scores, labels, loss_function)
            train_accuracy_list.append(train_acc)

            val_acc = get_accuracy(
                lstm, val_samples, val_scores, val_labels, loss_function
            )
            val_accuracy_list.append(val_acc)
            loss_counter = 0

test_samples, test_scores, test_labels = output_train_ints(False)
h_f1, m_f1 = get_f1(lstm, test_samples, test_scores, test_labels)


print(f"H F1: {h_f1}, M F1: {m_f1}\n")
print(f"Average F1: {(h_f1 + m_f1) / 2}\n")

#### PLOT ####
x_axis = [i for i in range(N_EPOCHS)]

plt.plot(x_axis, train_accuracy_list, color="r", marker=".", label="Training")
plt.plot(x_axis, val_accuracy_list, color="b", marker=".", label="Validation")
plt.title("Training and Validation Accuracy for LSTM")
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("./graphs/test.png")
plt.clf()