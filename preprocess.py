from collections import defaultdict
import string
import numpy as np
import torch

TRAIN_FILE = "./train.txt"
TEST_FILE = "./test.txt"
N_TOKENS = 7000

# index 0: source, index 1: reference, index 2: candidate, index 3: score, index 4: label


def split(file):
    with open(file, "r") as f:
        text = f.read()
        split_text = text.split("\n\n")
        split_text = list(
            map(
                lambda x: [
                    x.split("\n")[0]
                    + " "
                    + x.split("\n")[2].translate(
                        str.maketrans("", "", string.punctuation)
                    ),
                    float(x.split("\n")[3]),
                    x.split("\n")[4],
                ],
                split_text,
            )
        )
        return split_text


def translate_to_integer(data):
    word_to_idx = defaultdict(int)
    idx = 0
    result = []

    for sample in data:
        in_text = sample[0]
        score = sample[1]
        label = sample[2]
        in_text_ints = []

        for word in in_text.split():
            if word not in word_to_idx and idx <= N_TOKENS:
                # this way unknowns all automatically get assigned to '0'
                idx += 1
                word_to_idx[word] = idx

            if (
                word_to_idx[word] != 0
            ):  # temporary measure while I figure out what to do with unknowns
                in_text_ints.append(word_to_idx[word])

        result.append([torch.tensor(in_text_ints), score, label])
    return result


def pad(input, total_length):
    for sample_idx in range(len(input)):
        text_in = input[sample_idx][0]
        while len(text_in) < total_length:
            text_in.append(0)

        input[sample_idx][0] = torch.tensor(text_in)

    return input


def split_data_labels(data):
    data = np.array(data)
    return data[:, 0], data[:, 1], data[:, 2]


def give_numeric_labels(labels):
    for i in range(len(labels)):
        if labels[i] == "H":
            labels[i] = torch.tensor([0])
        else:
            labels[i] = torch.tensor([1])


def output_train_ints(train_file=True):
    if train_file:
        data = split(TRAIN_FILE)
    else:
        data = split(TEST_FILE)

    # padded_ints = pad(translate_to_integer(data), seq_length)
    # no need to pad anymore
    padded_ints = translate_to_integer(data)
    sample, score, labels = split_data_labels(padded_ints)
    give_numeric_labels(labels)
    return sample, score, labels


def get_label_from_output(output):
    return "H" if torch.argmax(output[0]).item() == 0 else "M"


def output_train_words(train_file=True):
    if train_file:
        data = split(TRAIN_FILE)
    else:
        data = split(TEST_FILE)
    samples, scores, labels = split_data_labels(data)
    give_numeric_labels(labels)
    return samples, scores, labels


# data = split(TRAIN_FILE)
# final = split_data_labels(pad(translate_to_integer(data), 50))
# give_numeric_labels(final[1])
# print(final[1])

# Checked balanced dataset
# data = split(TEST_FILE)
# samples, labels = split_data_labels(data)
# n_h = 0
# n_m = 0

# for label in labels:
#     if label == "H":
#         n_h += 1
#     else:
#         n_m += 1

# print(n_h, n_m)
