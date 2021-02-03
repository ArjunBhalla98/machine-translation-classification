from collections import defaultdict
import string

TRAIN_FILE = "./train.txt"
TEST_FILE = "./test.txt"
N_TOKENS = 5000
# index 0: source, index 1: reference, index 2: candidate, index 3: score, index 4: label


def split(file):
    with open(file, "r") as f:
        text = f.read()
        split_text = text.split("\n\n")
        split_text = list(
            map(
                lambda x: [
                    x.split("\n")[2].translate(
                        str.maketrans("", "", string.punctuation)
                    ),
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
        label = sample[1]
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

        result.append([in_text_ints, label])

    return result


data = split(TRAIN_FILE)
print(translate_to_integer(data)[145])

