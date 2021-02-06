import torch

# For calculating F1 score:
# Human F1: Precision = true H / everything classified as H,
# Recall = true H / total H seen
def calc_precision(true_pos, all_pos):
    return true_pos / all_pos


def calc_recall(true_pos, total_relevant):
    return true_pos / total_relevant


def calc_f1(precision, recall):
    epsilon = 0 if precision > 0 or recall > 0 else 1
    return 2 * (precision * recall) / (precision + recall + epsilon)


def get_accuracy(model, samples, scores, labels, loss_function):
    n_correct = 0
    cumulative_loss = 0

    for i in range(len(samples)):
        sample = samples[i]
        score = scores[i]
        label = labels[i]

        with torch.no_grad():
            output = model(sample, score, False)
            if model.name == "LogReg":
                model_prediction = 0 if output <= 0.5 else 1
                loss = loss_function(output, label.double())
                cumulative_loss += loss.item()
            else:
                model_prediction = torch.argmax(output[0]).item()
                loss = loss_function(output, label)
                cumulative_loss += loss.item()

            if label == model_prediction:
                n_correct += 1

    accuracy = n_correct / len(samples)
    return accuracy


def get_f1(model, test_samples, test_scores, test_labels):
    true_h = 0
    classified_as_h = 1
    total_h = 1

    true_m = 0
    classified_as_m = 1
    total_m = 1

    for i in range(len(test_samples)):
        sample = test_samples[i]
        score = test_scores[i]
        label = test_labels[i]

        with torch.no_grad():
            output = model(sample, score, False)
            if model.name == "LogReg":
                model_prediction = 0 if output <= 0.5 else 1
            else:
                model_prediction = torch.argmax(output[0]).item()

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
