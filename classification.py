# classification (20v20)

import json
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from utilities import get_labels, load_cls_tensor


def prompt_numbers_to_temp_data(prompts, cap, temp):
    split_tensors = [torch.chunk(load_cls_tensor(prompt, cap), 11, dim=0) for prompt in prompts]
    temp_tensors = [torch.cat((split_tensor[temp], split_tensor[10]), dim=0) for split_tensor in split_tensors]

    # stack tensors
    stack = torch.empty((0, 768))
    for tensor in temp_tensors:
        stack = torch.cat((stack, tensor), dim=0)
        
    x = stack.detach.numpy()
    y = np.tile(get_labels(), len(prompts))
    return x, y


def classify_temp(classifier, temp):
    # split into train and test
    random_numbers = list(range(10))
    random.shuffle(random_numbers)
    train_prompts, test_prompts = random_numbers[:8], random_numbers[8:]
    x_train, y_train = prompt_numbers_to_temp_data(train_prompts, 190, temp)
    x_test, y_test = prompt_numbers_to_temp_data(test_prompts, 190, temp)

    # get predications
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)

    return accuracy_score(y_test, y_pred)


def get_accuracies(classifier):
    # compute average accuracy at each temp
    number_trials = 50
    accuracies_2d = list()
    for _ in range(number_trials):
        accuracies = list()
        for temp in range(10):
            accuracies.append(classify_temp(classifier, temp))
        accuracies_2d.append(accuracies)
    accuracies_avg = [sum(x) / len(x) for x in zip(*accuracies_2d)]

    return accuracies_2d, accuracies_avg


### driver ###
accuracies = [get_accuracies(LogisticRegression(max_iter=2500)), get_accuracies(RandomForestClassifier(n_estimators=100))]
temp_vals = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

_, axs = plt.subplots(2, 1, figsize=(10, 15))
for i, ax in enumerate(axs):
    # plot and save averages
    classifier_string = "Logistic Regression" if i == 0 else "Random Forest"
    ax.plot(temp_vals, accuracies[i][1], linewidth=3)
    ax.set_xticks(temp_vals)
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy by Temperature: {classifier_string}")
    ax.set_ylim(0.8, 1)
    with open("classification/average_accuracies", 'a') as file:
        json.dump(accuracies[i][1], file)
    
    # plot individual samples
    for accuracies_1d in accuracies[i][0]:
        ax.plot(temp_vals, accuracies_1d, color="0.5", linestyle=":")

plt.savefig(f"classification/accuracies.png")
