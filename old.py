# old code

import json
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, adjusted_rand_score,
                             classification_report)
from sklearn.metrics.cluster import adjusted_rand_score



### original clustering (200 v 20)###

def get_true_labels(binary):
    if binary:
        return np.concatenate((np.zeros(190, dtype=int), np.ones(20, dtype=int)))
    else:
        return np.repeat(np.arange(11), 20)


def cluster_cls_tensor(clustering, prompt_number, cap, binary):
    number_clusters = 2 if binary else 11
    true_labels = get_true_labels(binary)

    cls_tensor = load_cls_tensor(prompt_number, cap)
    cls_array = cls_tensor.detach().numpy()

    clustering_algo = None
    if clustering == "kmeans":
        clustering_algo = KMeans(n_clusters=number_clusters, n_init="auto")
    elif clustering == "hac":
        clustering_algo = AgglomerativeClustering(n_clusters=number_clusters)
    elif clustering == "spectral":
        clustering_algo = SpectralClustering(
            n_clusters=number_clusters, assign_labels="discretize"
        )

    labels = clustering_algo.fit_predict(cls_array)
    rand_index = adjusted_rand_score(true_labels, labels)

    classification_type = "binary" if binary else "temp"
    return f"Rand Index for prompt {prompt_number} ({classification_type}): \t {rand_index}"


def save_clustering(clustering):
    outputs = list()
    for prompt in range(10):
        outputs.append(cluster_cls_tensor(clustering, prompt, 190, True))
        outputs.append(cluster_cls_tensor(clustering, prompt, 190, False))
    with open(f"clustering/{clustering}.txt", "w") as f:
        for output in outputs:
            f.write(output + "\n")


"""
for clustering in ["kmeans", "hac", "spectral"]:
    save_clustering(clustering) 
"""


### original classification (200 v 20) ###


def stack_tensors(tensors):
    stack = torch.empty((0, 768))
    for tensor in tensors:
        stack = torch.cat((stack, tensor), dim=0)
    return stack.detach()


def prompt_numbers_to_data(prompts, binary, cap):
    train_tensors = list()
    for prompt in prompts:
        train_tensors.append(load_cls_tensor(prompt, cap))
    x = stack_tensors(train_tensors).numpy()
    y = np.tile(get_true_labels(binary), len(prompts))
    return x, y


def classify(classifier, binary, classifier_string):
    # split into train and test
    random_numbers = list(range(10))
    random.shuffle(random_numbers)
    train_prompts, test_prompts = random_numbers[:8], random_numbers[8:]
    x_train, y_train = prompt_numbers_to_data(train_prompts, binary, 190)
    x_test, y_test = prompt_numbers_to_data(test_prompts, binary, 190)

    # get predications
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)

    # save metrics
    classification_type = "binary" if binary else "multiclass"
    labels = list(range(2)) if binary else list(range(11))
    report = classification_report(y_test, y_pred, labels=labels)
    with open(
        f"classification/{classification_type}_{classifier_string}.txt", "w"
    ) as out_file:
        print(
            f"{classification_type} classification using {classifier_string}\n",
            file=out_file,
        )
        print(report, file=out_file)


"""
for classifier, classifier_string in [(LogisticRegression(max_iter=2500), "logistic_regression"), (RandomForestClassifier(n_estimators=100), "random_forest")]:
    for binary in [True, False]:
        classify(classifier, binary, classifier_string)
"""

### converting GPT to reddit format ###
import nltk
from nltk.tokenize import word_tokenize

text = "A man dressed in a tuxedo smiled as he walked in a neighborhood. He grabbed a green paper and read the address of the person on this green paper. He looked toward a house, and kicked it open.\n\nHe entered a musty smelling house. There were cobwebs everywhere on each ceiling corner. He turned to a living room, the fire was crackling tantalizingly. There was a rocking chair that was moving rhythmically. The man that was seated in this chair was just staring at the fire.\n\n``Hello, you're late.'' He said, turning his head toward the man. The standing man plucked a green paper and tossed it to the sitting man. ``A warrant for murder?'' The sitting man chuckled. He opened the paper. ``Aww, I killed your mama. How sad.'' He seemed to enjoy himself.\n\nThe standing man pulled out a weapon, pointed it toward the sitting man. ``Goodbye, old man.'' He said before firing. Blood splattered across the wall next to the sitting man. He heard something pinging across the floor. He walked around the old man and saw a pen clattering and rolling away.\n\nHe then noticed the sitting man's coat. A flash of green was barely showing itself inside the coat. He chuckled to himself, if this man had a warrant to kill someone, perhaps he could take it over. So he pulled out the paper, opened it, and read it.\n\nIt had his name on it, fear crept up on him. He looked around and noticed the door he had kicked in was wired. The clever bastard booby-trapped him. He made a wry chuckle, he knew this house was going to blow up anytime soon. So he decided to read the reason why he was going to be murdered.\n\nThe paper said: ``He married my daughter.''\n\n-008"

placeholder = ' NEWLINE_PLACEHOLDER'
modified_text = text.replace('\n', placeholder)

#sentences = sent_tokenize(text)

tokenized_text = []
#for sentence in sentences:
tokens = word_tokenize(modified_text)
tokenized_text.extend(tokens)

formatted_text = ' '.join(tokenized_text)
formatted_text = formatted_text.replace(placeholder,  ' <newline> ')

print(formatted_text)


### most common prompts ###

""" 
line_counts = Counter(prompt_lines)
most_frequent = line_counts.most_common(10)
most_frequent_lines = [line for line, _ in most_frequent]

# gather indices of 10 most frequent prompt
# result: using just the training data, there exist 10 prompts with >= 89 responses
prompts = [line.strip() for line in most_frequent_lines]
most_frequent_indices = list()
for prompt in prompts:
    indices = [index for index, line in enumerate(prompt_lines) if (line.strip() == prompt)]
    most_frequent_indices.append((prompt, indices))
    
reordered_indices = list(range(10))
reordered_indices[0] = most_frequent_indices[3]
reordered_indices[1] = most_frequent_indices[0]
reordered_indices[2] = most_frequent_indices[6]
reordered_indices[3] = most_frequent_indices[8]
reordered_indices[4] = most_frequent_indices[5]
reordered_indices[5] = most_frequent_indices[4]
reordered_indices[6] = most_frequent_indices[2]
reordered_indices[7] = most_frequent_indices[9]
reordered_indices[8] = most_frequent_indices[1]
reordered_indices[9] = most_frequent_indices[7]

for p, i in reordered_indices:
    print(p) """
    
    
    
# combine train, test, and validate

"""
for type in ["source", "target"]:
    for name in ["train", "test", "valid"]:
        with open(f'reddit_dataset/{name}.wp_{type}', 'r') as input_file:
            with open(f'reddit_dataset/combined_{type}', 'a') as output_file:
                output_file.write(input_file.read())
"""



# clean reddit responses (extra spaces, punctuation, unicode artifacts)

import re
import unicodedata
import json

""" def convert_to_ascii(text):
    return (
        "".join(
            c
            for c in unicodedata.normalize("NFD", text)
            if unicodedata.category(c) != "Mn"
        )
        .encode("ascii", "ignore")
        .decode("utf-8")
    )
 """

def clean(text):
    # text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("''", '"')
    text = text.replace("``", '"')
    text = re.sub(r"\s([.,;!?'])", r"\1", text)
    text = text.replace("<newline> ", "\n")
    text = text.replace("<newline>", "\n")
    text = text.replace(" n't", "n't")
    text = text.replace(" )", ")")
    text = text.replace("( ", "(")
    return text


### driver ###
""" with open("responses/reddit_responses.json", "r") as in_file:
    all_responses = json.load(in_file)

cleaned_all_responses = list()
for prompt in all_responses:
    cleaned_responses = list()
    for response in prompt:
        cleaned_responses.append(clean(response))
    cleaned_all_responses.append(cleaned_responses)

with open("responses/cleaned_reddit_responses.json", "w") as out_file:
    json.dump(cleaned_all_responses, out_file)

"""
 

 
"""
# get CLS tokens from responses and save as torch tensors

import json

import torch
from transformers import AutoModel, AutoTokenizer


def load_responses():
    with open("responses/gpt_responses.json", "r") as in_file:
        gpt_responses = json.load(in_file)
    with open("responses/cleaned_reddit_responses.json", "r") as in_file:
        cleaned_reddit_responses = json.load(in_file)
    return gpt_responses, cleaned_reddit_responses


def cap_paragraph(paragraph, cap):
    split = paragraph.split()
    reduced = split[:cap]
    joined = " ".join(reduced)
    return joined


def get_capped_paragraphs(responses, prompt_number, cap):
    gpt_responses = responses[0]
    reddit_responses = responses[1]
    paragraphs = list()
    for t in range(10):
        for r in range(20):
            paragraphs.append(cap_paragraph(gpt_responses[t][prompt_number][r], cap))
    for r in range(20):
        paragraphs.append(cap_paragraph(reddit_responses[prompt_number][r], cap))
    return paragraphs


def paragraphs_to_CLS(paragraphs):
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    cls_vectors = []
    for paragraph in paragraphs:
        tokens = tokenizer.encode(paragraph)
        input_ids = torch.tensor(tokens).unsqueeze(0)
        output = model(input_ids)
        cls_vectors.append(output.pooler_output)
    cls_tensor = torch.cat(cls_vectors)
    return cls_tensor


def save_cls_tensor(cls_tensor, prompt_number, cap):
    torch.save(cls_tensor, f"cls_tensors/cls_prompt{prompt_number:03d}_cap{cap:03d}.pt")
    

### driver ###
for prompt_number in range(10):
    cap = 190       # 200 led to tokenization of >512 length 
    responses = load_responses()
    paragraphs = get_capped_paragraphs(responses, prompt_number, cap)
    cls_tensor = paragraphs_to_CLS(paragraphs)
    save_cls_tensor(cls_tensor, prompt_number, cap)
"""