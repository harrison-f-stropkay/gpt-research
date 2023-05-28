# clustering (20v20)
import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score

from utilities import load_cls_tensor


def cluster_rands(clustering, cap):
    true_labels = np.concatenate((np.zeros(20, dtype=int), np.ones(20, dtype=int)))

    clustering_algo = None
    if clustering == "k-means":
        clustering_algo = KMeans(n_clusters=2, n_init="auto")
    elif clustering == "HAC":
        clustering_algo = AgglomerativeClustering(n_clusters=2)

    all_rands = np.zeros((10, 10))
    for prompt_number in range(10):
        cls_tensor = load_cls_tensor(prompt_number, cap)
        cls_array = cls_tensor.detach().numpy()
        
        # prompts in rows, temps in columns
        rands = np.zeros(10)
        for temp in range(10):
            binary_array = np.vstack((cls_array[20 * temp : 20 * (temp + 1), :], cls_array[200:, :]))
            labels = clustering_algo.fit_predict(binary_array)
            rand_index = adjusted_rand_score(true_labels, labels)
            rands[temp] = rand_index
        all_rands[prompt_number, :] = rands
    return all_rands


def plot_rands(cap):
    _, axs = plt.subplots(2, 2, figsize=(10, 15))

    for i, clustering in enumerate(["k-means", "HAC"]):
        all_rands = cluster_rands(clustering, cap)

        prompt_means = np.mean(all_rands, axis=1)
        temp_means = np.mean(all_rands, axis=0)
        with open("clustering/prompt_means", 'a') as file:
            json.dump(prompt_means.tolist(), file)
        with open("clustering/temp_means", 'a') as file:
            json.dump(temp_means.tolist(), file)

        temp_vals = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        prompt_vals = np.arange(10)
        
        for j in range(2):
            vals = temp_vals if j == 0 else prompt_vals
            means = temp_means if j == 0 else prompt_means
            avg_type = "temperature" if j == 0 else "prompt"
            axs[j, i].scatter(vals, means)
            axs[j, i].set_title(f"{clustering}: 20 vs 20, average over {avg_type}")
            axs[j, i].set_xlabel(avg_type)
            axs[j, i].set_ylabel("rand index")
            axs[j, i].set_xticks(vals)
            axs[j, i].set_ylim(0, 1)

    plt.savefig(f"clustering/20v20-kmeans-hac.png")


### driver ###
plot_rands(190)
