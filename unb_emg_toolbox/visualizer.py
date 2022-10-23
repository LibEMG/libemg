from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import time
import socket
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from unb_emg_toolbox import raw_data
from datetime import datetime

from unb_emg_toolbox.feature_extractor import FeatureExtractor
from unb_emg_toolbox.utils import get_windows

plt.style.use('ggplot')

def plot_decision_stream(y_labels, predictions, probabilities):
    colors = {}

    plt.gca().set_ylim([0, 1.05])
    plt.gca().xaxis.grid(False)

    # Plot true class labels
    changed_locations = [0] + list(np.where((y_labels[:-1] != y_labels[1:]) == True)[0]) + [len(y_labels)-1]

    for i in range(1, len(changed_locations)):
        class_label = y_labels[changed_locations[i]]
        if class_label in colors.keys():
            plt.fill_betweenx([0,1.02], changed_locations[i-1], changed_locations[i], color=colors[class_label])
        else:
            val = plt.fill_betweenx([0,1.02], changed_locations[i-1], changed_locations[i], alpha=.2)
            colors[class_label] = val.get_facecolors().tolist()
        
    # Plot decision stream
    plt.title("Decision Stream")
    plt.xlabel("Class Output")
    plt.ylabel("Probability")
    for g in np.unique(predictions):
        i = np.where(predictions == g)
        if g == -1:
            plt.scatter(i, probabilities[i], label=g, alpha=1, color='black')
        else:
            plt.scatter(i, probabilities[i], label=g, alpha=1, color=colors[g])
    
    plt.legend(loc='lower right')
    plt.show()

def plot_pca(data, labels):
    pca = PCA(n_components=2)
    standardized = StandardScaler().fit_transform(data)
    principle_components = pca.fit_transform(standardized)
    
    plt.title("PCA Plot")
    plt.xlabel("Principle Component 1")
    plt.ylabel("Principle Component 2")
    for g in np.unique(labels):
        i = np.where(labels == g)
        pcs = principle_components[i]
        plt.scatter(pcs[:,0], pcs[:,1], label=g)
    
    plt.legend()
    plt.show()

def plot_raw_emg(data, channels=None):
    plt.title("Raw Data")
    _plot_raw_emg_helper(data, channels)
    plt.show()

def plot_features(windows, features, num_channels, channels=None):
    fig, ax = plt.subplots(len(features))
    fe = FeatureExtractor(num_channels=num_channels)
    _plot_features_helper(windows, features, channels, fig, ax, fe)
    plt.show()

def plot_live_emg(online_data_handler, channels=None, num_samples=500):
    plt.title("Raw Data")
    while True:
        data = np.array(online_data_handler.raw_data.get_emg())
        if len(data) > num_samples:
            data = data[-num_samples:]
        if len(data) > 0:
            _plot_raw_emg_helper(data, channels)
        plt.pause(0.1)

def _plot_features_helper(windows, features, channels, fig, ax, fe):
    ex_features = fe.extract_features(features, windows)
    index = 0
    labels = []
    for f in ex_features:
        for i in range(0,len(ex_features[f][0])):
            if channels is None or i in channels:
                x = list(range(0,len(ex_features[f])))
                lab = "CH"+str(i)
                ax[index].plot(x, ex_features[f][:], label=lab)
                if not lab in labels:
                    labels.append(lab)
            ax[index].set_ylabel(f)
        index += 1
        fig.suptitle('Features')
        fig.legend(labels, loc='upper right')
    plt.draw()

def _plot_raw_emg_helper(data, channels=None):
    plt.clf()
    plt.title("Raw Data")
    for i in range(0,len(data[0])):
        if channels is None or i in channels:
            x = list(range(0,len(data)))
            plt.plot(x, data[:,i], label="CH"+str(i))
    plt.legend(loc = 'lower right')