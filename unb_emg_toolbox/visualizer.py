import random
from turtle import color
import numpy as np
import matplotlib.pyplot as plt

class EMGVisualizer:
    def __init__(self, emg_data, offline_classifier):
        plt.style.use('ggplot')
        self.emg_data = emg_data
        self.offline_classifier = offline_classifier

    def plot_offline_decision_stream(self):
        predictions = self.offline_classifier.predictions.copy()
        probabilities = self.offline_classifier.probabilities.copy()
        colors = []

        plt.gca().set_ylim([0, 1.05])
        plt.gca().xaxis.grid(False)

        # Plot decision stream
        plt.title("Decision Stream")
        plt.xlabel("Class Output")
        plt.ylabel("Probability")
        for g in np.unique(predictions):
            i = np.where(predictions == g)
            val = plt.scatter(i, probabilities[i], label=g)
            colors.append(val.get_facecolors().tolist())
        
        # Plot true class labels
        testing_labels = self.offline_classifier.data_set['testing_labels'].copy()
        for g in np.unique(testing_labels):
            vals = np.where(testing_labels == g)[0]
            s_index = vals[0]
            for i in range(1, len(vals)):
                if vals[i] - vals[i-1] != 1:
                    plt.fill_betweenx([0,1.02], s_index, vals[i-1], alpha=.2, color=colors[int(g)])
                    s_index = vals[i]
            plt.fill_betweenx([0,1.02], s_index, vals[-1], alpha=.2, color=colors[int(g)])
        
        plt.legend(loc='lower right')
        plt.show()

    def online_decision_stream():
        for i in range(100):
            y = random.uniform(0,1)
            plt.scatter(i, y)
            plt.pause(0.5)
        plt.show()