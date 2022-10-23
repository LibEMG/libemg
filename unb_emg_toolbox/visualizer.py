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
        colors = {}

        plt.gca().set_ylim([0, 1.05])
        plt.gca().xaxis.grid(False)

        # Plot true class labels
        testing_labels = self.offline_classifier.data_set['testing_labels'].copy()
        changed_locations = [0] + list(np.where((testing_labels[:-1] != testing_labels[1:]) == True)[0]) + [len(testing_labels)-1]

        for i in range(1, len(changed_locations)):
            class_label = testing_labels[changed_locations[i]]
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
            val = plt.scatter(i, probabilities[i], label=g, alpha=1, color=colors[g])
        
        plt.legend(loc='lower right')
        plt.show()

    def online_decision_stream():
        for i in range(100):
            y = random.uniform(0,1)
            plt.scatter(i, y)
            plt.pause(0.5)
        plt.show()