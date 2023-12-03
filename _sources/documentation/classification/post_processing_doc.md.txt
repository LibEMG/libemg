# Post-Processing

## Rejection
Classifier outputs are overridden to a default or inactive state when the output decision is uncertain. This concept stems from the notion that it is often better (less costly) to incorrectly do nothing than it is to erroneously activate an output.  
- **Confidence <sup>[1]</sup>:** Rejects based on a predefined **confidence threshold** (between 0-1). If predicted probability is less than the confidence threshold, the decision is rejected. Figure 1 exemplifies rejection using an SVM classifier with a threshold of 0.8.

```Python
# Add rejection with 90% confidence threshold
classifier.add_rejection(threshold=0.9)
```

## Majority Voting <sup>[2,3]</sup>
Overrides the current output with the label corresponding to the class that occurred most frequently over the past $N$ decisions. As a form of simple low-pass filter, this introduces a delay into the system but reduces the likelihood of spurious false activations. Figure 1 exemplifies applying a majority vote of 5 samples to a decision stream.

```Python
# Add majority vote on 10 samples
classifier.add_majority_vote(num_samples=10)
```

## Velocity Control <sup>[4]</sup>
Outputs an associated *velocity* with each prediction that estimates the level of muscular contractions (normalized by the particular class). This means that within the same contraction, users can contract harder or lighter to control the velocity of a device. Note that ramp contractions should be accumulated during the training phase.

```Python
# Add velocity control
classifier.add_velocity(train_windows, train_labels)
```

Figure 1 shows the decision stream (i.e., the predictions over time) of a classifier with no post-processing, rejection, and majority voting. In this example, the shaded regions show the ground truth label, whereas the colour of each point represents the predicted label. All black points indicate predictions that have been rejected.

![alt text](decision_stream.png)
<center> <p> Figure 1: Decision Stream of No Post-Processing, Rejection, and Majority Voting. This can be created using the <b>.visualize()</b> method call. </p> </center>

## References
<a id="1">[1]</a> 
E. J. Scheme, B. S. Hudgins and K. B. Englehart, "Confidence-Based Rejection for Improved Pattern Recognition Myoelectric Control," in IEEE Transactions on Biomedical Engineering, vol. 60, no. 6, pp. 1563-1570, June 2013, doi: 10.1109/TBME.2013.2238939.

<a id="2">[2]</a> 
Scheme E, Englehart K. Training Strategies for Mitigating the Effect of Proportional Control on Classification in Pattern Recognition Based Myoelectric Control. J Prosthet Orthot. 2013 Apr 1;25(2):76-83. doi: 10.1097/JPO.0b013e318289950b. PMID: 23894224; PMCID: PMC3719876.

<a id="3">[3]</a> 
Wahid MF, Tafreshi R, Langari R. A Multi-Window Majority Voting Strategy to Improve Hand Gesture Recognition Accuracies Using Electromyography Signal. IEEE Trans Neural Syst Rehabil Eng. 2020 Feb;28(2):427-436. doi: 10.1109/TNSRE.2019.2961706. Epub 2019 Dec 23. PMID: 31870989.

<a id="4">[4]</a> 
E. Scheme, B. Lock, L. Hargrove, W. Hill, U. Kuruganti and K. Englehart, "Motion Normalized Proportional Control for Improved Pattern Recognition-Based Myoelectric Control," in IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 22, no. 1, pp. 149-157, Jan. 2014, doi: 10.1109/TNSRE.2013.2247421.

<a>[Sklearn]</a>
Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier Grisel, Mathieu Blondel, Peter Prettenhofer, Ron Weiss, Vincent Dubourg, Jake Vanderplas, Alexandre Passos, David Cournapeau, Matthieu Brucher, Matthieu Perrot, and Édouard Duchesnay. 2011. Scikit-learn: Machine Learning in Python. J. Mach. Learn. Res. 12, null (2/1/2011), 2825–2830.