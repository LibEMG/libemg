[View Source Code](https://github.com/libemg/LibEMG_DeepLearning_Showcase)

<style>
    .center {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 50%;
    }
</style>

Although there are many options for prebuilt classifiers incorporated into LibEMG, we also provide the ability to use custom classifiers for offline or online evaluation of EMG signals. As a demonstration of how to get custom classifiers working, in this example we will be training a convolutional neural network with PyTorch and using it as the classifier of the EMGClassifier object. This approach lets the designer provide any sort of classification strategy they want to implement, while still providing the helpful utilities packaged in the EMGClassifier, like majority vote, rejection, and velocity control.

Fortunately, we can leverage nearly the entire pipeline code that we have used before for setting up pipeline. We can leverage existing datasets by loading them into an OfflineDataHandler object. After loading the dataset in, we can divide the dataset into a training (first 4 reps), validation (5th rep), and testing set (6th rep). Then, by applying filters and standardization (learned from the training set), we can ensure the inputs are reasonable for the neural network. Just like was done in the examples requiring handcrafted features, we can window our training, validation, and testing sets.

```Python
# make our results repeatable
fix_random_seed(seed_value=0, use_cuda=True)
# download the dataset from the internet
dataset = OneSubjectMyoDataset(save_dir='dataset/',
                        redownload=False)
odh = dataset.prepare_data(format=OfflineDataHandler)

# split the dataset into a train, validation, and test set
# this dataset has a "sets" metadata flag, so lets split 
# train/test using that.
not_test_data = odh.isolate_data("sets",[0,1,2,3,4])
test_data = odh.isolate_data("sets",[5])
# lets further split up training and validation based on reps
train_data = not_test_data.isolate_data("sets",[0,1,2,3])
valid_data = not_test_data.isolate_data("sets",[4])

# let's perform the filtering on the dataset too (neural networks like
# inputs that are standardized).
fi = Filter(sampling_frequency=200)
standardize_dictionary = {"name":"standardize", "data": train_data}
fi.install_filters(standardize_dictionary)
fi.filter(train_data)
fi.filter(valid_data)
fi.filter(test_data)

# for each of these dataset partitions, lets get our windows ready
window_size, window_increment = 50, 10
train_windows, train_metadata = train_data.parse_windows(window_size, window_increment)
valid_windows, valid_metadata = valid_data.parse_windows(window_size, window_increment)
test_windows,  test_metadata  = test_data.parse_windows( window_size, window_increment)
```

The main differences between handcrafted feature pipelines and deep learning pipelines begins now. For handcrafted features we would have proceeded to extracting our features, but for deep learning we use the windows themselves as the input. For later processing, we can use PyTorch's built-in dataset and dataloader classes to make grabbing batches of inputs and labels easier. The dataset object we define requires two methods to interface with the dataloader, a __getitem__() method and a __len__() method. The __getitem__() method takes in an index and returns a tuple of all things desired for that sample (in this case the data and label, but we could include other labels like a limb position identifier or subject identifier). The __len__() method just returns the total number of samples in the dataset. We then can define a dataloader, which is a convinient object for grabbing batches of the tuples defined in the __getitem__() method. In PyTorch's implementation, we can provide a reference to our dataset class, the batch size we desire, and a collate method (how we are collecting many __getitem__() calls into a single torch tensor).

```Python
#------------------------------------------------#
#            Interfacing with data               #
#------------------------------------------------#
# we require a class for our dataset that has the windows and classes saved
# it needs to have a __getitem__ method that returns the data and label for that id.
# it needs to have a __len__     method that returns the number of samples in the dataset.
class DL_input_data(Dataset):
    def __init__(self, windows, classes):
        self.data = torch.tensor(windows, dtype=torch.float32)
        self.classes = torch.tensor(classes, dtype=torch.long)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.data[idx]
        label = self.classes[idx]
        return data, label

    def __len__(self):
        return self.data.shape[0]

def make_data_loader(windows, classes, batch_size=64):
    # first we make the object that holds the data
    obj = DL_input_data(windows, classes)
    # and now we make a dataloader with that object
    dl = DataLoader(obj,
    batch_size=batch_size,
    shuffle=True,
    collate_fn = collate_fn)
    return dl

def collate_fn(batch):
    # this function is used internally by the dataloader (see line 46)
    # it describes how we stitch together the examples into a batch
    signals, labels = [], []
    for signal, label in batch:
        # concat signals onto list signals
        signals += [signal]
        labels += [label]
    # convert back to tensors
    signals = torch.stack(signals)
    labels = torch.stack(labels).long()
    return signals, labels
```

With the infrastructure of the dataset complete, we can now get to the exciting part which is constructing and training the neural network classifier! We can define a relatively shallow three layer convolutional neural network. In the constructor, we define the layers we wish to have in the network. We then define how inputs are passed through these layers in the forward method. We can also include .fit(), .predict(), and .predict_proba() methods which train the neural network, provide the class label for the input, or provide the class-probabilities of the outputs, respectively. By defining these three methods, we satisfy everything the EMGClassifier object will be looking for when this network is incorporated into the pipeline.

```Python
#------------------------------------------------#
#             Deep Learning Model                #
#------------------------------------------------#
# we require having forward, fit, predict, and predict_proba methods to interface with the 
# EMGClassifier class. Everything else is extra.
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CNN(nn.Module):
    def __init__(self, n_output, n_channels, n_samples, n_filters=256):
        super().__init__()
        # let's have 3 convolutional layers that taper off
        l0_filters = n_channels
        l1_filters = n_filters
        l2_filters = n_filters // 2
        l3_filters = n_filters // 4
        # let's manually setup those layers
        # simple layer 1
        self.conv1 = nn.Conv1d(l0_filters, l1_filters, kernel_size=5)
        self.bn1   = nn.BatchNorm1d(l1_filters)
        # simple layer 2
        self.conv2 = nn.Conv1d(l1_filters, l2_filters, kernel_size=5)
        self.bn2   = nn.BatchNorm1d(l2_filters)
        # simple layer 3
        self.conv3 = nn.Conv1d(l2_filters, l3_filters, kernel_size=5)
        self.bn3   = nn.BatchNorm1d(l3_filters)
        # and we need an activation function:
        self.act = nn.ReLU()
        # now we need to figure out how many neurons we have at the linear layer
        # we can use an example input of the correct shape to find the number of neurons
        example_input = torch.zeros((1, n_channels, n_samples),dtype=torch.float32)
        conv_output   = self.conv_only(example_input)
        size_after_conv = conv_output.view(-1).shape[0]
        # now we can define a linear layer that brings us to the number of classes
        self.output_layer = nn.Linear(size_after_conv, n_output)
        # and for predict_proba we need a softmax function:
        self.softmax = nn.Softmax(dim=1)

        self.to("cuda" if torch.cuda.is_available() else "cpu")
        

    def conv_only(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act(x)
        return x

    def forward(self, x):
        x = self.conv_only(x)
        x = x.view(x.shape[0],-1)
        x = self.act(x)
        x = self.output_layer(x)
        return self.softmax(x)

    def fit(self, dataloader_dictionary, learning_rate=1e-3, num_epochs=100, verbose=True):
        # what device should we use (GPU if available)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # get the optimizer and loss function ready
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        loss_function = nn.CrossEntropyLoss()
        # setup a place to log training metrics
        self.log = {"training_loss":[],
                    "validation_loss": [],
                    "training_accuracy": [],
                    "validation_accuracy": []} 
        # now start the training
        for epoch in range(num_epochs):
            #training set
            self.train()
            for data, labels in dataloader_dictionary["training_dataloader"]:
                optimizer.zero_grad()
                data = data.to(device)
                labels = labels.to(device)
                output = self.forward(data)
                loss = loss_function(output, labels)
                loss.backward()
                optimizer.step()
                acc = sum(torch.argmax(output,1) == labels)/labels.shape[0]
                # log it
                self.log["training_loss"] += [(epoch, loss.item())]
                self.log["training_accuracy"] += [(epoch, acc.item())]
            # validation set
            self.eval()
            for data, labels in dataloader_dictionary["validation_dataloader"]:
                data = data.to(device)
                labels = labels.to(device)
                output = self.forward(data)
                loss = loss_function(output, labels)
                acc = sum(torch.argmax(output,1) == labels)/labels.shape[0]
                # log it
                self.log["validation_loss"] += [(epoch, loss.item())]
                self.log["validation_accuracy"] += [(epoch, acc.item())]
            if verbose:
                epoch_trloss = np.mean([i[1] for i in self.log['training_loss'] if i[0]==epoch])
                epoch_tracc  = np.mean([i[1] for i in self.log['training_accuracy'] if i[0]==epoch])
                epoch_valoss = np.mean([i[1] for i in self.log['validation_loss'] if i[0]==epoch])
                epoch_vaacc  = np.mean([i[1] for i in self.log['validation_accuracy'] if i[0]==epoch])
                print(f"{epoch}: trloss:{epoch_trloss:.2f}  tracc:{epoch_tracc:.2f}  valoss:{epoch_valoss:.2f}  vaacc:{epoch_vaacc:.2f}")
        self.eval()

    def predict(self, x):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if type(x) != torch.Tensor:
            x = torch.tensor(x, dtype=torch.float32)
        x = x.to(device)
        y = self.forward(x)
        predictions = torch.argmax(y, dim=1)
        return predictions.cpu().detach().numpy()

    def predict_proba(self, x):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if type(x) != torch.Tensor:
            x = torch.tensor(x, dtype=torch.float32)
        x = x.to(device)
        y = self.forward(x)
        return y.cpu().detach().numpy()
```

Back in the main code, we can now construct our dataset and dataloaders using the windows and classes of the training and validation sets:

```Python
# libemg supports deep learning, but we need to prepare the dataloaders
train_dataloader = make_data_loader(train_windows, train_metadata["classes"])
valid_dataloader = make_data_loader(valid_windows, valid_metadata["classes"])

# let's make the dictionary of dataloaders
dataloader_dictionary = {"training_dataloader": train_dataloader,
                            "validation_dataloader": valid_dataloader}
```

We can initialize our model:

```Python
# We need to tell the libEMG EMGClassifier that we are using a custom model
model = CNN(n_output   = np.unique(np.vstack(odh.classes[:])).shape[0],
            n_channels = train_windows.shape[1],
            n_samples  = train_windows.shape[2],
            n_filters  = 64)
```

And we can train this model with hyperparameters we specify:

```Python
# we can even make a dictionary of parameters that get passed into 
# the training process of the deep learning model
dl_dictionary = {"learning_rate": 1e-4,
                    "num_epochs": 50,
                    "verbose": True}
```

With these things defined, we can finally interface the CNN model with LibEMG. We can pass our PyTorch classifier to the EMGClassifier.fit() method as the chosen classifier. We can also pass our deep learning dictionary as the positional dataloader_dictionary argument to tell the library we are training a deep learning model. Finally, the deep learning hyperparameter dictionary is passed in with the parameters keyword (note: the keys of this dictionary are positional arguments in the .fit() function).

```Python
# Now that we've made the custom classifier object, libEMG knows how to 
# interpret it when passed in the dataloader_dictionary. Everything happens behind the scenes.
classifier = EMGClassifier(model)
classifier.fit(dataloader_dictionary=dataloader_dictionary, parameters=dl_dictionary)
```

After the classifier has been trained, we can use this EMGClassifier object the same way we always have done, despite now using a neural network.

Getting offline metrics for the test set using this classifier:

```Python
# get the classifier's predictions on the test set
preds = classifier.run(test_windows)
om = OfflineMetrics()
metrics = ['CA','AER','INS','REJ_RATE','CONF_MAT','RECALL','PREC','F1']
results = om.extract_offline_metrics(metrics, test_metadata['classes'], preds[0], null_label=2)
for key in results:
    print(f"{key}: {results[key]}")
```

We can even use the post-processing available to the EMGClassifier with custom classifiers!

```Python
classifier.add_majority_vote(3)
classifier.add_rejection(0.9)
classifier.add_velocity(train_windows, train_metadata["classes"])
```
