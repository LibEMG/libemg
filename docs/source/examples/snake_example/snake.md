<style>
    .box {
        margin:0;
        width:50%;
        float:left;
    }
</style>
[View Source Code](https://github.com/eeddy/Snake-Demo)

In this example, we created an adapted version of the traditional snake game. The idea of the game is that the player will collect as much "food" as possible to grow the snake. 
<div>
    <div class="box"> 
        <center><p> <b> <u> Myo </u> </b> </p></center>
        <img src="https://github.com/eeddy/Snake-Demo/blob/main/docs/myo_game.gif?raw=true"/>
    </div>
    <div class="box"> 
    <center><p> <b> <u> Delsys </u> </b> </p></center>
        <img src="https://github.com/eeddy/Snake-Demo/blob/main/docs/delsys.gif?raw=true" width="100%"/>
    </div>
</div>

# Game Design
To explore the game design code, please review `snake_game.py`. This section, however, focuses on the important design considerations for interfacing the game with EMG-based input.

Most game engines (including pygame) update the graphical user interface (GUI) based on a looping system. By setting `self.clock.tick(30)` the screen refresh rate is capped at 30 Hz. As a result, the GUI will update 30 times every second. Interestingly, this has significant consequences for our EMG-based control system. The important thing to note here is that the `handle_movement` function is called in this loop - meaning it is called 30 times a second.

```Python
# Game loop 
def run_game(self):
    while self.running: 
        # Listen for movement events
        self.handle_movement()

        # Additional game related code... (see snake_game.py)

        pygame.display.update()
        # Refresh rate is 30 Hz 
        self.clock.tick(30)

    pygame.quit()
```

In the initial version, the arrow keys were used to move the snake. Note that we are appending the movement direction to an array to keep track of the previous movement of all parts of the snake. The `move_snake` function moves each part of the snake in the specified direction.

```Python
def handle_movement(self):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            self.running = False 
            
        # Listen for key presses:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                self.previous_key_presses.append("left")
            elif event.key == pygame.K_RIGHT:
                self.previous_key_presses.append("right")
            elif event.key == pygame.K_UP:
                self.previous_key_presses.append("up")
            elif event.key == pygame.K_DOWN:
                self.previous_key_presses.append("down")
            else:
                return 

            self.move_snake()
```

Now let's look at how we can move the snake using EMG-based input. As our library streams all classification decisions over UDP, we need to set up a socket listening on the specified port and IP. By default, the OnlineEMGClassifier streams at `port:12346` and `ip:'127.0.0.1'`. 
```Python
# Socket for reading EMG
self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) 
self.sock.bind(('127.0.0.1', 12346))
```

After setting up the socket, moving the snake is almost identical to using key presses. Instead of looking for a specific key press, however, we check for the predicted class label and map that to a direction. For example, we have mapped 0-down, 1-up, 3-right, and 4-left. Otherwise, the code to handle the snake movement is nearly identical.
```Python
def handle_emg(self):
    data, _ = self.sock.recvfrom(1024)
    data = str(data.decode("utf-8"))
    if data:
        input_class = float(data.split(' ')[0])
        # 0 = Hand Closed = down
        if input_class == 0:
            self.previous_key_presses.append("down")
        # 1 = Hand Open
        elif input_class == 1:
            self.previous_key_presses.append("up")
        # 3 = Extension 
        elif input_class == 3:
            self.previous_key_presses.append("right")
        # 4 = Flexion
        elif input_class == 4:
            self.previous_key_presses.append("left")
        else:
            return
        
        self.move_snake()
```

# Menu
Now that we have shown how to leverage EMG predictions to replace traditional key presses for snake control, we need to explore the design of the control system. The first step in any EMG-based control scheme is accumulating training data to train the ML model. Leveraging the Training UI module, we have built the data accumulation directly into the game menu. This can be found in `game_menu.py`.

<div>
    <img src="https://github.com/eeddy/Snake-Demo/blob/main/docs/menu.PNG?raw=true" width="32%" display="inline-block" float="left"/>
    <img src="https://github.com/eeddy/Snake-Demo/blob/main/docs/training_screen1.PNG?raw=true" width="32%" float="left"/>
    <img src="https://github.com/eeddy/Snake-Demo/blob/main/docs/training_screen2.PNG?raw=true" width="32%" float="left"/>
</div>

First, lets look at the required imports from libemg:
```Python
from libemg.screen_guided_training import ScreenGuidedTraining
from libemg.data_handler import OnlineDataHandler, OfflineDataHandler
from libemg.streamers import myo_streamer
from libemg.utils import make_regex
from libemg.feature_extractor import FeatureExtractor
from libemg.emg_classifier import OnlineEMGClassifier, EMGClassifier
```

Note that we are passing an online data handler to the training UI. This same data handler will be used for training and classification.
```Python
self.odh = OnlineDataHandler()
self.odh.start_listening()
```

When the data accumulation button is pressed, we present the training UI. After finishing the data accumulation, we re-initialize the simple game menu. Note, that using the LibEMGGesture library, we automatically download the images for each class if they do not already exist.
```Python
def launch_training(self):
    # Destroy the menu 
    self.window.destroy()
    # Launch training ui
    training_ui = ScreenGuidedTraining()
    training_ui.download_gestures([1,2,3,4,5], "images/")
    training_ui.launch_training(self.odh, 2, 3, "images/", "data/", 1)
    self.initialize_ui()
```

When the play button is clicked, we create and start the game. However, the first thing that we have to do is spin up an `OnlineEMGClassifier` to make predictions in a separate process that the game can use.

```Python
def play_snake(self):
    self.window.destroy()
    self.set_up_classifier()
    SnakeGame().run_game()
    # Its important to stop the classifier after the game has ended
    # Otherwise it will continuously run in a seperate process
    self.classifier.stop_running()
    self.initialize_ui()
```

Creating an online classifier is quite easy when leveraging `libemg`. Step 1 involves parsing the accumulated training data for a particular user. This training data can be then split into `train_windows` and `train_metadata`. The reps values will be dependent on how many training reps are acquired for each class.

```Python
# Step 1: Parse offline training data
dataset_folder = 'data/'
classes_values = ["0","1","2","3","4"]
classes_regex = make_regex(left_bound = "_C_", right_bound=".csv", values = classes_values)
reps_values = ["0", "1", "2"]
reps_regex = make_regex(left_bound = "R_", right_bound="_C_", values = reps_values)
dic = {
    "reps": reps_values,
    "reps_regex": reps_regex,
    "classes": classes_values,
    "classes_regex": classes_regex
}

odh = OfflineDataHandler()
odh.get_data(folder_location=dataset_folder, filename_dic=dic, delimiter=",")
train_windows, train_metadata = odh.parse_windows(WINDOW_SIZE, WINDOW_INCREMENT)
```

Step 2 involves extracting features from the train_windows, as the online classifier expects features as input. We have decided to extract Hudgin's Time Domain features. However, testing different features or feature groups is trivial.
```Python
# Step 2: Extract features from offline data
fe = FeatureExtractor()
feature_list = fe.get_feature_groups()['HTD']
training_features = fe.extract_features(feature_list, train_windows)
```

Step 3 involves setting up the dataset dictionary that is passed into the classifier. 
```Python
# Step 3: Dataset creation
data_set = {}
data_set['training_features'] = training_features
data_set['training_labels'] = train_metadata['classes']
```

Step 4 involves setting up the EMGClassifier for the OnlineClassifier. We have chosen an LDA model since we know it works well and is reliable.
```Python
# Step 4: Create the EMG Classifier
o_classifier = EMGClassifier()
o_classifier.fit(model="LDA", feature_dictionary=data_set)
```

Finally, in step 5 we train an online EMG classifier and begin streaming predictions.
```Python
# Step 5: Create online EMG classifier and start classifying.
self.classifier = OnlineEMGClassifier(o_classifier, WINDOW_SIZE, WINDOW_INCREMENT, self.odh, feature_list)
self.classifier.run(block=False) # block set to false so it will run in a seperate process.
```

Notice that the online classifier has a window size and window increment option. Since the refresh rate of the game is 30 Hz, the minimum increment in time would be $\frac{1000\text{ms}}{30hz}=33 \text{ms}$. Anything less than this would result in decisions that would not be leveraged by the game. Selecting these values is not trivial and often takes experimentation. A smaller increments result in more responsive systems, but this may not be desired. A larger window size may result in more accurate predictions but introduces a lag into the system. In this example, we select a window size of 0.5s (100 samples for the Myo) and an increment of 0.25s (50 samples for the Myo). Since we move the snake 20 pixels for every input, the snake moves a total of 80 pixels in a given second (assuming no decisions are rejected).

```Python
WINDOW_SIZE = 100 
WINDOW_INCREMENT = 50
```