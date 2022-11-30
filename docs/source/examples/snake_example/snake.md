<style>
    .box {
        margin:0;
        width:50%;
        float:left;
    }
</style>
[View Source Code](https://github.com/eeddy/Snake-Demo)

In this example, we created an adapted version of the traditional snake game. The idea of the game is that the player will collect as much food as possible to grow the snake. The only difference between our version and the original is that we require individual input for each movement. This is dissimilar to the traditional snake game, where the snake continuously moves in a specified direction until a new input. We made this change because the original control scheme does not lend well to traditional continuous myoelectric control.
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

Now let's look at how we can move the snake using EMG-based input. As our library streams all classification decisions over UDP, we need to set up a socket listening on the specified port and IP. By default, the OnlineEMGClassifier streams at `port:12346` and `ip:'127.0.0.1'`. We opted not to change these default values. 
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
Now that we have shown how to leverage EMG predictions to replace traditional key presses for snake control, we need to explore the design of the control system. But first, all of this is impossible without training data. Leveraging the Training UI module, we have built the data accumulation directly into the game menu. This can be found in `game_menu.py`.

<div>
    <img src="https://github.com/eeddy/Snake-Demo/blob/main/docs/menu.PNG?raw=true" width="32%" display="inline-block" float="left"/>
    <img src="https://github.com/eeddy/Snake-Demo/blob/main/docs/training_screen1.PNG?raw=true" width="32%" float="left"/>
    <img src="https://github.com/eeddy/Snake-Demo/blob/main/docs/training_screen2.PNG?raw=true" width="32%" float="left"/>
</div>

Note that we are passing an online data handler into the training UI. This same data handler will be used for training and classification.
```Python
self.odh = OnlineDataHandler(emg_arr=True)
self.odh.start_listening()
```

When the data accumulation button is pressed, we present the training UI. After finishing the data accumulation, we re-initialize the simple game menu.
```Python
def launch_training(self):
    # Destroy the menu 
    self.window.destroy()
    # Launch training ui
    TrainingUI(num_reps=1, rep_time=5, rep_folder="classes/", output_folder="data/", data_handler=self.odh)
    self.initialize_ui()
```

The other button option is to play the game. When it is clicked, we create and start the game. However, the first thing that we have to do is spin up an `OnlineEMGClassifier` to make predictions in a separate process that the game can use.

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

Creating an online classifier is quite easy when leveraging the `libemg`. Step 1 involves parsing the accumulated training data for a particular user. This training data can be then split into `train_windows` and `train_metadata`.

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

Step 2 involves extracting features from the train_windows, as the online classifier expects features as input. We have decided to extract Hudgin's Time Domain features. However, other features or feature groups could easily be explored.
```Python
# Step 2: Extract features from offline data
fe = FeatureExtractor(num_channels=8)
feature_list = fe.get_feature_groups()['HTD']
training_features = fe.extract_features(feature_list, train_windows)
```

Step 3 involves setting up the dataset dictionary that is passed into the online classifier. 
```Python
# Step 3: Dataset creation
data_set = {}
data_set['training_features'] = training_features
data_set['training_labels'] = train_metadata['classes']
```

Finally, in step 4 we train an online EMG classifier. We have chosen an LDA model since we know it works well and is reliable. We have also decided to introduce rejection, as it has been shown to improve usability. We can easily swap out parameters here and explore a number of classifiers if desired. 
```Python
# Step 4: Create online EMG classifier and start classifying.
self.classifier = OnlineEMGClassifier(model="LDA", data_set=data_set, num_channels=8, window_size=WINDOW_SIZE, window_increment=WINDOW_INCREMENT, 
        online_data_handler=self.odh, features=feature_list, rejection_type='CONFIDENCE', rejection_threshold=0.95)
self.classifier.run(block=False) # block set to false so it will run in a seperate process.
```

Notice that the online classifier has a window size and window increment option. Since the refresh rate of the game is 30 Hz, the minimum increment in time would be $\frac{1000\text{ms}}{30hz}=33 \text{ms}$. Anything less than this would result in decisions that would not be leveraged by the game. Selecting these values is not trivial and often takes experimentation. A smaller increment can enable more precision, but this may not be desired. A larger window size may result in more accurate predictions but introduces a lag into the system. In this example, we select a window size of 0.5s (100 samples for the Myo) and an increment of 0.25s (50 samples for the Myo). Since we move the snake 20 pixels for every input, the snake moves a total of 80 pixels in a given second (assuming no decisions are rejected).

```Python
WINDOW_SIZE = 100 
WINDOW_INCREMENT = 50
```

# EMG Streamers
Finally, we needed to add a streamer for each of the devices. These streamers read the raw data and stream them over UDP to be read by the `OnlineDataHandler`. We have included streamers for the `Myo`, `Delsys`, and `SIFI Labs Armband`. These streamers can be found in `streamers.py`. 