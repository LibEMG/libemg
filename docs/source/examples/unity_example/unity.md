In [Example 1](https://github.com/eeddy/Snake-Demo/blob/main/README.md), we showed how to leverage the [UNB_EMG_Toolbox](https://eeddy.github.io/unb_emg_toolbox/) to interface with a simple pygame. Often, however, it may be desirable to use a different game engine or tech stack. For example, Unity is a common game development environment that enables high-quality games and VR/AR development. As such, developers may want to use it for their EMG-related applications. The good news is that these tools can still leverage the toolkit very easily! This example shows how to leverage the toolkit using a simple Unity game. It is important to note that although this is a simple game, the concept is applicable to more complex applications (e.g., AR/VR). 

<img src="https://github.com/eeddy/Momo-Demo/blob/main/Docs/Momo_Myo.gif?raw=True">

# The Fall of Momo
The Fall of Momo is a simple platformer game that was designed for myoeletric training purposes <sup>[1,2]</sup>. The goal of the game is to control the character "Momo" down the screen and avoid the spikes. Originally, this game was built in processing and the original version can be found [here](https://github.com/hcilab/Momo). We have recreated a simplified version in Unity for this demo. In this version, the 3 inputs and their respective controls are:

| Game Movement | Keyboard | EMG |
| --- | ----------- | --- |
| Move Left | Left Arrow Key | Wrist Flexion |
| Move Right | Right Arrow Key | Wrist Extension |
| Jump | Space Bar | Hand Closed |

\*Note: These controls are set up for playing with the right arm.

# Momo Unity Development
The first thing that we did was create the Momo-Unity game. There are many great online Unity tutorials, so we won't get into the intricate details of the game design. 

<div>
    <img src="https://github.com/eeddy/Momo-Demo/blob/main/Docs/main_menu.PNG?raw=True" width="47%"display="inline-block" float="left"/>
    <img src="https://github.com/eeddy/Momo-Demo/blob/main/Docs/game.PNG?raw=True" width="47%"  isplay="inline-block" float="left"/>
</div>

In the initial game design, we controlled the character using the keyboard. Similarly to other game engines, Unity updates in a loop at a default of 60Hz. This occurs in an update method that gets added to a C# file. As displayed bellow in `MovementController.cs` we have created a script that listens for key events in the update method (i.e., 60 times a second) and reacts accordingly. The `Rigidbody2D` is simply just a link to the "Momo" character rigidbody.

```C#
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MovementController : MonoBehaviour
{
    private float speed = 5;
    private float upwardsForce = 300;
    public Rigidbody2D rb;
    private Vector2 velocity;

    private SoundManager soundManager; 

    void Start() {
        soundManager = FindObjectOfType<SoundManager>();
    }

    void Update()
    {
        Vector3 pos = rb.transform.position;
        if (Input.GetKey(KeyCode.LeftArrow)) {
            pos.x -= speed * Time.deltaTime;
        } else if (Input.GetKey(KeyCode.RightArrow)) {
            pos.x += speed * Time.deltaTime;
        } else if (Input.GetKeyDown(KeyCode.Space)) {
            rb.AddForce(new Vector2(0,1) * upwardsForce);
            soundManager.PlayJumpSound();
        }
        rb.transform.position = pos;
    }
}
```

# Momo EMG Control
Once we developed the initial game to work with simple keyboard controls, we then implemented the EMG-based input. Unfortunately, since the toolbox is written in Python, we had to include the machine learning/training portion as a Python application. While there may be ways to call Python from within C#, this was outside the scope of this example. Instead, we created a simple UI with two buttons: `Get Training Data` and `Start Classifying`. All python code is located in `myo_control.py`. The toolkit imports required for this example are as follows:

```Python
from unb_emg_toolbox.training_ui import TrainingUI
from unb_emg_toolbox.data_handler import OnlineDataHandler, OfflineDataHandler
from unb_emg_toolbox.utils import make_regex
from unb_emg_toolbox.feature_extractor import FeatureExtractor
from unb_emg_toolbox.emg_classifier import OnlineEMGClassifier
```

<div>
    <img src="https://github.com/eeddy/Momo-Demo/blob/main/Docs/menu.PNG?raw=True" width="31%" float="left"/>
    <img src="https://github.com/eeddy/Snake-Demo/blob/main/docs/training_screen1.PNG?raw=True" width="31%" float="left"/>
    <img src="https://github.com/eeddy/Snake-Demo/blob/main/docs/training_screen2.PNG?raw=True" width="31%" float="left"/>
</div>

When the `Get Training Data` button is clicked we leverage the toolkit's Training UI module. To do this, we simply create the class and it handles the rest. Note that we also have a folder `Class_Images` with images associated with each class (No Movement, Flexion, Extension, and Hand Closed). All recorded EMG files will be written to the `data/` folder. 

```Python
def launch_training(self):
    self.window.destroy()
    # Launch training ui
    TrainingUI(num_reps=3, rep_time=3, rep_folder="Class_Images/", output_folder="data/", data_handler=self.odh)
    self.initialize_ui()
```

After training data is accumulated, you can start classifying predictions over a TCP socket so that they can be leveraged in unity. To do this we must create an `OnlineEMGClassifier` object.

The first step involes passing in the accumulated data into an `OfflineDataHandler`. Note that there are 4 classes [0,1,2,3] and 3 reps [0,1,2] - this aligns with the training data that we recorded.
```Python
# Step 1: Parse offline training data
dataset_folder = 'data/'
classes_values = ["0","1","2","3"]
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

The next step involves extracting features from the offline data. Lets experiment with the LS9 feature group.
```Python
# Step 2: Extract features from offline data
fe = FeatureExtractor(num_channels=8)
feature_list = fe.get_feature_groups()['LS9']
training_features = fe.extract_features(feature_list, train_windows)
```

After extracting the features from the training data, we have to create a dataset dictionary to pass to the online classifier. Note that we have also included the training_windows. This is done so that we can leverage velocity based control. 
```Python
data_set = {}
data_set['training_features'] = training_features
data_set['training_labels'] = train_metadata['classes']
data_set['training_windows'] = train_windows
```

Finally, lets create the `OnlineEMGClassifier` and begin streaming predictions. Note that we set block to false so that we don't block the UI thread. 

```Python
 # Step 4: Create online EMG classifier and start classifying.
self.classifier = OnlineEMGClassifier(model="SVM", data_set=data_set, num_channels=8, window_size=WINDOW_SIZE, window_increment=WINDOW_INCREMENT, 
                online_data_handler=self.odh, features=feature_list, rejection_type="CONFIDENCE", rejection_threshold=0.75, majority_vote=10, velocity=True, std_out=True)
self.classifier.run(block=False) # block set to false so it will run in a seperate process.
```

There is a lot to unpack in this online classifier, so let's go through it:
- **model="SVM":** In this example, we are using a Support Vector Machine for classification. 
- **rejection_type:** Since rejection is known to improve usability, we have decided to include it. 
- **rejection_threshold:** Since SVM is known to have a greater range of probability outputs (compared to LDA, for example), we have a much lower rejection threshold. If we set this too high, the majority of decisions will inevitably be rejected.
- **majority_vote:** To reduce spurious false activations (especially of the hand-closed class), we have decided to introduce a majority vote.
- **velocity:** Finally, we decided to leverage velocity-based control to augment the experience. This means that when users contract harder, their character will move faster. 

Now that we have the Python side set up, we had to create a way to listen for these TCP events in C#. To do this, we created `MyoEMGRawReader.cs`. It spins up a thread to continuously listen on a specific port (in this case 12346). Note that the IP and port are the default values that the OnlineEMGClassifier streams over. Every time it receives a value, it updates a global control and speed variable that can be used by the `MovementControllerEMG`.

```C#
String control = "";
float speed = 0.0f;

public string IP = "127.0.0.1";
public int port = 12346;

// receive thread function
private void ReceiveData()
{
    client = new UdpClient(port);
    while (true)
    {
        // receive bytes
        IPEndPoint anyIP = new IPEndPoint(IPAddress.Any, 0);
        byte[] buff = client.Receive(ref anyIP);

        // encode UTF8-coded bytes to text format
        string text = Encoding.UTF8.GetString(buff);
        string[] parts = text.Split(' ');
        control = parts[0];
        speed = float.Parse(parts[1]);
    }
}
```

```C#
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MovementControllerEMG : MonoBehaviour
{
    private float upwardsForce = 300;
    public Rigidbody2D rb;
    private Vector2 velocity;
    private MyoEMGRawReader emgReader;
    private float jumpTime;

    private SoundManager soundManager;

    void Start() {
        soundManager = FindObjectOfType<SoundManager>();
        emgReader = new MyoEMGRawReader();
        emgReader.StartReadingData();
    }

    void Update()
    {
        string control = emgReader.ReadControlFromArmband();
        float movSpeed = emgReader.ReadSpeedFromArmband();
        if(movSpeed > 5) {
            movSpeed = 5;
        }
        Vector3 pos = rb.transform.position;
        if (control == "0") {
            if (Time.time - jumpTime > 1.0f) {
                rb.AddForce(new Vector2(0,1) * 300);
                soundManager.PlayJumpSound();
                jumpTime = Time.time;
            }
        } else if (control == "2") {
            //Extension:
            pos.x += (movSpeed) * Time.deltaTime;
        } else if (control == "3") {
            //Flexion:
            pos.x -= (movSpeed) * Time.deltaTime;
        }
        rb.transform.position = pos;
    }
}
```

# References
<a id="1">[1]</a> 
Aaron Tabor, Scott Bateman, Erik Scheme, David R. Flatla, and Kathrin Gerling. 2017. Designing Game-Based Myoelectric Prosthesis Training. In Proceedings of the 2017 CHI Conference on Human Factors in Computing Systems (CHI '17). Association for Computing Machinery, New York, NY, USA, 1352–1363. https://doi-org.proxy.hil.unb.ca/10.1145/3025453.3025676

<a id="2">[2]</a>
A. Tabor, S. Bateman and E. Scheme, "Evaluation of Myoelectric Control Learning Using Multi-Session Game-Based Training," in IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 26, no. 9, pp. 1680-1689, Sept. 2018, doi: 10.1109/TNSRE.2018.2855561.