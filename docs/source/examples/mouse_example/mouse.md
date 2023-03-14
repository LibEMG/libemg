**Note, this controller is by no means a robust mouse interface and is is instead a simple demonstration highlighting the possibilities of the library.**

[View Source Code](https://github.com/anon/Mouse)

The main control scheme leveraged by the prosthetics community involves continuously predicting user intent to micro-adjust a prosthesis. These predictions occur for relatively small windows of data (in the order of milliseconds) to enable responsive controllers. For general-purpose computing, continuous control can be useful for continuous tasks such as moving an object across a screen (e.g., a cursor). However, HCI is also full of discrete actions such as button clicks and key presses. This example explores the combination of continuous and discrete inputs to enable hands-free cursor control. 

As technology progresses, a need arises to compute in atypical environments (e.g., driving or walking to work). EMG is a promising input modality in these environments due to its "always-available" nature and subtlety. In this example, a small subset of gestures is leveraged to control a cursor in any ubiquitous computing environment. 

![](Mouse.gif)
<center> <p> Figure 1: An example of using the mouse interface to control a calculator. </p> </center>

# Mouse Design Considerations

## Controls
The controls used for the mouse interface are highlighted below:

| Control| Contraction |
| --- | --- |
| Activate/De-activate Mouse | Finger Gun |
| Move Right | Extension | 
| Move Left | Flexion |
| Move Up | Hand Open | 
| Move Down | Hand Close | 
| Left Click | Finger Tap |

## Velocity Control
Velocity-based control should be used to enable more efficient and effective cursor control. When users contract harder, the cursor will move faster in the specified direction. Alternatively, more precise control can be achieved by eliciting lighter contractions. 

## Wake Words
Since EMG is inherently noisy, and false activations are inevitable, **wake word** should be used. This wake word will enable users to turn on and off the mouse controller. The `finger gun` gesture was selected as previous work suggests that this gesture has the least amount of overlap with activities of daily living. 

# Classifier
While this example won't walk through each step of the online classifier, a couple of noteworthy components are highlighted below. The classifier code can be found in `main_menu.py`.

1. The LS4 feature group is used as it is ideal for devices with low sampling rates (i.e., the Myo).
1. An LDA classifier is used due to its robustness and success within the prosthetics community.
1. Rejection is used to reduce the number of false activations.
1. Velocity-based control is added to enable a more effective mouse interface. 

# Mouse Controller 
To view the code associated with the mouse controller, please view `myo_mouse.py`.

## Dealing with Continuous Inputs 
Moving the cursor was no different than the continuous control schemes used in the previous examples. By leveraging the `pyautogui` library, we moved the cursor by a certain number of pixels for every prediction. In this case, `self.VEL` is the number of pixels and the multiplier is a representation of the contraction intensity. 

```Python
# Deal with continuous inputs 
if self.action == Action.HAND_CLOSE:
    # Move Down
    pyautogui.moveRel(0, self.VEL * multiplier)
elif self.action == Action.HAND_OPEN:
    # Move Up
    pyautogui.moveRel(0, -self.VEL * multiplier)
elif self.action == Action.EXTENSION:
    # Move Right
    pyautogui.moveRel(self.VEL * multiplier, 0)
elif self.action == Action.FLEXION:
    # Move Left
    pyautogui.moveRel(-self.VEL * multiplier, 0)
```

## Dealing with Discrete Inputs 

Maintaining the current state of the controller. 
```Python
class Action(Enum):
    REJECT = -1
    FINGER_GUN = 0
    HAND_CLOSE = 1
    HAND_OPEN = 2
    FINGER_TAP = 3 
    REST = 4
    EXTENSION = 5
    FLEXION = 6 
```

Wake word:
```Python
# Turn on/off mouse 
if self.action == Action.FINGER_GUN:
    if self.steady_state >= 3 and (datetime.datetime.now() - self.button_debounce).total_seconds() > 2:
        self.mouse_active = not self.mouse_active
        self.steady_state = 0
        self.button_debounce = datetime.datetime.now() 
        print("Mouse Active: " + str(self.mouse_active))
```

Checking for button clicks.
```Python
if self.action == Action.FINGER_TAP:
    # Button Click
    if self.steady_state >= 3 and (datetime.datetime.now() - self.button_debounce).total_seconds() > 0.5:
        self.steady_state = 0
        self.button_debounce = datetime.datetime.now() 
        pyautogui.click()       
```