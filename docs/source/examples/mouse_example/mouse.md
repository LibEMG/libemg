[View Source Code](https://github.com/libemg/LibEMG_Cursor_Showcase)

**Note, this controller is by no means a robust mouse interface and is is instead a simple demonstration highlighting the possibilities of the library.**

The main control scheme leveraged by the prosthetics community involves continuously predicting user intent to micro-adjust a prosthesis. These predictions occur for relatively small windows of data (in the order of milliseconds) to enable responsive controllers. There are many applications where this continuous control could be useful in HCI, including cursor control. In this example, we use proportional (i.e., velocity) control to create a cursor interface for a desktop computer. We then use it as an alternative way to play a fun online game [snake.io](https://snake.io/).

![](Snake-Demo.gif)
<center> <p> Figure 1: An example of using the mouse interface to play an online game. </p> </center>

# Mouse Design Considerations

## Controls
The controls used for the mouse interface are highlighted below:

| Control| Contraction |
| --- | --- |
| Move Right | Wrist Extension | 
| Move Left | Wrist Flexion |
| Move Up | Hand Open | 
| Move Down | Hand Close | 
| Do Nothing | Rest |

## Velocity Control
Velocity-based control should be used to enable more efficient and effective cursor control. Therefore, in this example, when users contract harder, the cursor will move faster in the specified direction. Alternatively, more precise control can be achieved by eliciting lighter contractions. 

# Classifier
While this example won't walk through each step of the online classifier, a couple of noteworthy components are highlighted below. The classifier code can be found in `main_menu.py`.

1. The LS4 feature group is used as it is ideal for devices with low sampling rates (i.e., the Myo).
1. An LDA classifier is used due to its robustness and success within the prosthetics community.
1. Rejection is used to reduce the number of false activations.
1. Velocity-based control is added to enable a more effective mouse interface. 

# Mouse Controller 
Moving the cursor was no different than the continuous control schemes used in the previous examples. By leveraging the `pyautogui` library, we moved the cursor by a certain number of pixels for every prediction. In this case, `self.VEL` is the number of pixels and the multiplier is a representation of the contraction intensity. To view the code associated with the mouse controller, please view `myo_mouse.py`.

```Python
if input_class == 0:
    # Move Down
    pyautogui.moveRel(0, self.VEL * multiplier)
elif input_class == 1:
    # Move Up
    pyautogui.moveRel(0, -self.VEL * multiplier)
elif input_class == 3:
    # Move Right
    pyautogui.moveRel(self.VEL * multiplier, 0)
elif input_class == 4:
    # Move Left
    pyautogui.moveRel(-self.VEL * multiplier, 0)
```