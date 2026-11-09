[View Source Code](https://github.com/libemg/LibEMG_MixedReality_Showcase)
![](mr.gif)

One reason for the renewed interest in EMG as a general-purpose input modality stems from its inherent potential for mixed reality. With the continued advancement of mixed reality technology, myoelectric control may become desirable for several reasons due to its subtlety and convenience. In this example, we show how LibEMG can be leveraged to navigate a menu on the HoloLens 2. While this example is not a robust MR application, it's the foundation for future work to explore EMG in MR.

# EMG-Based Control

## Controls
The controls used for menu navigation are highlighted below:

| Control| Contraction |
| --- | --- |
| Scroll Up | Wrist Extension | 
| Scroll Down | Wrist Flexion |
| Open Menu | Hand Open | 
| Press Button | Finger Tap | 
| Do Nothing | Rest |

## Control Scheme
The file associated with the myoelectric control scheme in this example can be found in `emg_control.py`. The only difference compared to the previous examples is that we stream predictions from the classifier over TCP instead of UDP. We do this because the Hololens has better compatibility for TCP network communication. As shown, we are streaming over the network computer's IP and through port 8099.

The online classifier sends its outputs wherever its output writers tell it to. In this example the Python side acts as the TCP server: the Hololens (or the Unity editor) connects to it as a client and reads one predicted class per line (see the Myo Client below). LibEMG's built-in `SocketOutputWriter(..., protocol='TCP')` works the other way around (it connects out to a server that is already listening, and sends `"<value> <timestamp>"` without a newline), so we write a small `OutputWriter` that hosts the server and sends `"<predicted class>\n"`.

```Python
import socket
from libemg.output_writer import OutputWriter, ConsoleOutputWriter

class TCPServerOutputWriter(OutputWriter):
    # Hosts a TCP server and sends each predicted class, one per line, to the client that connects
    def __init__(self, ip, port):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((ip, port))
        self.server.listen()
        self.conn, _ = self.server.accept() # Wait for the Hololens / Unity editor to connect

    def write(self, info):
        self.conn.sendall(f"{info['model_output']}\n".encode('utf-8'))

# Step 5: Create online EMG classifier and start classifying.
IP = 'x.x.x.x' # Replace with computer IP
# IP = '127.0.0.1' # Local host if running in editor
# ConsoleOutputWriter also prints each prediction to the terminal
output_writers = [TCPServerOutputWriter(IP, 8099), ConsoleOutputWriter('model_output')]
self.classifier = OnlineEMGClassifier(o_classifier, WINDOW_SIZE, WINDOW_INCREMENT, self.odh, feature_list, output_writers=output_writers)
self.classifier.run(block=True) # block=True runs the classifier in this process, so the open connection is used directly
```

# Hololens Support 
To develop the Hololens portion of the app, we leveraged the [MRTK](https://learn.microsoft.com/en-us/windows/mixed-reality/mrtk-unity/mrtk2/?view=mrtkunity-2022-05), a toolkit to facilitate the development of MR systems on the Hololens. This gave us access to several predefined assets, such as menus and buttons. We created two scripts in this example: `MenuController.cs` and `MyoReaderClient.cs`. 

## Menu Controller
The menu implemented is simply a scrollable view consisting of ten button objects (from the MRTK). To turn the continuous predictions into more discrete events, the menu controller waits for five consecutive decisions before doing anything. Since the window increment is 20 samples and the sampling rate is 200 Hz, each action is at least 0.5s. Therefore, to open the menu, the user will have to keep their hand open for 0.5s. The goal was to reduce spurious false activations and enable the system to recognize discrete actions (e.g., hand open or wrist swipes) rather than continuous contractions.

```C#
void Update() {
    // First check for 5 consecutive actions
    if (emgReader.consecutive == 5) {
        if (emgReader.control == "4") {
            // Move Up - Flexion
            UpScroll();
        } else if (emgReader.control == "3") {
            // Move Down - Extension
            DownScroll();
        } else if (emgReader.control == "0") {
            // Hand close
            ButtonClick();
        } else if (emgReader.control == "1") {
            // Hand Open - Open.Close Menu
            menu.SetActive(!menu.activeSelf);
        }
        // Updates the button text color based on which is selected
        if (menu.activeSelf) {
            UpdateButtonColors();
        }
        // Reset - This acts as a bit of a debounce.
        emgReader.control = ""; 
        emgReader.consecutive = 0;
    }
}
```
## Myo Client 
To connect LibEMG to the Hololens, we stream predictions over TCP. When testing the application in the Unity editor, this entails one protocol (localhost), and for streaming to the Hololens it is another. This logic, which is split into the `#if UNITY_EDITOR` blocks, means the Myo Client script works in both cases. This script should remain relatively consistent if leveraged for a different use-case.

```C#
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using System.IO;
using UnityEngine.Networking;
using System.Net.Sockets;
using System.Net;
using System.Text;
using System;
using TMPro;

#if !UNITY_EDITOR
using System.Threading.Tasks;
#else 
using System.Threading;
#endif

public class MyoReaderClient : MonoBehaviour {

#if !UNITY_EDITOR
    private Windows.Networking.Sockets.StreamSocket socket;
    private Task socketListenTask;
#endif

    // Listen on port 8099
    private int port = 8099;
    private string host = "127.0.0.1";
    private string ipAddress = "x.x.x.x"; // Replace with computer IP
    private string portUWP = "8099";
    public string control = "Starting!";
    public int consecutive = 0; // Used for discrete control in the menucontroller
    private StreamReader reader;
    private bool connected = false;


#if UNITY_EDITOR
    TcpClient socketClient;
#endif

    void Start () {
#if !UNITY_EDITOR
        ConnectSocketUWP();
#else
        ConnectSocketUnity();
        var readThread = new Thread(new ThreadStart(ListenForDataUnity));
        readThread.IsBackground = true;
        readThread.Start();
#endif
    }
    
    void Update () {
#if !UNITY_EDITOR
        if(socketListenTask == null || socketListenTask.IsCompleted)
        {
            socketListenTask = new Task(async() =>{ListenForDataUWP();});
            socketListenTask.Start();
        }
#endif
    }
    
#if UNITY_EDITOR
    void ConnectSocketUnity()
    {
        IPAddress ipAddress = IPAddress.Parse(host);

        socketClient = new TcpClient();
        try
        {
            socketClient.Connect(ipAddress, port);
        }

        catch
        {
            Debug.Log("error when connecting to server socket");
        }
    }
#else
    private async void ConnectSocketUWP()
    {
        try
        {
            socket = new Windows.Networking.Sockets.StreamSocket();
            Windows.Networking.HostName serverHost = new Windows.Networking.HostName(ipAddress);
            await socket.ConnectAsync(serverHost, portUWP);
            Stream streamIn = socket.InputStream.AsStreamForRead();
            reader = new StreamReader(streamIn, Encoding.UTF8);
            connected = true;
            control = "Connected";
        }
        catch (Exception e)
        {
            control = "Connection Error";
        }
    }
#endif

#if UNITY_EDITOR
    void ListenForDataUnity()
    {
        int data;
        while(true){
            byte[] bytes = new byte[socketClient.ReceiveBufferSize];
            NetworkStream stream = socketClient.GetStream();
            data = stream.Read(bytes, 0, socketClient.ReceiveBufferSize);
            string tControl = Encoding.UTF8.GetString(bytes, 0, data).Trim();
            // Keep track of consecutive agreements
            if (tControl == control) {
                consecutive += 1; 
            } else {
                consecutive = 0;
            }
            control = tControl;

        }
    }
#else
    private void ListenForDataUWP()
    {
        try {
            // Keep track of consecutive predictions
            string tControl = reader.ReadLine().Trim();
            if (tControl == control) {
                consecutive += 1; 
            } else {
                consecutive = 0;
            }
            control = tControl;
        } catch (Exception e) {
            //Do nothing
        }
    }
#endif
}
```