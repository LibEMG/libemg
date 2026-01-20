import torch
import torch.nn as nn
import numpy as np
import os
import urllib.request

MODEL_URL = "https://github.com/eeddy/DiscreteMCI/raw/main/Other/Discrete.model"
DEFAULT_MODEL_PATH = os.path.join("./Discrete.model")


class DiscreteClassifier(nn.Module):
    """Architecture definition required for torch.load to deserialize the model."""
    def __init__(self, emg_size, file_name=None, temporal_hidden_size=128, temporal_layers=3,
                 mlp_layers=[128, 64, 32], n_classes=6, type='GRU',
                 conv_kernel_sizes=[3, 3, 3], conv_out_channels=[16, 32, 64]):
        super().__init__()

        self.file_name = file_name
        self.log = {}
        self.min_loss = 0

        dropout = 0.2

        self.conv_layers = nn.ModuleList()
        in_channels = emg_size[1]
        for i in range(len(conv_out_channels)):
            self.conv_layers.append(nn.Conv1d(in_channels=in_channels, out_channels=conv_out_channels[i],
                                              kernel_size=conv_kernel_sizes[i], padding='same'))
            self.conv_layers.append(nn.BatchNorm1d(conv_out_channels[i]))
            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(nn.MaxPool1d(kernel_size=2))
            self.conv_layers.append(nn.Dropout(dropout))
            in_channels = conv_out_channels[i]

        spoof_emg_input = torch.zeros((1, *emg_size))
        conv_out = self.forward_conv(spoof_emg_input)
        conv_out_size = conv_out.shape[-1]

        self.temporal = nn.GRU(conv_out_size, temporal_hidden_size, num_layers=temporal_layers,
                                   batch_first=True, dropout=dropout)

        emg_output_shape = self.forward_temporal(conv_out).shape[-1]

        self.initial_layer = nn.Linear(emg_output_shape, mlp_layers[0])
        self.layer1 = nn.Linear(mlp_layers[0], mlp_layers[1])
        self.layer2 = nn.Linear(mlp_layers[1], mlp_layers[2])
        self.output_layer = nn.Linear(mlp_layers[-1], n_classes)
        self.relu = nn.ReLU()

    def forward_conv(self, x):
        batch_size, seq_len, channels, samples = x.shape
        x = x.view(batch_size * seq_len, channels, samples)
        for layer in self.conv_layers:
            x = layer(x)
        _, channels_out, samples_out = x.shape
        x = x.view(batch_size, seq_len, channels_out * samples_out)
        return x

    def forward_temporal(self, emg, lengths=None):
        out, _ = self.temporal(emg)
        if lengths is not None:
            out = torch.stack([s[lengths[i]-1] for i, s in enumerate(out)])
        else:
            out = out[:, -1, :]
        return out

    def forward_mlp(self, x):
        out = self.initial_layer(x)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.relu(out)
        out = self.output_layer(out)
        return out

    def forward_once(self, emg, emg_len=None):
        out = self.forward_conv(emg)
        out = self.forward_temporal(out, emg_len)
        out = self.forward_mlp(out)
        return out

    def predict(self, x, device='cpu'):
        self.to(device)
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        preds = self.forward_once(x.to(device))
        return np.array([p.argmax().item() for p in preds])

    def predict_proba(self, x, device='cpu'):
        self.to(device)
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        logits = self.forward_once(x.to(device))
        probs = torch.softmax(logits, dim=-1)
        return probs.detach().cpu().numpy()


class MyoCrossUserPretrained:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = DEFAULT_MODEL_PATH
        self.model_path = model_path
        self.model = None
        self._ensure_model_downloaded()
        self._load_model()

        self.args = {
            'window_size': 10, 'window_increment': 5, 'null_label': 0, 'feature_list': None, 'template_size': 250, 'min_template_size': 150, 'gesture_mapping': ['Nothing', 'Close', 'Flexion', 'Extension', 'Open', 'Pinch'], 'buffer_size': 5,
        }

        print("This model has defined args (self.args) which need to be passed into the OnlineDiscreteClassifier.")

    def _ensure_model_downloaded(self):
        if os.path.exists(self.model_path):
            return

        model_dir = os.path.dirname(self.model_path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)

        print(f"Downloading model to {self.model_path}...")
        urllib.request.urlretrieve(MODEL_URL, self.model_path)
        print("Download complete.")

    def _load_model(self):
        import sys
        # Register DiscreteClassifier in sys.modules so torch.load can find it
        # (the saved model was pickled with DiscreteClassifier as the module name)
        sys.modules['DiscreteClassifier'] = sys.modules[__name__]

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = torch.load(self.model_path, map_location=device, weights_only=False)
        self.model.eval()

    def predict(self, x, device='cpu'):
        return self.model.predict(x, device=device)

    def predict_proba(self, x, device='cpu'):
        return self.model.predict_proba(x, device=device)