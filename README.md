<h1 align="center">EMG Feature Extraction Toolkit</h1>

The goal of this toolkit is to provide an easy to use API for EMG feature extraction. This is an open sourced toolkit developed at the University of New Brunswick by lab members at the Institute of Biomedical Engineering.

Authors: Ethan Eddy, Evan Campbell and Erik Scheme

__Table of Contents__

* [Features](#features)
* [FeatureSets](#featuresets)
* [Documentation](#documentation)
* [Examples](#examples)
* [References](#references)

## Features

Let $x_{i}$ represents the signal in segment i and $N$ denotes the number of samples in the timeseries.

#### **Mean Absolute Value (MAV)**
The average of the absolute value of the EMG signal. This is one of the most commonly used features for EMG.
```math
    \text{MAV} = \sum_{i=1}^{N} |x_{i}|
```

#### **Zero Crossings (ZC)**
The number of times that the amplitude of the EMG signal crosses a zero amplitude threshold. The goal of this feature is to avoid low-level voltage fluctuations and background noise. 
```math
    \text{ZC} = \sum_{i=1}^{N-1}[\text{sgn}(x_{i} \times x_{i+1}) \cap |x_{i} - x_{i+1}| \ge \text{threshold}] 
```
```math
    \text{sgn(}x\text{)} = \left\{\begin{array}{lr}
        1, & \text{if } x \ge \text{threshold} \\
        0, & \text{otherwise }
        \end{array}\right\}
```

#### **Slope Sign Change (SSC)**
The number of times that the slope of the EMG signal changes (i.e., the number of times that it changes between positive and negative). This is used to help eliminate background noise.
```math
    \text{SSC} = \sum_{i=2}^{N-1}[f[(x_{i} - x_{i-1}) \times (x_{i} - x_{i+1})]] \\
```
```math
    f(x) = \left\{\begin{array}{lr}
        1, & \text{if } x \ge \text{threshold} \\
        0, & \text{otherwise }
        \end{array}\right\}
```

#### **Waveform Length (WL)**
The cumulative length of the EMG waveform over the passed in window. This feature is used to measure the overall complexity of the signal (i.e., higher WL means more complex).
```math
    \text{WL} = \sum_{i=1}^{N-1}|x_{i+1} - x_{i}|
```

#### **L-score (LS)**
A feature that uses the Legendre Shift Polynomial.
```math
    \text{LS} = \sum_{i=1}^{N} \text{LSP} * B
```
**(?)Should probably elaborate here, but I don't know a condense shorthand for the math**

#### **Maximum Fractal Length (MFL)**
A nonlinear information feature for the EMG waveform that is relatively invariant to contraction intensity.
```math
    \text{MFL} = log_{10} \sum_{i=1}^{N-1}  | x_{i+1} - x_{i} |
```

#### **? (MSR)**

#### **Willison Amplitude (WAMP)**
The number of times that there is a difference between amplitudes of two seperate samples exceeding a pre-defined threshold. This feature is related to the firing of motor unit action potentials and muscle contraction force.
```math
    \text{WAMP} = \sum_{i=1}^{N-1}[f(|x_{n}-x_{n+1})] \\
```
```math    
    f(x) = \left\{\begin{array}{lr}
        1, & \text{if } x \ge \text{threshold} \\
        0, & \text{otherwise }
        \end{array}\right\}
```

#### **Root Mean Square (RMS)**
```math
    \text{RMS} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}x_{i}^{2}}
```

#### **Integral of Absolute Value (IAV)**
The integral of the absolute value.
```math
    \text{IAV} = \sum_{i=i}^{N} |x_{i}|
```


#### **Absolute Square Average Difference Value (DASDV)**

```math
    \text{DASDV}=|\sqrt{\sum_{i=1}^{N-1} (x_{i+1}^2 - x_{i}^2)  ) /(N-1)} |
```

#### **Variance (VAR)**
```math
    \text{VAR} = \frac{1}{N-1}\sum_{i=1}^{N} x_{i}^{2}
```

#### **First Temporal Moment (M0)**
A commonly-used feature part of the TDPSD feature set.
```math
    \text{M0} = log( |\frac{\sqrt{\sum_{i=1}^{N} x_{i}^2}^0.1}{0.1} | )
```

#### **Second Temporal Moment (M2)**
A commonly-used feature part  of the TDPSD feature set.
```math
    \text{M2} = log ( | \frac{\sqrt{\sum_{i=1}^{N-1} (x_{i+1} - x_{i})^2}^0.1}{0.1} | )
```

#### **Fourth Temporal Moment (M4)**
A commonly-used feature part of the TDPSD feature set.
```math
    \text{M4} = log ( | \frac{\sqrt{\sum_{i=1}^{N-2} (x_{i+2} + x_{i})^2}^0.1}{0.1} | )
```

#### **Sparsness (SPARSI)**
A commonly-used feature part of the TDPSD feature set.
```math
    \text{SPARSI} = log( |\frac{\text{M0}}{\sqrt{ | \text{M0}*\text{M4} | }} |)
```

#### **Irregularity Factor (IRF)**
A commonly-used feature part of the TDPSD feature set.
```math
    \text{SPARSI} = log ( | \frac{\text{M2}}{\sqrt{ | \text{M0}*\text{M4} | }} | )
```

#### **Waveform Length Factor (WLF)**
A commonly-used feature part of the TDPSD feature set.
```math
    \text{WLF} = log ( | \sum_{i=1}{N-1} | x_{i+1} - x_{i} | - \sum_{i=1}{N-2} | x_{i+2} + x_{i} |  | )
```


#### **Autoregressive Coefficients (AR)**
For order r, the autoregressive coefficient can be calculated by:
```math
    \text{AR}_{r} = \sum_{i=1}^{N-r} x_{i+r}*x_{i}
```

#### **?Correlation Coefficient (CC)**


#### **? (LD)**
```math
    \text{LD} = e^{\sum_{i=1}{N} log(|x_{i}|) /N}
```

#### **Mean Absolute Value First Difference (MAVFD)**
Gets the mean absolute value of the slope of the signal.
```math
    \text{MAVFD} = \sum_{i=1}^{N-1}  (x_{i+1}-x_{i})  /(N-1)
```


#### **Mean Absolute Value Slope (MAVSLP)**
Get the slope between p sections within a window. Here demonstrated for p=2.
```math
    \text{MAVSLP}_{p} = \sum{i=N/p + 1}^{N} |x_{i}| / (N/p) - \sum_{j=1}^{N/p} |x_{j}|/(N/p) 
```

#### **? (MDF)**

#### **? (MNF)**

#### **? (MNP)**

#### **? (MPK)**

#### **? (SAMPEN)**

#### **? (SKEW)**

#### **? (KURT)**

## FeatureSets
#### **Hudgin's Time Domain (HTD)** 
1. Mean Absolute Value (MAV)
2. Zero Crossings (ZC)
3. Slope Sign Change (SSC)
4. Waveform Length (WL)

#### **Phinyomark's Time Domain 4 (TD4)**
1. L-Score (LS)
2. Maximum Fractal Length (MFL)
3. ? (MSR)
4. Willison Amplitude (WAMP)

#### **Phinyomark's Time Domain 9 (TD9)**
1. TD4 Features 
2. Zero Crossings (ZC)
3. Root Mean Square (RMS)
4. Integral of the Absolute Value (IAV) 
5. Absolute Square Average Difference Value (DASDV)
6. Variance (VAR)

## Documentation
The documentation can currently be found in the `docs/` directory in the [GitHub Repository](https://github.com/eeddy/emg-feature-extraction).

## Examples
### Feature Extraction
```python
from unb_emg_toolbox.feature_extractor import FeatureExtractor as featureextractor
import numpy as np

def main():
    test_file = "data.txt" # NxM where N is samples and M is channels
    data = np.loadtxt(test_file, delimiter=',')
    num_channels = data.shape[1]

    feature_extractor = fe(num_channels=num_channels)
    windows = feature_extractor.get_windows(data, window_size=200, window_increment=100)

    # Extract Individual Features:
    features = feature_extractor.extract_features(['MAV', 'ZC'], windows)

    # Extract Feature Groups:
    features = fe.extract_feature_group('TD4', windows)
```

## References
```
Angkoon Phinyomark, Pornchai Phukpattaranont, Chusak Limsakul,
Feature reduction and selection for EMG signal classification,
Expert Systems with Applications,
Volume 39, Issue 8,
2012,
Pages 7420-7431,
ISSN 0957-4174,
https://doi.org/10.1016/j.eswa.2012.01.102.
(https://www.sciencedirect.com/science/article/pii/S0957417412001200)
```
```
Phinyomark, A.; N. Khushaba, R.; Scheme, E. 
Feature Extraction and Selection for Myoelectric Control Based on Wearable EMG Sensors. 
Sensors 2018, 18, 1615. 
https://doi.org/10.3390/s18051615
```