<h1 align="center">EMG Toolkit</h1>

The goal of this toolkit is to provide an easy to use API for EMG feature extraction. This is an open sourced toolkit developed at the [University of New Brunswick](https://www.unb.ca/) by the [Institute of Biomedical Engineering Lab](https://www.unb.ca/ibme/).

**Authors**: Ethan Eddy, Evan Campbell, and Erik Scheme

__Table of Contents__

- [EMG Features](#emg-features)
- [Documentation](#documentation)
- [Examples](#examples)
  - [Feature Extraction](#feature-extraction)
- [References](#references)

## EMG Features
All available features can be found in `docs/features.md` in the [GitHub Repository](https://github.com/eeddy/emg-feature-extraction).

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