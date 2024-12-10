import numpy as np
import random
from scipy.interpolate import interp1d

class DataAugmenter:
    """
    Data augmentation class.
    """
    def __init__(self, augmentations, aug_chance=0.5):
        self.augmentations = augmentations
        self.aug_chance = aug_chance

    def apply_augmentations(self, data, data2=None, params=None):
        for a in self.augmentations:
            aug = self.get_augmentation_list()[a]
            if random.random() < self.aug_chance:
                if a == 'MIXUP':
                    if params:
                        data = aug(data, data2, params)
                    else: 
                        data = aug(data, data2)
                else:
                    if params:
                        data = aug(data, params)
                    else:
                        data = aug(data)
        return data 

    def get_augmentation_list(self):
        """Gets a list of all available augmentations.
        
        Returns
        ----------
        list
            A list of all available augmentations.
        """
        return {
            'GNOISE': self.augGNOISE,           # Add Gaussian noise
            'CS': self.augCS,                   # Channel swapping
            'CR': self.augCR,                   # Channel rotation
            'CROP': self.augCROP,               # Randomly crop out part of signal
            'MAG': self.augMAG,                 # Scale the signal by some factor
            'WARP': self.augWARP,               # Stretch/shrink the signal by some factor
            'POL': self.augPOL,                 # Flips the polarity of the signal
            'ZERO': self.augZERO,               # Randomly zeros values
            'TIMESHIFT': self.augTIMESHIFT,     # Shifts the signal (based on time)
            'NOISE': self.augNOISE,             # Generic noise
            'CUTOUT': self.augCUTOUT,           # Zero a portion of the signal
            'MIXUP': self.augMIXUP,             # Combine multiple samples
            'NONE': self.augNONE,
        }

    def augGNOISE(self, data, max_mag=1):
        noise_factor = random.random()
        gaussian_noise = np.random.normal(np.mean(data, axis=0) * noise_factor * max_mag, np.std(data, axis=0) * noise_factor * max_mag, data.shape)
        return data + gaussian_noise

    def augCS(self, data):
        return data[:, np.random.permutation(data.shape[1])]
    
    def augCR(self, data):
        shift = np.random.randint(-1, 2)
        return np.roll(data, shift, axis=1)

    def augCROP(self, data, crop_percent=0.2):
        crop_percent = 1 - crop_percent
        crop_length = int(data.shape[0] * crop_percent)
        start_idx = np.random.randint(0, data.shape[0] - crop_length)
        end_idx = start_idx + crop_length
        return data[start_idx:end_idx, :]
    
    def augMAG(self, data, mag=0.5):
        min_mag = mag  
        max_mag = 1 + (1 - mag)
        mag = np.random.uniform(min_mag, max_mag)
        return data * mag
    
    def augWARP(self, data, warp_factor=0.5):
        time_warped_data = []
        warp_amount = np.random.uniform(-warp_factor, warp_factor)
        for channel in data.T:
            # Extract the scalar value from warp_factor
            f = interp1d(np.linspace(0, len(channel)-1, len(channel)), channel)
            time_warped_channel = f(np.linspace(0, len(channel)-1, int(len(channel)*(1+warp_amount))))
            time_warped_data.append(time_warped_channel)
        return np.array(time_warped_data).T

    def augPOL(self, data):
        return data * -1
    
    def augZERO(self, data, percent=0.1):
        vals = np.random.choice([0, 1], size=(len(data),), p=[percent, 1-percent])
        return (data.T * vals).T
    
    def augTIMESHIFT(self, data, shift_range=0.2):
        shift_amount = int(np.random.uniform(-shift_range, shift_range) * data.shape[0])
        shifted_data = np.roll(data, shift_amount, axis=0)
        return shifted_data
    
    def augNOISE(self, data, noise_level=1):
        noise_level = noise_level * np.std(data)
        noise = np.random.randn(*data.shape) * noise_level
        return data + noise
    
    def augCUTOUT(self, data, drop_prob=0.2):
        num_channels = data.shape[1]
        channels_to_zero = np.random.choice(num_channels, size=int(num_channels * drop_prob), replace=False)
        data[:, channels_to_zero] = 0

        return data
    
    def augMIXUP(self, data1, data2, alpha=0.8):
        lambda_ = np.random.beta(alpha, alpha)
        mixed_data = lambda_ * data1 + (1 - lambda_) * data2
        return mixed_data
    
    def augNONE(self, data):
        return data