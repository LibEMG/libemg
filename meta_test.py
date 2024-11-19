from libemg.datasets import *

ds = get_dataset_list('REGRESSION')['EMG2POSE']('Meta/emg2pose_data/')
data = ds.prepare_data(subjects=[0], split=True)

x = data['Train'].data 
