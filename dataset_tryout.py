from libemg.datasets import *

accs = evaluate('LDA', 300, 100, feature_list=['MAV','SSC','ZC','WL'], included_datasets=['FougnerLP'], save_dir='')
print('\n' + str(accs))