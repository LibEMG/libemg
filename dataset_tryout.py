from libemg.datasets import *

accs = evaluate('LDA', 300, 100, feature_list=['MAV','SSC','ZC','WL'], included_datasets=['ContractionIntensity'])
print('\n' + str(accs))