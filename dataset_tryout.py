from libemg.datasets import *
from libemg.feature_extractor import *
from libemg.emg_predictor import EMGClassifier
from libemg.offline_metrics import OfflineMetrics
 
print(get_dataset_list())

dataset = get_dataset_list()['MyoDisCo'](cross_day=True)
dataset.get_info()
data = dataset.prepare_data()
 
train_data = data['Train']
test_data = data['Test']

print("Loaded Data")
 
accuracies = []
for s in range(0, dataset.num_subjects):
    print("Subject: " + str(s))
    s_train_dh = train_data.isolate_data('subjects', [s])
    s_test_dh = test_data.isolate_data('subjects', [s])
    train_windows, train_meta = s_train_dh.parse_windows(30, 5)
    test_windows, test_meta = s_test_dh.parse_windows(30, 5)
 
    fe = FeatureExtractor()
    train_feats = fe.extract_features(['WENG'], train_windows)
    test_feats = fe.extract_features(['WENG'], test_windows)

    # fe.visualize_feature_space(train_feats, 'PCA', train_meta['classes'])
 
    model = EMGClassifier(model='LDA')
    ds = {
        'training_features': train_feats,
        'training_labels': train_meta['classes']
    }
    model.fit(ds)
 
    preds, probs = model.run(test_feats)
    om = OfflineMetrics()
    accuracies.append(om.get_CA(test_meta['classes'], preds))
    conf_mat = om.get_CONF_MAT(preds, test_meta['classes'])
    # om.visualize_conf_matrix(conf_mat)
    print(om.get_CA(test_meta['classes'], preds))
 
print('CA: ' + str(np.mean(accuracies)) + ' +/- ' + str(np.std(accuracies)))
 
 