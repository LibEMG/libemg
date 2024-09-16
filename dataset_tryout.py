from libemg.datasets import *
from libemg.feature_extractor import *
from libemg.emg_predictor import EMGClassifier
from libemg.offline_metrics import OfflineMetrics
import pickle 

info = {
    'dataset': [],
    'features': [],
    'model': [],
    'accuracies': [],
    'subject': []
}

for d in get_dataset_list().keys():
    dataset = get_dataset_list()[d]()
    dataset.get_info()
    data = dataset.prepare_data(split=True)
    
    train_data = data['Train']
    test_data = data['Test']
    
    for s in range(0, dataset.num_subjects):
        s_train_dh = train_data.isolate_data('subjects', [s])
        s_test_dh = test_data.isolate_data('subjects', [s])
        train_windows, train_meta = s_train_dh.parse_windows(int(dataset.sampling/1000 * 300), int(dataset.sampling/1000 * 50))
        test_windows, test_meta = s_test_dh.parse_windows(int(dataset.sampling/1000 * 300), int(dataset.sampling/1000 * 50))

        for f_i, feats in enumerate([[['WENG'], {'WENG_fs': dataset.sampling}], [['MAV', 'SSC', 'WL', 'ZC'], {}]]):
            fe = FeatureExtractor()
            train_feats = fe.extract_features(feats[0], train_windows, feats[1])
            test_feats = fe.extract_features(feats[0], test_windows, feats[1])

            model = EMGClassifier(model='LDA')
            ds = {
                'training_features': train_feats,
                'training_labels': train_meta['classes']
            }
            model.fit(ds)
        
            preds, probs = model.run(test_feats)
            om = OfflineMetrics()
            conf_mat = om.get_CONF_MAT(preds, test_meta['classes'])
            print(om.get_CA(test_meta['classes'], preds))

            info['accuracies'].append(om.get_CA(test_meta['classes'], preds))
            info['dataset'].append(d)
            info['features'].append(f_i)
            info['model'].append('LDA')
            info['subject'].append(s)

            # Save info every iteration 
            with open('results.pickle', 'wb') as handle:
                pickle.dump(info, handle, protocol=pickle.HIGHEST_PROTOCOL)
 