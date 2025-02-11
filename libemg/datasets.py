from libemg._datasets._3DC import _3DCDataset
from libemg._datasets.one_subject_myo import OneSubjectMyoDataset
from libemg._datasets.one_subject_emager import OneSubjectEMaGerDataset
from libemg._datasets.emg_epn612 import EMGEPN_UserDependent, EMGEPN_UserIndependent
from libemg._datasets.ciil import CIIL_MinimalData, CIIL_ElectrodeShift, CIIL_WeaklySupervised
from libemg._datasets.grab_myo import GRABMyoBaseline, GRABMyoCrossDay
from libemg._datasets.continous_transitions import ContinuousTransitions
from libemg._datasets.nina_pro import NinaproDB2, NinaproDB8
from libemg._datasets.user_compliance import UserComplianceDataset
from libemg._datasets.fors_emg import FORSEMG
from libemg._datasets.radmand_lp import RadmandLP
from libemg._datasets.fougner_lp import FougnerLP
from libemg._datasets.intensity import ContractionIntensity
from libemg._datasets.hyser import Hyser1DOF, HyserNDOF, HyserRandom, HyserPR, HyserMVC # HyserMVC is not used in this script but is imported so it's public in libemg API
from libemg._datasets.kaufmann_md import KaufmannMD
from libemg._datasets.tmr_shirleyryanabilitylab import TMR_Post, TMR_Pre
from libemg.feature_extractor import FeatureExtractor
from libemg.emg_predictor import EMGClassifier, EMGRegressor
from libemg.offline_metrics import OfflineMetrics
from libemg.filtering import Filter
from libemg._datasets.emg2pose import EMG2POSEUD, EMG2POSECU
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np

def get_dataset_list(type='CLASSIFICATION', cross_user=False):
    """Gets a list of all available datasets.

    Parameters
    ----------
    type: str (default='CLASSIFICATION')
        The type of datasets to return. Valid Options: 'CLASSIFICATION', 'REGRESSION', 'WEAKLYSUPERVISED', and 'ALL'.
    cross_user: bool (default=False)

    
    Returns
    ----------
    dictionary
        A dictionary with the all available datasets and their respective classes.
    """
    type = type.upper()
    if type not in ['CLASSIFICATION', 'REGRESSION', 'WEAKLYSUPERVISED', 'CROSSUSER', 'ALL']:
        print('Valid Options for type parameter: \'CLASSIFICATION\', \'REGRESSION\', or \'ALL\'.')
        return {}
    
    cross_user_classification = {
        'EMGEPN612': EMGEPN_UserIndependent,
    }

    cross_user_regression = {
        'EMG2POSE': EMG2POSECU,
    }
    
    classification = {
        'OneSubjectMyo': OneSubjectMyoDataset,
        '3DC': _3DCDataset,
        'MinimalTrainingData': CIIL_MinimalData,
        'ElectrodeShift': CIIL_ElectrodeShift,
        'GRABMyoBaseline': GRABMyoBaseline,
        'GRABMyoCrossDay': GRABMyoCrossDay,
        'ContinuousTransitions': ContinuousTransitions,
        'NinaProDB2': NinaproDB2,
        'FORS-EMG': FORSEMG,
        'EMGEPN612': EMGEPN_UserDependent,
        'ContractionIntensity': ContractionIntensity,
        'RadmandLP': RadmandLP,
        'FougnerLP': FougnerLP,
        'KaufmannMD': KaufmannMD,
        'TMR_Post' : TMR_Post,
        'TMR_Pre': TMR_Pre,
        'HyserPR': HyserPR,
    }

    regression = {
        'OneSubjectEMaGer': OneSubjectEMaGerDataset,
        'NinaProDB8': NinaproDB8,
        'Hyser1DOF': Hyser1DOF,
        'HyserNDOF': HyserNDOF, 
        'HyserRandom': HyserRandom,
        'UserCompliance': UserComplianceDataset,
        'EMG2POSE': EMG2POSEUD
    }

    weaklysupervised = {
        'WeaklySupervised': CIIL_WeaklySupervised
    }
    
    if type == 'CLASSIFICATION':
        if cross_user:
            return cross_user_classification
        return classification
    elif type == 'REGRESSION':
        if cross_user:
            return cross_user_regression
        return regression 
    elif type == "WEAKLYSUPERVISED":
        return weaklysupervised
    else:
        # Concatenate all datasets
        classification.update(regression)
        classification.update(weaklysupervised)
        return classification
    
def get_dataset_info(dataset):
    """Prints out the information about a certain dataset. 
    
    Parameters
    ----------
    dataset: string
        The name of the dataset you want the information of.
    """
    if dataset in get_dataset_list():
        get_dataset_list()[dataset]().get_info()
    else:
        print("ERROR: Invalid dataset name")

def evaluate(model, window_size, window_inc, feature_list=['MAV'], feature_dic={}, included_datasets=['OneSubjectMyo', '3DC'], output_file='out.pkl', regression=False, metrics=['CA'], normalize_data=False, normalize_features=False):
    """Evaluates an algorithm against all included datasets.
    
    Parameters
    ----------
    window_size: int
        The window size (**in ms**). 
    window_inc: int
        The window increment (**in ms**). 
    feature_list: list (default=['MAV'])
        A list of features. Pass in None for CNN.
    feature_dic: dic (default={})
        A dictionary of parameters for the passed in features.
    included_datasets: list (str) or list (DataSets)
        The name of the datasets you want to evaluate your model on. Either pass in strings (e.g., '3DC') for names or the dataset objects (e.g., _3DCDataset()). 
    output_file: string (default='out.pkl')
        The name of the directory you want to incrementally save the results to (it will be a pickle file).
    regression: boolean (default=False)
        If True, will create an EMGRegressor object. Otherwise creates an EMGClassifier object. 
    metrics: list (default=['CA']/['MSE'])
        The metrics to extract from each dataset.
    normalize_data: boolean (default=False)
        If True, the data will be normalized.
    normalize_features: boolean (default=False)
        If True, features will get normalized.
    Returns
    ----------
    dictionary
        A dictionary with a set of accuracies for different datasets
    """

    # -------------- Setup -------------------
    if metrics == ['CA'] and regression:
        metrics = ['MSE']

    metadata_operations = None 
    label_val = 'classes'
    if regression:
        metadata_operations = {'labels': 'last_sample'}
        label_val = 'labels'

    om = OfflineMetrics()

    # --------------- Run -----------------
    accuracies = {}
    for d_i, d in enumerate(included_datasets):
        print(f"Evaluating {d} dataset...")
        if isinstance(d, str):
            dataset = get_dataset_list('ALL')[d]()
        else:
            dataset = d

        # Get feature dic 
        if isinstance(feature_dic, list):
            f_dic = feature_dic[d_i]
        else:
            f_dic = feature_dic
        
        accs = []
        for s_i in range(0, dataset.num_subjects):
            data = dataset.prepare_data(split=True, subjects=[s_i])

            if data == None:
                print('Skipping Subject... No data found.')
                continue

            s_train_dh = data['Train']
            s_test_dh = data['Test']

            print(str(s_i) + '/' + str(dataset.num_subjects) + ' completed.')

            # Normalize Data
            if normalize_data:
                filter = Filter(dataset.sampling)
                filter.install_filters({'name': 'standardize', 'data': s_train_dh})
                filter.filter(s_train_dh)
                filter.filter(s_test_dh)

            train_windows, train_meta = s_train_dh.parse_windows(int(dataset.sampling/1000 * window_size), int(dataset.sampling/1000 * window_inc), metadata_operations=metadata_operations)
            test_windows, test_meta = s_test_dh.parse_windows(int(dataset.sampling/1000 * window_size), int(dataset.sampling/1000 * window_inc), metadata_operations=metadata_operations)

            if feature_list is not None:
                fe = FeatureExtractor()
                if normalize_features:
                    train_feats, scaler = fe.extract_features(feature_list, train_windows, feature_dic=f_dic, normalize=True, fix_feature_errors=True)
                    test_feats, _ = fe.extract_features(feature_list, test_windows, feature_dic=f_dic, normalize=True, normalizer=scaler, fix_feature_errors=True)      
                else:
                    train_feats = fe.extract_features(feature_list, train_windows, feature_dic=f_dic, fix_feature_errors=True)
                    test_feats = fe.extract_features(feature_list, test_windows, feature_dic=f_dic, fix_feature_errors=True)     
            else:
                train_feats = train_windows
                test_feats = test_windows

            ds = {
                'training_features': train_feats,
                'training_labels': train_meta[label_val]
            }

            if not regression:
                clf = EMGClassifier(model)
            else:
                clf = EMGRegressor(model)
            clf.fit(ds)
            
            if regression:
                preds = clf.run(test_feats)
            else:
                preds, _ = clf.run(test_feats)
                
            metrics = om.extract_offline_metrics(metrics, test_meta[label_val], preds)
            accs.append(metrics)
                
            print(metrics)    
        accuracies[str(d)] = accs

        with open(output_file, 'wb') as handle:
            pickle.dump(accuracies, handle, protocol=pickle.HIGHEST_PROTOCOL)   


def evaluate_crossuser(model, window_size, window_inc, feature_list=['MAV'], feature_dic={}, included_datasets=['EMGEPN612'], output_file='out_cross.pkl', regression=False, metrics=['CA'], normalize_data=False, normalize_features=False, memory_efficient=False):
    """Evaluates an algorithm against all the cross-user datasets.
    
    Parameters
    ----------
    window_size: int
        The window size (**in ms**). 
    window_inc: int
        The window increment (**in ms**). 
    feature_list: list (default=['MAV'])
        A list of features.
    feature_dic: dic or list (default={})
        A dictionary or list of dictionaries of parameters for the passed in features.
    included_datasets: list (str) or list (DataSets)
        The name of the datasets you want to evaluate your model on. Either pass in strings (e.g., '3DC') for names or the dataset objects (e.g., _3DCDataset()). 
    output_file: string (default='out.pkl')
        The name of the directory you want to incrementally save the results to (it will be a pickle file).
    regression: boolean (default=False)
        If True, will create an EMGRegressor object. Otherwise creates an EMGClassifier object. 
    metrics: list (default=['CA'])
        The metrics to extract from each dataset.
    normalize_data: boolean (default=False)
        If True, the data will be normalized.
    normalize_features: boolean (default=False)
        If True, features will get normalized.
    memory_efficient: boolean (default=false)
        If True, features will be extracted in the prepare method. You wont have access to the raw data and normalization won't be possible. 
    Returns
    ----------
    dictionary
        A dictionary with a set of accuracies for different datasets
    """
    # -------------- Setup -------------------
    if metrics == ['CA'] and regression:
        metrics = ['MSE']

    metadata_operations = None 
    label_val = 'classes'
    if regression:
        metadata_operations = {'labels': 'last_sample'}
        label_val = 'labels'

    om = OfflineMetrics()
    fe = FeatureExtractor()

    # --------------- Run -----------------
    accuracies = {}
    for d_i, d in enumerate(included_datasets):
        print(f"Evaluating {d} dataset...")
        if isinstance(d, str):
            if regression:
                dataset = get_dataset_list('REGRESSION', True)[d]()
            else:
                dataset = get_dataset_list('CLASSIFICATION', True)[d]()
        else:
            dataset = d

        # Get feature dic 
        if isinstance(feature_dic, list):
            f_dic = feature_dic[d_i]
        else:
            f_dic = feature_dic
        
        if memory_efficient:
            data = dataset.prepare_data(split=True, feature_list=feature_list, feature_dic=f_dic, window_size=int(dataset.sampling/1000 * window_size), window_inc=int(dataset.sampling/1000 * window_inc))
        else:
            data = dataset.prepare_data(split=True)
        
        train_data = data['Train']
        test_data = data['Test']

        # Normalize Data
        if normalize_data and not memory_efficient:
            filter = Filter(dataset.sampling)
            filter.install_filters({'name': 'standardize', 'data': train_data})
            filter.filter(train_data)
            filter.filter(test_data)

        if memory_efficient:
            train_feats = np.vstack(train_data.data)
            train_labels = np.vstack(getattr(train_data, label_val)).squeeze()
            if normalize_features:
                normalizer = StandardScaler()
                train_feats = normalizer.fit_transform(train_feats)
        else:
            train_windows, train_meta = train_data.parse_windows(int(dataset.sampling/1000 * window_size), int(dataset.sampling/1000 * window_inc), metadata_operations=metadata_operations)
            if normalize_features:
                train_feats, normalizer = fe.extract_features(feature_list, train_windows, feature_dic=f_dic, normalize=True, fix_feature_errors=True)
            else:
                train_feats = fe.extract_features(feature_list, train_windows, feature_dic=f_dic, fix_feature_errors=True)
            del train_windows
            train_labels = train_meta[label_val]

        ds = {
            'training_features': train_feats,
            'training_labels': train_labels
        }
        
        if not regression:
            clf = EMGClassifier(model)
        else:
            clf = EMGRegressor(model)
        clf.fit(ds)

        del train_feats
        del ds

        unique_subjects = np.unique(np.hstack([t.flatten() for t in test_data.subjects]))
        
        accs = []
        for s_i, s in enumerate(unique_subjects):
            print(str(s_i) + '/' + str(len(unique_subjects)) + ' completed.')
            s_test_dh = test_data.isolate_data('subjects', [s])

            if memory_efficient:
                test_feats = np.vstack(s_test_dh.data)
                test_labels = np.vstack(getattr(s_test_dh, label_val)).squeeze()
                if normalize_features:
                    test_feats = normalizer.transform(test_feats)
            else:
                test_windows, test_meta = s_test_dh.parse_windows(int(dataset.sampling/1000 * window_size), int(dataset.sampling/1000 * window_inc), metadata_operations=metadata_operations)
                if normalize_features:
                    test_feats, _ = fe.extract_features(feature_list, test_windows, feature_dic=f_dic, normalize=True, normalizer=normalizer, fix_feature_errors=True)
                else:
                    test_feats = fe.extract_features(feature_list, test_windows, feature_dic=f_dic, fix_feature_errors=True)
                test_labels = test_meta[label_val]

            if regression:
                preds = clf.run(test_feats)
            else:
                preds, _ = clf.run(test_feats)
                
            metrics = om.extract_offline_metrics(metrics, test_labels, preds)
            accs.append(metrics)
                
            print(metrics)    
        accuracies[str(d)] = accs

        with open(output_file, 'wb') as handle:
            pickle.dump(accuracies, handle, protocol=pickle.HIGHEST_PROTOCOL)

def evaluate_weaklysupervised(model, window_size, window_inc, feature_list=['MAV'], feature_dic={}, included_datasets=['CIIL_WeaklySupervised'], output_file='out.pkl', regression=False, metrics=['CA'], normalize_data=False, normalize_features=False):
    """Evaluates an algorithm against all included datasets.
    
    Parameters
    ----------
    window_size: int
        The window size (**in ms**). 
    window_inc: int
        The window increment (**in ms**). 
    feature_list: list (default=['MAV'])
        A list of features. Pass in None for CNN.
    feature_dic: dic (default={})
        A dictionary of parameters for the passed in features.
    included_datasets: list (str) or list (DataSets)
        The name of the datasets you want to evaluate your model on. Either pass in strings (e.g., '3DC') for names or the dataset objects (e.g., _3DCDataset()). 
    output_file: string (default='out.pkl')
        The name of the directory you want to incrementally save the results to (it will be a pickle file).
    regression: boolean (default=False)
        If True, will create an EMGRegressor object. Otherwise creates an EMGClassifier object. 
    metrics: list (default=['CA']/['MSE'])
        The metrics to extract from each dataset.
    normalize_data: boolean (default=False)
        If True, the data will be normalized.
    normalize_features: boolean (default=False)
        If True, features will get normalized.
    Returns
    ----------
    dictionary
        A dictionary with a set of accuracies for different datasets
    """

    # -------------- Setup -------------------
    if metrics == ['CA'] and regression:
        metrics = ['MSE']

    metadata_operations = None 
    label_val = 'classes'
    if regression:
        metadata_operations = {'labels': 'last_sample'}
        label_val = 'labels'

    om = OfflineMetrics()

    # --------------- Run -----------------
    accuracies = {}
    for d in included_datasets:
        print(f"Evaluating {d} dataset...")
        if isinstance(d, str):
            dataset = get_dataset_list('WEAKLYSUPERVISED')[d]()
        else:
            dataset = d
        
        accs = []
        for s_i in range(0, dataset.num_subjects):
            data = dataset.prepare_data(split=True, subjects=[s_i])

            if data == None:
                print('Skipping Subject... No data found.')
                continue
            
            s_pretrain_dh = data['Pretrain']
            s_pretrain_dh.extra_attributes.remove('classes')
            delattr(s_pretrain_dh,"classes")
            s_train_dh = data['Train']
            s_test_dh = data['Test']

            # Normalize Data
            if normalize_data:
                filter = Filter(dataset.sampling)
                filter.install_filters({'name': 'standardize', 'data': s_pretrain_dh})
                filter.filter(s_pretrain_dh)
                filter.filter(s_train_dh)
                filter.filter(s_test_dh)

            pretrain_windows, pretrain_meta = s_pretrain_dh.parse_windows(int(dataset.sampling/1000 * window_size), int(dataset.sampling/1000 * window_inc), metadata_operations=metadata_operations)
            train_windows, train_meta = s_train_dh.parse_windows(int(dataset.sampling/1000 * window_size), int(dataset.sampling/1000 * window_inc), metadata_operations=metadata_operations)
            test_windows, test_meta = s_test_dh.parse_windows(int(dataset.sampling/1000 * window_size), int(dataset.sampling/1000 * window_inc), metadata_operations=metadata_operations)

            if feature_list is not None:
                fe = FeatureExtractor()
                if normalize_features:
                    pretrain_feats, scaler = fe.extract_features(feature_list, pretrain_windows, feature_dic=feature_dic, normalize=True, fix_feature_errors=True)
                    train_feats, _ = fe.extract_features(feature_list, train_windows, feature_dic=feature_dic, normalize=True, normalizer=scaler, fix_feature_errors=True)
                    test_feats, _ = fe.extract_features(feature_list, test_windows, feature_dic=feature_dic, normalize=True, normalizer=scaler, fix_feature_errors=True)      
                else:
                    pretrain_feats = fe.extract_features(feature_list, pretrain_windows, feature_dic=feature_dic, fix_feature_errors=True)
                    train_feats = fe.extract_features(feature_list, train_windows, feature_dic=feature_dic, fix_feature_errors=True)
                    test_feats = fe.extract_features(feature_list, test_windows, feature_dic=feature_dic, fix_feature_errors=True)     
            else:
                pretrain_feats = pretrain_windows
                train_feats = train_windows
                test_feats = test_windows

            ds = {
                'training_features': train_feats,
                'training_labels': train_meta[label_val],
                'pretraining_features': pretrain_feats
            }

            model.fit(ds)

            if not regression:
                clf = EMGClassifier(model)
            else:
                clf = EMGRegressor(model)
            
            if regression:
                preds = clf.run(test_feats)
            else:
                preds, _ = clf.run(test_feats)
                
            metrics = om.extract_offline_metrics(metrics, test_meta[label_val], preds)
            accs.append(metrics)
            print(str(s_i) + '/' + str(dataset.num_subjects) + ' completed.')
            print(metrics)    
        accuracies[str(dataset)] = accs

        with open(output_file, 'wb') as handle:
            pickle.dump(accuracies, handle, protocol=pickle.HIGHEST_PROTOCOL)   