import math
import numpy as np
import numpy.matlib as matlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA, FastICA
from sklearn.manifold import TSNE, Isomap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.stats import skew, kurtosis
from librosa import lpc
from pywt import wavedec, upcoef
from sklearn.preprocessing import StandardScaler

class FeatureExtractor:
    """
    Feature extraction class including feature groups, feature list, and feature extraction code.
    """
    def get_feature_groups(self):
        """Gets a list of all available feature groups.
        
        Returns
        ----------
        dictionary
            A dictionary with the all available feature groups and their respective features.
        """
        feature_groups = {'HTD': ['MAV', 'ZC', 'SSC', 'WL'],
                          'TSTD': ['MAVFD','DASDV','WAMP','ZC','MFL','SAMPEN','M0','M2','M4','SPARSI','IRF','WLF'],
                          'DFTR': ['DFTR'],
                          'ITD': ['ISD','COR','MDIFF','MLK'],
                          'HJORTH': ['ACT','MOB','COMP'],
                          'LS4': ['LS', 'MFL', 'MSR', 'WAMP'],
                          'LS9': ['LS', 'MFL', 'MSR', 'WAMP', 'ZC', 'RMS', 'IAV', 'DASDV', 'VAR'],
                          'TDPSD': ['M0','M2','M4','SPARSI','IRF','WLF'],
                          'TDAR': ['MAV', 'ZC', 'SSC', 'WL', 'AR'],
                          'COMB': ['WL', 'SSC', 'LD', 'AR9'],
                          'MSWT': ['WENG','WV','WWL','WENT']
                          }
        return feature_groups

    def get_feature_list(self):
        """Gets a list of all available features.
        
        Returns
        ----------
        list
            A list of all available features.
        """
        feature_list = ['MAV',
                        'ZC',
                        'SSC',
                        'WL',
                        'LS',
                        'MFL',
                        'MSR',
                        'WAMP',
                        'RMS',
                        'IAV',
                        'DASDV',
                        'VAR',
                        'M0',
                        'M2',
                        'M4',
                        'SPARSI',
                        'IRF',
                        'WLF',
                        'AR',
                        'CC',
                        'LD',
                        'MAVFD',
                        'MAVSLP',
                        'MDF',
                        'MNF',
                        'MNP',
                        'MPK',
                        'SKEW',
                        'KURT',
                        "RMSPHASOR",
                        "PAP",
                        "WLPHASOR",
                        "MZP",
                        "TM",
                        "SM",
                        "SAMPEN",
                        "FUZZYEN",
                        "DFTR",
                        "ISD",
                        "COR",
                        "MDIFF",
                        "MLK",
                        "ACT",
                        "MOB",
                        "COMP",
                        "WENG",
                        "WV",
                        "WWL",
                        "WENT",
                        "MEAN"]
        return feature_list
        
    def get_projection_list(self):
        """Gets a list of all available feature projections.
        
        Returns
        ----------
        list
            A list of all available projections.
        """
        projection_list = ['pca', 'kernelpca', 'ica', 'lda', 'tsne', 'isomap']
        return projection_list

    def extract_feature_group(self, feature_group, windows, feature_dic={}, array=False):
        """Extracts a group of features.
        
        Parameters
        ----------
        feature_group: string
            The group of features to extract. See the get_feature_list() function for valid options.
        windows: list 
            A list of windows - should be computed directly from the OfflineDataHandler or the utils.get_windows() method.
        feature_dic: dict
            A dictionary containing the parameters you'd like passed to each feature. ex. {"MDF_sf":1000}
        array: bool (optional), default=False 
            If True, the dictionary will get converted to a list. 
        Returns
        ----------
        dictionary or list 
            A dictionary where each key is a specific feature and its value is a list of the computed 
            features for each window.
        """
        features = {}
        if not feature_group in self.get_feature_groups():
            return features
        feats = self.extract_features(self.get_feature_groups()[feature_group], windows, feature_dic)
        if array:
            return self._format_data(feats)
        return feats 

    def extract_features(self, feature_list, windows, feature_dic={}, array=False, normalize=False, normalizer=None, fix_feature_errors=False):
        """Extracts a list of features.
        
        Parameters
        ----------
        feature_list: list
            The group of features to extract. Run get_feature_list() or checkout the API documentation 
            to find an up-to-date feature list.  
        windows: list 
            A list of windows - should be computed directly from the OfflineDataHandler or the utils.get_windows() method.
        feature_dic: dict
            A dictionary containing the parameters you'd like passed to each feature. ex. {"MDF_sf":1000}
        array: bool (optional), default=False 
            If True, the dictionary will get converted to a list.
        normalize: bool (optional), default=False
            If True, the features will be normalized between using sklearn StandardScaler. The returned object will be a list.
        normalizer: StandardScaler, default=None
            This should be set to the output from feature extraction on the training data. Do not normalize testing features without this as this could be considered information leakage. 
        fix_feature_errors: bool (optional), default=False
            If true, fixes all feature errors (NaN=0, INF=0, -INF=0).
        Returns
        ----------
        dictionary or list 
            A dictionary where each key is a specific feature and its value is a list of the computed 
            features for each window.
        StandardScaler
            If normalize is true it will return the normalizer object. This should be passed into the feature extractor for test data.
        """
        features = {}
        scaler = None
        for feature in feature_list:
            if feature in self.get_feature_list():
                method_to_call = getattr(self, 'get' + feature + 'feat')
                valid_keys = [i for i in list(feature_dic.keys()) if feature+"_" in i]
                smaller_dictionary = dict((k, feature_dic[k]) for k in valid_keys if k in feature_dic)
                feats = method_to_call(windows, **smaller_dictionary)
                if fix_feature_errors:
                    if self.check_features(feats, False):
                        feats = np.nan_to_num(feats, neginf=0, nan=0, posinf=0)
                features[feature] = feats
        if array:
            features = self._format_data(features)
        if normalize:
            if isinstance(features, dict):
                features = self._format_data(features)
            if not normalizer:
                scaler = StandardScaler()
                features = scaler.fit_transform(features)
            else:
                features = normalizer.transform(features)
            return features, scaler 
        return features 

    def check_features(self, features, silent=False):
        """Assesses a features object for np.nan, np.inf, and -np.inf. Can be used to check for clean data. 
        
        Parameters
        ----------
        features: np.ndarray or dict
            A group of features extracted with the feature extraction package in either dictionary or np.ndarray format
        silent: bool (default=False)
            If True, will silence all prints from this function.
        
        Returns
        ----------
        violations: int
            A number of violations found within the data. This is the number of types of violations (nan, inf, -inf) per feature
            summed across all features. Returning 0 indicates that the features contain no invalid elements.
        """
        violations = 0
        if type(features) == dict:
            violations = self._check_dict_features(features, silent)
        elif type(features) == np.ndarray:
            violations = self._check_ndarray_features(features, silent)
        return violations

    def _check_dict_features(self, features, silent=False):
        """A helper function that assesses specifically dictionary of np.ndarrays (what is returned from the feature extraction module)
        Parameters
        ----------
        features: dict
            A group of features extracted with the feature extraction package in dictionary format
        
        Returns
        ----------
        violations: int
            A number of violations found within the dictionary. This is the number of types of violations (nan, inf, -inf) per feature
            summed across all features. Returning 0 indicates that the features contain no invalid elements.
        """
        feature_list = list(features.keys())
        # sanity check that no errors were found in feature computation
        violations = 0
        for fk in feature_list:
            if (features[fk] == np.nan).any():
                violations += 1
                if not silent:
                    print(f"nan in  feature {fk}.")
            if (features[fk] == np.inf).any():
                violations += 1
                if not silent:
                    print(f"inf in feature {fk}.")
            if (features[fk] == -1*np.inf).any():
                violations += 1
                if not silent:
                    print(f"-inf in feature {fk}.")
        return violations

    def _check_ndarray_features(self, features, silent=False):
        """A helper function that assesses np.ndarrays directly.
        Parameters
        ----------
        features: np.ndarray
            A group of features extracted with the feature extraction package in np.ndarray format
        
        Returns
        ----------
        violations: int
            A number of violations found within the np.ndarray. This is the number of types of violations (np.nan, np.inf, -np.inf)
            across all features. Returning 0 indicates that the features contain no invalid elements. Unlike _check_dict_features, this 
            does not indicate the feature the violation arose from.
        """
        violations = 0
        if (features == np.nan).any():
            violations += 1
            if not silent:
                print(f"nan in  features.")
        if (features == np.inf).any():
            violations += 1
            if not silent:
                print(f"inf in features.")
        if (features == -1*np.inf).any():
            violations += 1
            if not silent:
                print(f"-inf in features.")
        return violations

    '''
    -----------------------------------------------------------------------------------------
    The following methods are all feature extraction methods. They should follow the same
    format that already exists (i.e., get<FEAT_ABBREVIATION>feat). The feature abbreviation
    should be added to the get feature_list function. 
    '''

    def getMAVfeat(self, windows):
        """Extract Mean Absolute Value (MAV) feature.
        
        Parameters
        ----------
        windows: list 
            A list of windows - should be computed directly from the OfflineDataHandler or the utils.get_windows() method.
        
        Returns
        ----------
        list
            The computed features associated with each window. 
        """
        feat = np.mean(np.abs(windows),2)
        return feat

    def getMEANfeat(self, windows):
        """Extract mean of signal (MEAN) feature.
        
        Parameters
        ----------
        windows: list 
            A list of windows - should be computed directly from the OfflineDataHandler or the utils.get_windows() method.
        Returns
        ----------
        list
            The computed features associated with each window. 
        """
        return np.mean(windows, -1)
    

    def getZCfeat(self, windows):
        """Extract Zero Crossings (ZC) feature.
        
        Parameters
        ----------
        windows: list 
            A list of windows - should be computed directly from the OfflineDataHandler or the utils.get_windows() method.
        
        Returns
        ----------
        list
            The computed features associated with each window. 
        """
        sgn_change = np.diff(np.sign(windows),axis=2)
        neg_change = sgn_change == -2
        pos_change = sgn_change ==  2
        feat_a = np.sum(neg_change,2)
        feat_b = np.sum(pos_change,2)
        return feat_a+feat_b
    

    def getSSCfeat(self, windows,SSC_threshold=0.0):
        """Extract Slope Sign Change (SSC) feature.
        
        Parameters
        ----------
        windows: list 
            A list of windows - should be computed directly from the OfflineDataHandler or the utils.get_windows() method.
        SSC_threshold: float
            The threshold the derivative must exceed to be counted as a sign change.

        Returns
        ----------
        list
            The computed features associated with each window. 
        """
        assert type(SSC_threshold) == float 
        w_2 = windows[:,:,2:]
        w_1 = windows[:,:,1:-1]
        w_0 = windows[:,:,:-2]
        con = (((w_1-w_0)*(w_1-w_2)) >= SSC_threshold)
        return np.sum(con,axis=2)

    def getWLfeat(self, windows):
        """Extract Waveform Length (WL) feature.
        
        Parameters
        ----------
        windows: list 
            A list of windows - should be computed directly from the OfflineDataHandler or the utils.get_windows() method.
        
        Returns
        ----------
        list
            The computed features associated with each window. 
        """
        feat = np.sum(np.abs(np.diff(windows,axis=2)),2)
        return feat

    def getLSfeat(self, windows):
        """Extract L-Score (LS) feature.
        
        Parameters
        ----------
        windows: list 
            A list of windows - should be computed directly from the OfflineDataHandler or the utils.get_windows() method.
        
        Returns
        ----------
        list
            The computed features associated with each window. 
        """
        feat = np.zeros((windows.shape[0],windows.shape[1]))
        for w in range(0, windows.shape[0],1):
            for c in range(0, windows.shape[1],1):
                tmp = self.__lmom(np.reshape(windows[w,c,:],(1,windows.shape[2])),2)
                feat[w,c] = tmp[0,1]
        return feat

    def __lmom(self, signal, nL):
        # same output to matlab when ones vector of various sizes are input
        b = np.zeros((1,nL-1))
        l = np.zeros((1,nL-1))
        b0 = np.zeros((1,1))
        b0[0,0] = np.mean(signal)
        n = signal.shape[1]
        signal = np.sort(signal, axis=1)
        for r in range(1,nL,1):
            num = np.tile(np.asarray(range(r+1,n+1)),(r,1))  - np.tile(np.asarray(range(1,r+1)),(1,n-r))
            num = np.prod(num,axis=0)
            den = np.tile(np.asarray(n),(1,r)) - np.asarray(range(1,r+1))
            den = np.prod(den)
            b[r-1] = 1/n * np.sum(num / den * signal[0,r:n])
        tB = np.concatenate((b0,b))
        B = np.flip(tB,0)
        for i in range(1, nL, 1):
            Spc = np.zeros((B.shape[0]-(i+1),1))
            Coeff = np.concatenate((Spc, self.__LegendreShiftPoly(i)))
            l[0,i-1] = np.sum(Coeff * B)
        L = np.concatenate((b0, l),1)

        return L

    def __LegendreShiftPoly(self, n):
        # Verified: this has identical function to MATLAB function for n = 2:10 (only 2 is used to compute LS feature)
        pk = np.zeros((n+1,1))
        if n == 0:
            pk = 1
        elif n == 1:
            pk[0,0] = 2
            pk[1,0] = -1
        else:
            pkm2 = np.zeros(n+1)
            pkm2[n] = 1
            pkm1 = np.zeros(n+1)
            pkm1[n] = -1
            pkm1[n-1] = 2

            for k in range(2,n+1,1):
                pk = np.zeros((n+1,1))
                for e in range(n-k+1,n+1,1):
                    pk[e-1] = (4*k-2)*pkm1[e]+ (1-2*k)*pkm1[e-1] + (1-k) * pkm2[e-1]
                pk[n,0] = (1-2*k)*pkm1[n] + (1-k)*pkm2[n]
                pk = pk/k

                if k < n:
                    pkm2 = pkm1
                    pkm1 = pk

        return pk

    def getMFLfeat(self, windows):
        """Extract Maximum Fractal Length (MFL) feature.
        
        Parameters
        ----------
        windows: list 
            A list of windows - should be computed directly from the OfflineDataHandler or the utils.get_windows() method.
        
        Returns
        ----------
        list
            The computed features associated with each window. 
        """
        feat = np.log10(np.sum(np.abs(np.diff(windows, axis=2)),axis=2))
        return feat

    def getMSRfeat(self, windows):
        """Extract Mean Squared Ratio (MSR) feature.
        
        Parameters
        ----------
        windows: list 
            A list of windows - should be computed directly from the OfflineDataHandler or the utils.get_windows() method.
        
        Returns
        ----------
        list
            The computed features associated with each window. 
        """
        feat = np.abs(np.mean(np.sqrt(windows.astype('complex')),axis=2))
        return feat

    def getWAMPfeat(self, windows, WAMP_threshold=2e-3):
        """Extract Willison Amplitude (WAMP) feature.
        
        Parameters
        ----------
        windows: list 
            A list of windows - should be computed directly from the OfflineDataHandler or the utils.get_windows() method.
        WAMP_threshold: float
            The value that must be exceeded by the derivative to be counted as a high variability sample
        Returns
        ----------
        list
            The computed features associated with each window. 
        """
        assert type(WAMP_threshold) == float
        feat = np.sum(np.abs(np.diff(windows, axis=2)) > WAMP_threshold, axis=2)
        return feat

    def getRMSfeat(self, windows):
        """Extract Root Mean Square (RMS) feature.
        
        Parameters
        ----------
        windows: list 
            A list of windows - should be computed directly from the OfflineDataHandler or the utils.get_windows() method.
        
        Returns
        ----------
        list
            The computed features associated with each window. 
        """
        feat = np.sqrt(np.mean(np.square(windows),2))
        return feat

    def getIAVfeat(self, windows):
        """Extract Integral of Absolute Value (IAV) feature.
        
        Parameters
        ----------
        windows: list 
            A list of windows - should be computed directly from the OfflineDataHandler or the utils.get_windows() method.
        
        Returns
        ----------
        list
            The computed features associated with each window. 
        """
        feat = np.sum(np.abs(windows),axis=2)
        return feat

    def getDASDVfeat(self, windows):
        """Difference Absolute Standard Deviation Value (DASDV) feature.
        
        Parameters
        ----------
        windows: list 
            A list of windows - should be computed directly from the OfflineDataHandler or the utils.get_windows() method.
        
        Returns
        ----------
        list
            The computed features associated with each window. 
        """
        feat = np.sqrt(np.mean(np.diff(windows,axis=2)**2,axis=2))
        return feat

    def getVARfeat(self, windows):
        """Extract Variance (VAR) feature.
        
        Parameters
        ----------
        windows: list 
            A list of windows - should be computed directly from the OfflineDataHandler or the utils.get_windows() method.
        
        Returns
        ----------
        list
            The computed features associated with each window. 
        """
        feat = np.var(windows,axis=2)
        return feat

    def getM0feat(self, windows):
        """Extract First Temporal Moment (M0) feature.
        
        Parameters
        ----------
        windows: list 
            A list of windows - should be computed directly from the OfflineDataHandler or the utils.get_windows() method.
        
        Returns
        ----------
        list
            The computed features associated with each window. 
        """
        
        def closure(w):
            m0 = np.sqrt(np.sum(w**2,axis=2))/(w.shape[2]-1)
            m0 = m0 ** 0.1 / 0.1
            return np.log(np.abs(m0))
        m0_ebp=closure(windows)
        m0_efp=closure(np.log(windows**2+np.spacing(1)))

        num=-2*np.multiply(m0_efp,m0_ebp)
        den=np.multiply(m0_efp, m0_efp) + np.multiply(m0_ebp, m0_ebp)

        #Feature extraction goes here
        return num/den
    
    def getM2feat(self, windows):
        """Extract Second Temporal Moment (M2) feature.
        
        Parameters
        ----------
        windows: list 
            A list of windows - should be computed directly from the OfflineDataHandler or the utils.get_windows() method.
        
        Returns
        ----------
        list
            The computed features associated with each window. 
        """
        def closure(w):
            m0 = np.sqrt(np.sum(w**2,axis=2))/(w.shape[2]-1)
            m0 = m0 ** 0.1 / 0.1
            d1 = np.diff(w, n=1, axis=2)
            m2 = np.sqrt(np.sum(d1 **2, axis=2)/ (w.shape[2]-1))
            m2 = m2 ** 0.1 / 0.1
            return np.log(np.abs(m0-m2))
        m2_ebp=closure(windows)
        m2_efp=closure(np.log(windows**2+np.spacing(1)))

        num=-2*np.multiply(m2_efp,m2_ebp)
        den=np.multiply(m2_efp, m2_efp) + np.multiply(m2_ebp, m2_ebp)
        
        return num/den

    def getM4feat(self, windows):
        """Extract Fourth Temporal Moment (M4) feature.
        
        Parameters
        ----------
        windows: list 
            A list of windows - should be computed directly from the OfflineDataHandler or the utils.get_windows() method.
        
        Returns
        ----------
        list
            The computed features associated with each window. 
        """
        def closure(w):
            m0 = np.sqrt(np.sum(w**2,axis=2))/(w.shape[2]-1)
            m0 = m0 ** 0.1 / 0.1
            d1 = np.diff(w, n=1, axis=2)
            d2 = np.diff(d1, n=1, axis=2)
            m4 = np.sqrt(np.sum(d2 **2, axis=2)/ (w.shape[2]-1))
            m4 = m4 ** 0.1 / 0.1
            return np.log(np.abs(m0-m4))
        m4_ebp=closure(windows)
        m4_efp=closure(np.log(windows**2+np.spacing(1)))
        
        num=-2*np.multiply(m4_efp,m4_ebp)
        den=np.multiply(m4_efp, m4_efp) + np.multiply(m4_ebp, m4_ebp)
        
        return num/den
    
    def getSPARSIfeat(self, windows):
        """Extract Sparsness (SPARSI) feature.
        
        Parameters
        ----------
        windows: list 
            A list of windows - should be computed directly from the OfflineDataHandler or the utils.get_windows() method.
        
        Returns
        ----------
        list
            The computed features associated with each window. 
        """
        def closure(w):
            m0 = np.sqrt(np.sum(w**2,axis=2))/(w.shape[2]-1)
            m0 = m0 ** 0.1 / 0.1
            d1 = np.diff(w, n=1, axis=2)
            m2 = np.sqrt(np.sum(d1 **2, axis=2)/ (w.shape[2]-1))
            m2 = m2 ** 0.1 / 0.1
            d2 = np.diff(d1, n=1, axis=2)
            m4 = np.sqrt(np.sum(d2 **2, axis=2)/ (w.shape[2]-1))
            m4 = m4 ** 0.1 / 0.1
            sparsi = np.sqrt(np.abs((m0-m2)*(m0-m4)))/m0
            return np.log(np.abs(sparsi))
        sparsi_ebp=closure(windows)
        sparsi_efp=closure(np.log(windows**2+np.spacing(1)))
        
        num=-2*np.multiply(sparsi_efp,sparsi_ebp)
        den=np.multiply(sparsi_efp, sparsi_efp) + np.multiply(sparsi_ebp, sparsi_ebp)

        return num/den

    def getIRFfeat(self, windows):
        """Extract Irregularity Factor (IRF) feature.
        
        Parameters
        ----------
        windows: list 
            A list of windows - should be computed directly from the OfflineDataHandler or the utils.get_windows() method.
        
        Returns
        ----------
        list
            The computed features associated with each window. 
        """
        def closure(w):
            m0 = np.sqrt(np.sum(w**2,axis=2))/(w.shape[2]-1)
            m0 = m0 ** 0.1 / 0.1
            d1 = np.diff(w, n=1, axis=2)
            m2 = np.sqrt(np.sum(d1 **2, axis=2)/ (w.shape[2]-1))
            m2 = m2 ** 0.1 / 0.1
            d2 = np.diff(d1, n=1, axis=2)
            m4 = np.sqrt(np.sum(d2 **2, axis=2)/ (w.shape[2]-1))
            m4 = m4 ** 0.1 / 0.1
            irf = m2/np.sqrt(m0*m4)
            return np.log(np.abs(irf))
        irf_ebp=closure(windows)
        irf_efp=closure(np.log(windows**2+np.spacing(1)))
        
        num=-2*np.multiply(irf_efp,irf_ebp)
        den=np.multiply(irf_efp, irf_efp) + np.multiply(irf_ebp, irf_ebp)

        return num/den

    def getWLFfeat(self, windows):
        """Waveform Length Factor (WLF) feature.
        
        Parameters
        ----------
        windows: list 
            A list of windows - should be computed directly from the OfflineDataHandler or the utils.get_windows() method.
        
        Returns
        ----------
        list
            The computed features associated with each window. 
        """
        def closure(w):
            d1 = np.diff(w, n=1, axis=2)
            
            d2 = np.diff(d1, n=1, axis=2)
            
            wlf = np.sqrt(np.sum(np.abs(d1),axis=2)/np.sum(np.abs(d2),axis=2))
            return np.log(np.abs(wlf))
        wlf_ebp=closure(windows)
        wlf_efp=closure(np.log(windows**2+np.spacing(1)))
        
        num=-2*np.multiply(wlf_efp,wlf_ebp)
        den=np.multiply(wlf_efp, wlf_efp) + np.multiply(wlf_ebp, wlf_ebp)

        return num/den

    def getARfeat(self, windows, AR_order=4):
        """Extract Autoregressive Coefficients (AR) feature.
        
        Parameters
        ----------
        windows: list 
            A list of windows - should be computed directly from the OfflineDataHandler or the utils.get_windows() method.
        AR_order: int, default=4
            The order of the autoregressive model.

        Returns
        ----------
        list
            The computed features associated with each window. 
        """
        assert type(AR_order) == int
        feature = np.reshape(lpc(windows, order=AR_order,axis=2)[:,:,1:],(windows.shape[0],AR_order*windows.shape[1]),order="C")
        return feature

    def getCCfeat(self, windows, CC_order=4):
        """Extract Cepstral Coefficient (CC) feature.
        
        Parameters
        ----------
        windows: list 
            A list of windows - should be computed directly from the OfflineDataHandler or the utils.get_windows() method.
        CC_order: int, default=4
            The order of the autoregressive and cepstral coefficient models.

        Returns
        ----------
        list
            The computed features associated with each window. 
        """
        assert type(CC_order) == int
        AR = self.getARfeat(windows, CC_order)
        cc = np.zeros_like(AR)
        cc[:,::CC_order] = -1*AR[:,::CC_order]
        if CC_order > 2:
            for p in range(1,CC_order):
                for l in range(1, p):
                    cc[:,p::CC_order] = cc[:,p::CC_order]+(AR[:,p::CC_order] * cc[:,p-l::CC_order] * (1-(l/p)))
                cc[:,p::CC_order] = -1*AR[:,p::CC_order]-cc[:,p::CC_order]
        return cc
    
    def getLDfeat(self, windows):
        """Extract Log Detector (LD) feature.
        
        Parameters
        ----------
        windows: list 
            A list of windows - should be computed directly from the OfflineDataHandler or the utils.get_windows() method.
        
        Returns
        ----------
        list
            The computed features associated with each window. 
        """
        return np.exp(np.mean(np.log(np.abs(windows)+1), 2))

    def getMAVFDfeat(self, windows):
        """Extract Mean Absolute Value First Difference (MAVFD) feature.
        
        Parameters
        ----------
        windows: list 
            A list of windows - should be computed directly from the OfflineDataHandler or the utils.get_windows() method.
        
        Returns
        ----------
        list
            The computed features associated with each window. 
        """
        mavfd = np.mean(np.abs(np.diff(windows,axis=2)),axis=2)
        return mavfd

    def getMAVSLPfeat(self, windows, MAVSLP_segment=2):
        """Extract Mean Absolute Value Slope (MAVSLP) feature.
       
        Parameters
        ----------
        windows: list 
            A list of windows - should be computed directly from the OfflineDataHandler or the utils.get_windows() method.
        MAVSLP_segment: int
            The number of segments to divide the window into
        Returns
        ----------
        list
            The computed features associated with each window. 
        """
        assert type(MAVSLP_segment) == int
        m = int(round(windows.shape[2]/MAVSLP_segment))
        mav = []
        mavslp = []
        for i in range(0,MAVSLP_segment):
            mav.append(np.mean(np.abs(windows[:,:,i*m:(i+1)*m]), axis=2))
        for i in range (0, MAVSLP_segment-1):
            mavslp.append(mav[i+1]- mav[i])
        mavslp = np.array(mavslp)
        return mavslp.reshape(mavslp.shape[1], -1)

    def getMDFfeat(self, windows,MDF_fs=1000):
        """Extract Median Frequency (MDF) feature.
        
        Parameters
        ----------
        windows: list 
            A list of windows - should be computed directly from the OfflineDataHandler or the utils.get_windows() method.
        MDF_fs: int, float
            The sampling frequency of the signal

        Returns
        ----------
        list
            The computed features associated with each window. 
        """
        assert type(MDF_fs) == int or type(MDF_fs) == float 
        def closure(winsize):
            return 1 if winsize==0 else 2**math.ceil(math.log2(winsize))
        nextpow2 = closure(windows.shape[2])
        spec = np.fft.fft(windows,nextpow2, axis=2)/windows.shape[2]
        spec = spec[:,:,0:int(nextpow2/2)]
        POW = np.real(spec * np.conj(spec))
        totalPOW = np.sum(POW, axis=2)
        cumPOW   = np.cumsum(POW, axis=2)
        medfreq = np.zeros((windows.shape[0], windows.shape[1]))
        for i in range(0, windows.shape[0]):
            for j in range(0, windows.shape[1]):
                medfreq[i,j] = (MDF_fs/2)*np.argwhere(cumPOW[i,j,:] > totalPOW[i,j] /2)[0]/(nextpow2/2)
        return medfreq

    def getMNFfeat(self, windows, MNF_fs=1000):
        """Extract Mean Frequency (MNF) feature.
        
        Parameters
        ----------
        windows: list 
            A list of windows - should be computed directly from the OfflineDataHandler or the utils.get_windows() method.
        MNF_fs: int, float
            The sampling frequency of the signal
        Returns
        ----------
        list
            The computed features associated with each window. 
        """
        assert type(MNF_fs) == int or type(MNF_fs) == float 
        def closure(winsize):
            return 1 if winsize==0 else 2**math.ceil(math.log2(winsize))
        nextpow2 = closure(windows.shape[2])
        spec = np.fft.fft(windows, n=nextpow2,axis=2)/windows.shape[2]
        f = np.fft.fftfreq(nextpow2)*MNF_fs
        spec = spec[:,:,0:int(round(spec.shape[2]/2))]
        f = f[0:int(round(nextpow2/2))]
        f = np.repeat(f[np.newaxis, :], spec.shape[0], axis=0)
        f = np.repeat(f[:, np.newaxis,:], spec.shape[1], axis=1)
        POW = spec * np.conj(spec)
        return np.real(np.sum(POW*f,axis=2)/np.sum(POW,axis=2))

    def getMNPfeat(self, windows):
        """Extract Mean Power (MNP) feature.
        
        Parameters
        ----------
        windows: list 
            A list of windows - should be computed directly from the OfflineDataHandler or the utils.get_windows() method.
        
        Returns
        ----------
        list
            The computed features associated with each window. 
        """
        def closure(winsize):
            return 1 if winsize==0 else 2**math.ceil(math.log2(winsize))
        nextpow2 = closure(windows.shape[2])
        spec = np.fft.fft(windows,n=nextpow2,axis=2)/windows.shape[2]
        spec = spec[:,:,0:int(round(nextpow2/2))]
        POW  = np.real(spec[:,:,:int(nextpow2)]*np.conj(spec[:,:,:int(nextpow2)]))
        return np.sum(POW, axis=2)/POW.shape[2]

    def getMPKfeat(self, windows):
        """Extract (MPK) feature.
        
        Parameters
        ----------
        windows: list 
            A list of windows - should be computed directly from the OfflineDataHandler or the utils.get_windows() method.
        
        Returns
        ----------
        list
            The computed features associated with each window. 
        """
        return windows.max(axis=2)

    def getSKEWfeat(self, windows):
        """Extract Skewness (SKEW) feature.
        
        Parameters
        ----------
        windows: list 
            A list of windows - should be computed directly from the OfflineDataHandler or the utils.get_windows() method.
        
        Returns
        ----------
        list
            The computed features associated with each window. 
        """
        return skew(windows, axis=2)

    def getKURTfeat(self, windows):
        """Extract Kurtosis (KURT) feature.
        
        Parameters
        ----------
        windows: list 
            A list of windows - should be computed directly from the OfflineDataHandler or the utils.get_windows() method.
        
        Returns
        ----------
        list
            The computed features associated with each window. 
        """
        return kurtosis(windows, axis=2, fisher=False)

    def getRMSPHASORfeat(self, windows):
        """Extract RMS Phasor feature (RMSPHASOR) feature. This 
        
        Parameters
        ----------
        windows: list 
            A list of windows - should be computed directly from the OfflineDataHandler or the utils.get_windows() method.
        
        Returns
        ----------
        list
            The computed features associated with each window. 
        """
        n_channels = windows.shape[1]
        n_windows = windows.shape[0]
        D_r = np.zeros((n_windows, n_channels*(n_channels-1)//2))
        del_D_r = np.zeros((n_windows, n_channels*(n_channels-1)//2))
        c_vec = np.zeros((n_windows, n_channels), dtype=np.complex64)
        for ch in range(n_channels):
            c_vec[:,ch] = np.exp(1j * (ch)*2*math.pi/n_channels)
        
        d1 = np.diff(windows,axis=2)

        r_c   = np.sqrt(np.mean(windows**2,axis=2)) * c_vec
        r_del = np.sqrt(np.mean(d1**2,axis=2)) * c_vec
        counter = 0
        for ch_i in range(n_channels):
            for ch_j in range(ch_i+1,n_channels):
                D_r[:,counter]     =np.real(np.sqrt((r_c[:,ch_i]-r_c[:,ch_j])*np.conj(r_c[:,ch_i]-r_c[:,ch_j])))
                del_D_r[:,counter] = np.real(np.sqrt((r_del[:,ch_i]-r_del[:,ch_j])*np.conj(r_del[:,ch_i]-r_del[:,ch_j])))
                counter += 1
        log_D_r = np.log(D_r)
        log_del_D_r = np.log(D_r/del_D_r)
        return np.concatenate((log_D_r, log_del_D_r),axis=1)

    def getWLPHASORfeat(self, windows):
        """Extract WL Phasor feature (WLPHASOR) feature. This 
        
        Parameters
        ----------
        windows: list 
            A list of windows - should be computed directly from the OfflineDataHandler or the utils.get_windows() method.
        
        Returns
        ----------
        list
            The computed features associated with each window. 
        """
        n_channels = windows.shape[1]
        n_windows = windows.shape[0]
        D_r = np.zeros((n_windows, n_channels*(n_channels-1)//2))
        del_D_r = np.zeros((n_windows, n_channels*(n_channels-1)//2))
        c_vec = np.zeros((n_windows, n_channels), dtype=np.complex64)
        for ch in range(n_channels):
            c_vec[:,ch] = np.exp(1j * (ch)*2*math.pi/n_channels)
        
        d1 = np.diff(windows,axis=2)
        
        r_c   = np.sum(np.abs(np.diff(windows,axis=2)),axis=2) * c_vec
        r_del = np.sum(np.abs(np.diff(d1,axis=2)),axis=2) * c_vec
        counter = 0
        for ch_i in range(n_channels):
            for ch_j in range(ch_i+1,n_channels):
                D_r[:,counter]     = np.real(np.sqrt((r_c[:,ch_i]-r_c[:,ch_j])*np.conj(r_c[:,ch_i]-r_c[:,ch_j])))
                del_D_r[:,counter] = np.real(np.sqrt((r_del[:,ch_i]-r_del[:,ch_j])*np.conj(r_del[:,ch_i]-r_del[:,ch_j])))
                counter += 1
        log_D_r = np.log(D_r)
        log_del_D_r = np.log(D_r/del_D_r)
        return np.concatenate((log_D_r, log_del_D_r),axis=1)

    def getPAPfeat(self, windows):
        """Extract Peak Average Power (PaP) feature.
        
        Parameters
        ----------
        windows: list 
            A list of windows - should be computed directly from the OfflineDataHandler or the utils.get_windows() method.
        
        Returns
        ----------
        list
            The computed features associated with each window. 
        """
        d1 = np.diff(windows, axis=2)
        d2 = np.diff(d1, axis=2)
        m0 = np.sqrt(np.sum(windows**2,axis=2))
        m2 = np.sqrt(np.sum(d1**2, axis=2))
        m4 = np.sqrt(np.sum(d2**2, axis=2))
        return  m0 / (m4/m2)


    def getMZPfeat(self, windows):
        """Extract Multiplication of Power and Peaks (MZP) feature.
        
        Parameters
        ----------
        windows: list 
            A list of windows - should be computed directly from the OfflineDataHandler or the utils.get_windows() method.
        
        Returns
        ----------
        list
            The computed features associated with each window. 
        """
        d1 = np.diff(windows, axis=2)
        phi = np.sqrt(1/windows.shape[2] * np.sum(d1 ** 2, axis=2))
        return phi

    def getTMfeat(self, windows, TM_order=3):
        """Extract Temporal Moment (TM) feature. Order should be defined.
        
        Parameters
        ----------
        windows: list 
            A list of windows - should be computed directly from the OfflineDataHandler or the utils.get_windows() method.
        TM_order: int, default=3
            The exponent the time series is raised to before the MAV is computed.
        
        Returns
        ----------
        list
            The computed features associated with each window. 
        """
        assert type(TM_order)==int
        return np.mean(np.abs(windows**TM_order), axis=2)

    def getSMfeat(self, windows, SM_order=2, SM_fs=1000):
        """Extract Spectral Moment (TM) feature. Order should be defined. Sampling frequency should be accurate
        for getting accurate frequency moments (physiological meaning). For pure pattern recognition problems, the 
        sampling frequency parameter is not a large issue.
        
        Parameters
        ----------
        windows: list 
            A list of windows - should be computed directly from the OfflineDataHandler or the utils.get_windows() method.
        SM_order: int, default=2
            The exponent that the frequency domain is raised to.
        SM_fs: float, default=1000
            The sampling frequency (in Hz).
        
        Returns
        ----------
        list
            The computed features associated with each window. 
        """
        assert type(SM_order)==int
        assert type(SM_fs)==int or type(SM_fs) == float
        def closure(winsize):
            return 1 if winsize==0 else 2**math.ceil(math.log2(winsize))
        nextpow2 = closure(windows.shape[2])
        spec = np.fft.fft(windows,n=nextpow2,axis=2)/windows.shape[2]
        pow  =  np.real(spec[:,:,0:int(round(nextpow2/2))] * np.conj(spec[:,:,0:int(round(nextpow2/2))]))
        f = np.fft.fftfreq(nextpow2)*SM_fs
        f = f[0:int(round(nextpow2/2))]
        f = np.repeat(f[np.newaxis, :], spec.shape[0], axis=0)
        f = np.repeat(f[:, np.newaxis,:], spec.shape[1], axis=1)
        return np.sum( pow*(f**SM_order),axis=2)

    def getSAMPENfeat(self, windows, SAMPEN_dim=2, SAMPEN_tolerance=0.3):
        """Extract Sample Entropy (SAMPEN) feature. SAMPEN_dim should be specified and is the number of samaples that 
        are used to define patterns. SAMPEN_tolerance depends on the dataset and is the minimum distance between patterns
        to be considered the same pattern; we recommend a value near 0.3.
        
        Parameters
        ----------
        windows: list 
            A list of windows - should be computed directly from the OfflineDataHandler or the utils.get_windows() method.
        SAMPEN_dim: int, default=2
            The number of samples patterns are defined under. Note: SAMPEN_dim and SAMPEN_dim+1 samples are used to get the two
            patterns of the final logarithmic ratio.
        SAMPEN_tolerance: float, default=0.3
            The threshold for patterns to be considered similar
        
        Returns
        ----------
        list
            The computed features associated with each window. 
        """
        assert type(SAMPEN_dim) == int
        assert SAMPEN_dim > 1
        assert type(SAMPEN_tolerance) == float
        assert SAMPEN_tolerance > 0
        #standardize within window
        N = windows.shape[2]
        window_mean = np.mean(windows,axis=2)
        window_std = np.std(windows, axis=2)
        series = (windows - np.repeat(window_mean[:,:,None], N, axis=2))/np.repeat(window_std[:,:,None], N, axis=2)
        # get sampen feature variable ready
        sampen = np.zeros((windows.shape[0], windows.shape[1]))
        # We don't have an efficient implementation (doing all channels/windows with matrix multiplication)
        # so do it one window at a time
        for w  in range(windows.shape[0]):
            for ch in range(windows.shape[1]):
                results = []
                for j in [1,2]:
                    m = SAMPEN_dim + j - 1
                    patterns = np.zeros((m,N-m+1))
                    count = np.zeros((N-m))
                    # if 1 d embedding
                    if m == 1:
                        patterns = series[w,ch,:]
                    else:
                        for k in range(0,m):
                            patterns[k,:] = series[w,ch, k:N-m+k+1]
                    
                    # Count the number of patterns whose distance is less than the tolerance
                    for k in range(N-m):
                        # compute the distance between each pattern and other patterns
                        if m == 1:
                            tmp = np.abs(patterns - matlib.repmat(patterns[:,k],1,N-m+1))
                        else:
                            tmp = np.max(np.abs(patterns - matlib.repmat(patterns[:,k,np.newaxis],1,N-m+1)),axis=0)
                        mask = (tmp <= SAMPEN_tolerance)
                        count[k] = (np.sum(mask)-1) # we remove 1 to avoid self comparison, in theory this means we can eventually do log of 0 (error)
                        # that is why we need the eps / np.spacing(1)
                    # average the number of similar patterns
                    count = count / (N-SAMPEN_dim-1)
                    results.append(np.mean(count)) 
                sampen[w,ch] = np.log((results[0]+np.spacing(1))/(results[1]+np.spacing(1)))
        return sampen

    def getFUZZYENfeat(self, windows, FUZZYEN_dim=2, FUZZYEN_tolerance=0.3, FUZZYEN_win=2):
        """Extract Fuzzy Entropy (FUZZYEN) feature. SAMPEN_dim should be specified and is the number of samaples that 
        are used to define patterns. SAMPEN_tolerance depends on the dataset and is the minimum distance between patterns
        to be considered the same pattern; we recommend a value near 0.3.
        
        Parameters
        ----------
        windows: list 
            A list of windows - should be computed directly from the OfflineDataHandler or the utils.get_windows() method.
        FUZZYEN_dim: int, default=2
            The number of samples patterns are defined under. Note: SAMPEN_dim and SAMPEN_dim+1 samples are used to get the two
            patterns of the final logarithmic ratio.
        FUZZYEN_tolerance: float, default=0.3
            The threshold for patterns to be considered similar
        FUZZYEN_win: list, default=2
            the order the distance matrix is raised to prior to determining the similarity.
        
        Returns
        ----------
        list
            The computed features associated with each window. 
        """
        # check arguments
        assert type(FUZZYEN_dim) == int
        assert FUZZYEN_dim > 1
        assert type(FUZZYEN_tolerance) == float
        assert FUZZYEN_tolerance > 0
        assert type(FUZZYEN_win) == int
        assert FUZZYEN_win > 0

        #standardize within window
        FUZZYEN_tolerance = FUZZYEN_tolerance*np.std(windows,axis=2)
        N = windows.shape[2]
        fuzzyen = np.zeros((windows.shape[0], windows.shape[1]))
        # We don't have an efficient implementation (doing all channels/windows with matrix multiplication)
        # so do it one window at a time
        for w  in range(windows.shape[0]):
            for ch in range(windows.shape[1]):
                results = []
                for j in [1,2]:
                    m = FUZZYEN_dim + j - 1
                    dataMat = np.zeros((m,N-m+1))
                    phi = np.zeros((N-m+1))
                    # setup the patterns
                    if m == 1:
                        dataMat[m,:] = windows[w,ch,:]
                    else:
                        for k in range(0,m):
                            dataMat[k,:] = windows[w,ch, k:N-m+k+1]
                    
                    # Count the number of patterns whose distance is less than the tolerance
                    for k in range(N-m+1):
                        # make the patterns zero mean
                        dataMat[:,k] = dataMat[:,k]-np.mean(dataMat[:,k])

                    for k in range(N-m):
                        # compute the distance between each pattern and other patterns
                        if m == 1:
                            tmp = np.abs(dataMat - matlib.repmat(dataMat[:,k],1,N-m+1))
                        else:
                            tmp = np.max(np.abs(dataMat - matlib.repmat(dataMat[:,k,np.newaxis],1,N-m+1)),axis=0)
                        # now get the similarity
                        simi = np.exp(((-1)*((tmp)**FUZZYEN_win))/FUZZYEN_tolerance[w,ch])
                        phi[k]=(np.sum(simi)-1) / (windows.shape[2]-m-1)

                    # average the number of similar patterns
                    
                    results.append(np.sum(phi)/(N-m)) 
                fuzzyen[w,ch] = np.log((results[0]+np.spacing(1))/(results[1]+np.spacing(1)))
        return fuzzyen

    def getDFTRfeat(self, windows, DFTR_fs=1000):
        """Extract Discrete Time Fourier Transform Representation (DFTR) feature. He et al., 2015 used this feature set with various normalization approaches
        (channel-wise and global), and achieved robustness to contraction intensity variability, but the normalization is required for the robustness. DFTR divides 
        the frequency spectrum into bins (20-92, 92-163, 163-235, 235-307, 307-378, 378-450). The number of returned bands depends on the DFTR_fs supplied. Note, it will
        only extract the band if the entire band lies under the frequency bin.
        
        Parameters
        ----------
        windows: list 
            A list of windows - should be computed directly from the OfflineDataHandler or the utils.get_windows() method.
        DFTR_fs: float, default=1000
        
        Returns
        ----------
        list
            The computed features associated with each window. 
        """
        assert type(DFTR_fs)==int or type(DFTR_fs) == float
        def closure(winsize):
            return 1 if winsize==0 else 2**math.ceil(math.log2(winsize))
        init_freq = 20
        upper_freqs = [92, 163, 235, 305, 378, 450]
        nyquist = DFTR_fs/2
        num_bins = sum([i <nyquist for i in upper_freqs])
        feat = np.zeros((windows.shape[0], windows.shape[1]*num_bins))

        nextpow2 = closure(windows.shape[2])
        spec = np.fft.fft(windows,n=nextpow2,axis=2)/windows.shape[2]
        pow  =  np.abs(spec[:,:,0:int(round(nextpow2/2))])
        f = np.fft.fftfreq(nextpow2)*DFTR_fs
        f = f[0:int(round(nextpow2/2))]
        for bin in range(num_bins):
            upper_freq = upper_freqs[bin]
            included_bins =  np.logical_and((f > init_freq) , (f < upper_freq))
            bin_energy = np.sum(pow[:,:,included_bins], axis=2)/ np.sum(included_bins)
            feat[:,bin*windows.shape[1]: (bin+1)*windows.shape[1]] = bin_energy **(2/3)
            init_freq = upper_freqs[bin]
        return feat

    def getISDfeat(self, windows):
        """Extract Integral Square Descriptor (ISD) feature. Another signal amplitude and power feature. In the invTD feature set, this feature
        is meant to capture energy levels (not be robust to contraction intensity).
        
        Parameters
        ----------
        windows: list 
            A list of windows - should be computed directly from the OfflineDataHandler or the utils.get_windows() method.
        
        Returns
        ----------
        list
            The computed features associated with each window. 
        """
        return np.sum(windows**2, axis=2)



    def getCORfeat(self, windows):
        """Extract Coefficient of Regularization (COR) feature. Very similar formulation to some of the TDPSD features.
        
        Parameters
        ----------
        windows: list 
            A list of windows - should be computed directly from the OfflineDataHandler or the utils.get_windows() method.
        
        Returns
        ----------
        list
            The computed features associated with each window. 
        """
        ISD = np.sum(windows**2, axis=2)
        normRSd1 = np.sum(np.diff(windows,axis=2)**2,axis=2)/windows.shape[2]
        normRSd2 = np.sum(np.diff(np.diff(windows, axis=2),axis=2)**2,axis=2)/windows.shape[2]
        COR = normRSd1/(normRSd2*ISD)
        return COR
        
    def getMDIFFfeat(self, windows):
        """Extract Mean Difference Derivative (MDIFF) feature. This is a feature that is the same metric as normRSD1 from COR.
        
        Parameters
        ----------
        windows: list 
            A list of windows - should be computed directly from the OfflineDataHandler or the utils.get_windows() method.
        
        Returns
        ----------
        list
            The computed features associated with each window. 
        """
        return np.sum(np.diff(windows,axis=2)**2,axis=2)/windows.shape[2]

    def getMLKfeat(self, windows):
        """Extract Mean Logarithm Kernel (MLK) feature. 
        
        Parameters
        ----------
        windows: list 
            A list of windows - should be computed directly from the OfflineDataHandler or the utils.get_windows() method.
        
        Returns
        ----------
        list
            The computed features associated with each window. 
        """
        return np.log(np.sum(np.abs(windows),axis=2)+np.spacing(1))/windows.shape[2]

    def getACTfeat(self, windows):
        """Extract Activation (ACT) feature. This feature is very similar to the zeroth order moment feature of TDPSD (M0); however, it undergoes 
        no nonlinear normalization.
        
        Parameters
        ----------
        windows: list 
            A list of windows - should be computed directly from the OfflineDataHandler or the utils.get_windows() method.
        
        Returns
        ----------
        list
            The computed features associated with each window. 
        """
        return np.mean(windows**2,axis=2)

    def getMOBfeat(self, windows):
        """Extract Mobility (MOB) feature. This feature is sqrt(m2/m0), where m0 and m2 are the first and second order moments found via
        Parseval's theorem. Interestingly, the Gabor frequency tells us that the number of zero crossings per unit time can be described 
        using 1/pi * mobility.
        
        Parameters
        ----------
        windows: list 
            A list of windows - should be computed directly from the OfflineDataHandler or the utils.get_windows() method.
        
        Returns
        ----------
        list
            The computed features associated with each window. 
        """
        m0 =  self.getACTfeat(windows)
        m2 =  np.sum(np.diff(windows,axis=2)**2,axis=2)/windows.shape[2]
        return np.sqrt(m2/m0)

    def getCOMPfeat(self, windows):
        """Extract Complexity (COMP) feature. This feature is sqrt(m4/m2), where m2 and m4 are the second and fourth order moments found via
        Parseval's theorem. It is a measure of the the similarity of the shape of a signal compared to a pure sine waveform. Because the Gabor frequency 
        tells us the number of zero crossings per unit time, the derivative of this metric gives us the number of extrema per unit time.
        
        Parameters
        ----------
        windows: list 
            A list of windows - should be computed directly from the OfflineDataHandler or the utils.get_windows() method.
        
        Returns
        ----------
        list
            The computed features associated with each window. 
        """
        m2 =  np.sum(np.diff(windows,axis=2)**2,axis=2)/windows.shape[2]
        m4 =  np.sum(np.diff(np.diff(windows, axis=2),axis=2)**2)/windows.shape[2]
        return np.sqrt(m4/m2)
    
    def getWENGfeat(self, windows, WENG_fs = 1000):
        """Extract Wavelet Energy (WENG) feature.
        
        Parameters
        ----------
        windows: list 
            A list of windows - should be computed directly from the OfflineDataHandler or the utils.get_windows() method.
        WENG_fs: int
            The sampling frequency of the signal. This determines the number of wavelets used to decompose the signal.
        
        Returns
        ----------
        list
            The computed features associated with each window. Size: Wx((order+1)*Nchannels)
        """
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # get the highest power of 2 the nyquist rate is divisible by
            order    = math.floor(np.log(WENG_fs/2)/np.log(2) - 1)
            # Khushaba et al suggests using sym8
            # note, this will often throw a WARNING saying the user specified order is too high -- but this is what the 
            # original paper suggests using as the order.
            wavelets =  wavedec(windows, wavelet='sym8', level=order,axis=2)
            # for every order, compute the energy (sum of DWT) - total of the squared signal
            features = np.hstack([np.log(np.sum(i**2, axis=2)+1e-10) for i in wavelets])
            return features


    def getWVfeat(self, windows, WV_fs=1000):
        """Extract Wavelet Variance (WV) feature.
        
        Parameters
        ----------
        windows: list 
            A list of windows - should be computed directly from the OfflineDataHandler or the utils.get_windows() method.
        WV_fs: int
            The sampling frequency of the signal. This determines the number of wavelets used to decompose the signal.
        Returns
        ----------
        list
            The computed features associated with each window. 
        """
        # get the highest power of 2 the nyquist rate is divisible by
        order    = math.floor(np.log(WV_fs/2)/np.log(2) - 1)
        # Khushaba et al suggests using sym8
        # note, this will often throw a WARNING saying the user specified order is too high -- but this is what the 
        # original paper suggests using as the order.
        wavelets =  wavedec(windows, wavelet='sym8', level=order,axis=2)
        # for every order, compute the variance  (squared sum of DWT) - this is variance of the energy, so we keep the square
        features = np.hstack([np.log(np.var(i**2, axis=2)+1e-10) for i in wavelets])
        return features

    def getWWLfeat(self, windows, WWL_fs=1000):
        """Extract Wavelet Waveform Length (WWL) feature.
        
        Parameters
        ----------
        windows: list 
            A list of windows - should be computed directly from the OfflineDataHandler or the utils.get_windows() method.
        WWL_fs: int
            The sampling frequency of the signal. This determines the number of wavelets used to decompose the signal.
        Returns
        ----------
        list
            The computed features associated with each window. 
        """
        # get the highest power of 2 the nyquist rate is divisible by
        order    = math.floor(np.log(WWL_fs/2)/np.log(2) - 1)
        # Khushaba et al suggests using sym8
        # note, this will often throw a WARNING saying the user specified order is too high -- but this is what the 
        # original paper suggests using as the order.
        wavelets =  wavedec(windows, wavelet='sym8', level=order,axis=2)
        # for every order, compute the waveform length (sum of absolute differences) -- this is WL of the energy, so we keep the square
        features = np.hstack([np.log(np.sum(np.abs(np.diff(i**2, axis=2)),axis=2)+1e-10) for i in wavelets])
        return features

    def getWENTfeat(self, windows, WENT_fs=1000):
        """Extract Wavelet Entropy (WENT) feature.
        
        Parameters
        ----------
        windows: list 
            A list of windows - should be computed directly from the OfflineDataHandler or the utils.get_windows() method.
        WENT_fs: int
            The sampling frequency of the signal. This determines the number of wavelets used to decompose the signal.
        Returns
        ----------
        list
            The computed features associated with each window. 
        """# get the highest power of 2 the nyquist rate is divisible by
        order    = math.floor(np.log(WENT_fs/2)/np.log(2) - 1)
        # Khushaba et al suggests using sym8
        # note, this will often throw a WARNING saying the user specified order is too high -- but this is what the 
        # original paper suggests using as the order.
        wavelets =  wavedec(windows, wavelet='sym8', level=order,axis=2)
        longs    = np.expand_dims(np.array([i.shape[2] for i in wavelets]),(1,2))
        # for every order, compute the energy (squared sum of DWT)
        window_energy = [i **2 for i in wavelets]
        # within each window:
        # 1. find the percentage of total energy a sample has (normalize by channel/wavelet amplitude)
        # 2. once you have this "probability" convert it to an entropy on a per sample basis with:
        #       entropy = x log x; where x is a probability
        # 3. with the per sample entropy, just take the sum 
        total_energy    = [np.sum(i, 2) for i in window_energy]
        prob = [i/np.expand_dims(j,2) for  i,j in zip(window_energy, total_energy)]
        features = np.hstack([-np.sum(i*np.log(i), axis=2) for i in prob])
        return features

    def getMEANfeat(self, windows):
        """Extract mean of signal (MEAN) feature.
        
        Parameters
        ----------
        windows: list 
            A list of windows - should be computed directly from the OfflineDataHandler or the utils.get_windows() method.
        Returns
        ----------
        list
            The computed features associated with each window. 
        """
        return np.mean(windows, -1)

    def visualize(self, feature_dic):
        """Visualize a set of features.
        
        Parameters
        ----------
        feature_dic: dict
            A dictionary consisting of the different features. This is the output from the 
            extract_features method.
        """
        plt.style.use('ggplot')
        if len(feature_dic) > 1:
            fig, ax = plt.subplots(len(feature_dic))
            index = 0
            labels = []
            for f in feature_dic:
                for i in range(0,len(feature_dic[f][0])):
                    x = list(range(0,len(feature_dic[f])))
                    lab = "CH"+str(i+1)
                    ax[index].plot(x, feature_dic[f][:], label=lab)
                    if not lab in labels:
                        labels.append(lab)
                    ax[index].set_ylabel(f)
                index += 1
                fig.suptitle('Features')
                fig.legend(labels, loc='lower right')
        else:
            key = list(feature_dic.keys())[0]
            plt.title(key)
            plt.plot(list(range(0,len(feature_dic[key]))), feature_dic[key])
        plt.show()
    
    def visualize_all_distributions(self, feature_dic, classes=None, savedir=None, render=True):
        """Visualize the distribution of each feature using a histogram. This will render the histograms all together.
        
        Parameters
        ----------
        feature_dic: dict
            A dictionary consisting of the different features. This is the output from the 
            extract_features method.
        classes: list
            The classes for each window.
        savedir: string
            The location the plot should be saved to. Specify the full filepath i.e., "figs/subject1.png".
        render: boolean
            Boolean to indicate whether the plot is shown or not.
        """
        # visualize the distribution of features
        plt.style.use('ggplot')
        if classes is not None:
            class_list = np.unique(classes)
        # get highest number of "features" per metric
        largest_metric = 0
        for key in feature_dic.keys():
            if feature_dic[key].shape[1] > largest_metric:
                largest_metric = feature_dic[key].shape[1]
        num_features = len(feature_dic.keys())
        fig, ax = plt.subplots(num_features, largest_metric, figsize=(largest_metric*4, num_features*4))
        if len(ax.shape) == 1:
            ax = ax[np.newaxis,:]
        for i, k in enumerate(feature_dic.keys()):
            for f in range(feature_dic[k].shape[1]):
                if classes is None:
                    ax[i,f].hist(feature_dic[k][:,f], 30)
                else:
                    for c in class_list:
                        class_id = classes == c
                        ax[i,f].hist(feature_dic[k][class_id,f],30,label=str(c),alpha=0.75)
                    if i == 0 and f == 0:
                        ax[i,f].legend()
                ax[i,f].set_title(k+str(f))
        plt.tight_layout()
        if savedir is not None:
            plt.savefig(savedir)
        if render:
            plt.show()

    def visualize_single_distributions(self, feature_dic, classes=None, savedir=None, render=False):
        """Visualize the distribution of each feature using a histogram. This will render one histogram per feature.
        
        Parameters
        ----------
        feature_dic: dict
            A dictionary consisting of the different features. This is the output from the 
            extract_features method.
        classes: list
            The classes for each window. Easily obtained from the odh.parse_windows() method.
        savedir: string
            The location the plot should be saved to. Specify the prefix i.e., "figs/subject". Feature names and .png are appended to this prefix
        render: boolean
            Boolean to indicate whether the plot is shown or not. Defaults to False.
        """
        # visualize the distribution of features
        plt.style.use('ggplot')
        if classes is not None:
            class_list = np.unique(classes)
        # get highest number of "features" per metric
        
        for i, k in enumerate(feature_dic.keys()):
            metric_length = feature_dic[k].shape[1]
            nearest_square = int(np.ceil(np.sqrt(metric_length)))
            fig, ax = plt.subplots(nearest_square, nearest_square, figsize=(nearest_square*4, nearest_square*4))
            ax = np.reshape(ax,-1)
 
            for f in range(feature_dic[k].shape[1]):
                if classes is None:
                    ax[f].hist(feature_dic[k][:,f], 30)
                else:
                    for c in class_list:
                        class_id = classes == c
                        ax[f].hist(feature_dic[k][class_id,f],30,label=str(c),alpha=0.75)
                    if i == 0 and f == 0:
                        ax[f].legend()
                ax[f].set_title(k+str(f))
            plt.tight_layout()
            if savedir is not None:
                plt.savefig(savedir+k+".png")
            if render:
                plt.show()

    def visualize_feature_space(self, feature_dic, projection, classes=None, savedir=None, render=True, test_feature_dic=None, t_classes=None, normalize=True, projection_params=None):
        """Visualize the the feature space through a certain projection.
        
        Parameters
        ----------
        feature_dic: dict
            A dictionary consisting of the different features. This is the output from the 
            extract_features method.
        classes: list
            The classes for each window.
        savedir: string
            The location the plot should be saved to. Specify the prefix i.e., "figs/subject". Feature names and .png are appended to this prefix
        render: boolean
            Boolean to indicate whether the plot is shown or not. Defaults to False.
        test_feature_dic: dict
            A dictionary consisting of the different features. This is the output from the extract_features method.
        t_classes: boolean
            The classes for each window of testing data.
        normalize: boolean
            Whether the user wants to scale features to zero mean and unit standard deviation before projection (recommended).
        projection_params: dict
            Extra parameters taken by the projection method.
        """

        if projection.lower() in self.get_projection_list():
            for i, k in enumerate(feature_dic.keys()):
                feature_matrix = feature_dic[k] if i == 0 else np.hstack((feature_matrix, feature_dic[k]))
                if test_feature_dic is not None:
                    t_feature_matrix = test_feature_dic[k] if i == 0 else np.hstack((t_feature_matrix, test_feature_dic[k]))
                else :
                    t_feature_matrix = None
            
            # normalization
            if normalize:
                feature_means = np.mean(feature_matrix, axis=0)
                feature_stds  = np.std(feature_matrix, axis=0)
                feature_matrix = (feature_matrix - feature_means) / feature_stds
                if test_feature_dic is not None:
                    t_feature_matrix = (t_feature_matrix - feature_means)/feature_stds


            projection_engine = self.__build_projection(projection, feature_matrix.shape[1],  projection_params)
            train_data, test_data = self.__project_data(projection, projection_engine, feature_matrix, classes, t_feature_matrix)

            fig, ax = plt.subplots()
            if classes is not None:
                class_list = np.unique(classes)
                n_classes = len(class_list)
                for c in class_list:
                    class_ids = classes == c
                    ax.scatter(train_data[class_ids,0], train_data[class_ids,1], marker='.', alpha=0.75, label="tr "+str(int(c)))
            else:
                ax.scatter(train_data[:,0], train_data[:,1], marker=".", label="tr", color='black', alpha=0.70)
                n_classes = 1
            ax.set_prop_cycle(None)

            if test_feature_dic is not None:
                if t_classes is not None:
                    t_class_list = np.unique(t_classes)
                    n_classes += len(t_class_list)
                    for c in t_class_list:
                        class_ids = t_classes == c 
                        ax.scatter(test_data[class_ids,0], test_data[class_ids,1], marker='+', alpha=0.75, label="te "+str(int(c)))
                else:
                    ax.scatter(test_data[:,0], test_data[:,1], marker="+", label="te", color='black', alpha=0.70)
                    n_classes += 1
            
            #dynamic legend based on the number of classes in the training and testing data. 
            ncol = 2*(int(n_classes//16)) if n_classes > 16 else 2
            text_size = 8 if n_classes > 16 else 10
            ax.legend(prop={'size': text_size}, ncol=ncol)
            ax.set_xlabel("Feature 1")
            ax.set_ylabel("Feature 2")
            ax.set_title("{} Visualization".format(projection.upper()))

            if projection.lower() == "pca":
                fig_pca, ax_pca = plt.subplots()
                ax_pca.stem(np.cumsum(projection_engine.explained_variance_ratio_))
                ax_pca.set_xlabel("Number of Components")
                ax_pca.set_ylabel("Explained Variance")
                ax_pca.set_title("Cumulative Variance Expressed by PCA")

            plt.tight_layout()

            if savedir is not None:
                fig.savefig(savedir+projection+".png")
                if projection.lower() == "pca":
                    fig_pca.savefig(savedir+"PCA_variance.png")
            if render:
                plt.show()
                
        else:
            raise ValueError("Unsupported projection method passed to FeatureExtractor.visualize_feature_space")
    
    def __build_projection(self, projection, n_components, projection_params):
        """Build the projection engine .
        
        Parameters
        ----------
        projection: string
            The projection method to be used. See the get_projection_list()
        n_components: int
            The number of components to projected.
        projection_params: dict
            The parameters for the projection method.
        
        Returns
        ----------
        projection_engine: sklearn.decomposition, sklearn.manifold, sklearn.discriminant_analysis or  
            The projection engine used.
        """ 
        if projection_params is None:
            projection_params = {}

        if "pca" not in projection.lower():
            n_components = 2

        if projection.lower() == "kernelpca":
            projection_engine = KernelPCA(n_components=n_components, **projection_params)
        elif projection.lower() == "isomap":
            projection_engine = Isomap(n_components=n_components, **projection_params)
        elif projection.lower() == "ica":
            projection_engine = FastICA(n_components=n_components, **projection_params)
        else:
            projection_engine = eval(projection.upper()+"(n_components=n_components, **projection_params)")
        return projection_engine
    
    def __project_data(self, projection, projection_engine, feature_matrix, classes, t_feature_matrix):
        """Project the data using the projection engine.
        
        Parameters
        ----------
        projection: string
            The projection method to be used. See the get_projection_list()
        projection_engine: sklearn.decomposition, sklearn.manifold, sklearn.discriminant_analysis or  
            The projection engine used.
        feature_matrix: np.ndarray
            The feature matrix to be projected.
        classes: list
            The classes for each window.
        t_feature_matrix: np.ndarray
            The feature matrix to be projected.
        t_classes: list
            The classes for each window.
        
        Returns
        ----------
        train_data: np.ndarray
            The projected training data.
        test_data: np.ndarray
            The projected testing data.
        """ 
        train_data = None
        test_data = None

        if projection.lower() == "lda":
            if classes is None:
                raise ValueError("LDA requires class labels to be passed to the FeatureExtractor.visualize_feature_space method")
            else:
                train_data = projection_engine.fit_transform(feature_matrix, classes)
                if t_feature_matrix is not None:
                    test_data = projection_engine.transform(t_feature_matrix)
            
        elif projection.lower() == "tsne" or projection.lower() == "isomap":
            if t_feature_matrix is not None:
                concatened_feature_matrix = np.concatenate((feature_matrix, t_feature_matrix), axis=0)
                projected_data = projection_engine.fit_transform(concatened_feature_matrix)
                train_data = projected_data[0:feature_matrix.shape[0],:]
                test_data = projected_data[feature_matrix.shape[0]:,:]
            else :
                train_data = projection_engine.fit_transform(feature_matrix)
        else:
            train_data = projection_engine.fit_transform(feature_matrix)
            if t_feature_matrix is not None:
                test_data = projection_engine.transform(t_feature_matrix)

        return train_data, test_data
    
    def _format_data(self, feature_dictionary):
        if not isinstance(feature_dictionary, dict):
            return feature_dictionary
        
        arr = None
        for feat in feature_dictionary:
            if arr is None:
                arr = feature_dictionary[feat]
            else:
                arr = np.hstack((arr, feature_dictionary[feat]))
        return arr
            
