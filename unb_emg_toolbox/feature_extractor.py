import numpy as np
from scipy.stats import skew, kurtosis

class FeatureExtractor:
    """
    Feature extraction class including feature groups, feature list, and feature extraction code.

    Parameters
    ----------
    num_channels: int > 0
        The number of EMG channels. 
    feature_list: array_like (optional)
        A list of features to remember within the object. Used for when this object is passed to an EMGClassifier.
    feature_group: string (optional)
         A feature group to remember. Used for when this object is passed to an EMGClassifier.
    """
    def __init__(self, num_channels, feature_list=[], feature_group=None):
        self.num_channels = num_channels

    def get_feature_groups(self):
        """Gets a list of all available feature groups.

        Returns
        ----------
        dictionary
            A dictionary with the all available feature groups.
        """
        feature_groups = {'HTD': ['MAV', 'ZC', 'SSC', 'WL'],
                          'TD4': ['LS', 'MFL', 'MSR', 'WAMP'],
                          'TD9': ['LS', 'MFL', 'MSR', 'WAMP', 'ZC', 'RMS', 'IAV', 'DASDV', 'VAR'],
                          'TDPSD': ['M0','M2','M4','SPARSI','IRF','WLF']}
        return feature_groups

    def get_feature_list(self):
        """Gets a list of all available features.

        Returns
        ----------
        array_like
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
                        'AR', # note: TODO: AR could probably represent the PACF, not the ACF.
                        'CC',
                        'LD',
                        'MAVFD',
                        'MAVSLP',
                        'MDF',
                        'MNF',
                        'MNP',
                        'MPK',
                        'SKEW',
                        'KURT']
        return feature_list
    
    def extract_feature_group(self, feature_group, windows):
        """Extracts a group of features.

        Parameters
        ----------
        feature_group: string
            The group of features to extract. Valid options include: 'HTD', 'TD4', 'TD9' and 'TDPSD'.
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        dictionary
            A dictionary where each key is a specific feature and its value is a list of the computed 
            features for each window.
        """
        features = {}
        if not feature_group in self.get_feature_groups():
            return features
        return self.extract_features(self.get_feature_groups()[feature_group], windows)

    def extract_features(self, feature_list, windows):
        """Extracts a list of features.

        Parameters
        ----------
        feature_list: list
            The group of features to extract. Run get_feature_list() or checkout the github documentation 
            to find an up-to-date feature list.  
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        dictionary
            A dictionary where each key is a specific feature and its value is a list of the computed 
            features for each window.
        """
        features = {}
        for feature in feature_list:
            if feature in self.get_feature_list():
                method_to_call = getattr(self, 'get' + feature + 'feat')
                features[feature] = method_to_call(windows)
            
        return features

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
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
            The computed features associated with each window. 
        """
        feat = np.mean(np.abs(windows),2)
        return feat
    
    # TODO: Add threshold
    def getZCfeat(self, windows):
        """Extract Zero Crossings (ZC) feature.

        Parameters
        ----------
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
            The computed features associated with each window. 
        """
        sgn_change = np.diff(np.sign(windows),axis=2)
        neg_change = sgn_change == -2
        pos_change = sgn_change ==  2
        feat_a = np.sum(neg_change,2)
        feat_b = np.sum(pos_change,2)
        return feat_a+feat_b
    
    # TODO: Add threshold
    def getSSCfeat(self, windows):
        """Extract Slope Sign Change (SSC) feature.

        Parameters
        ----------
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
            The computed features associated with each window. 
        """
        d_sig = np.diff(windows,axis=2)
        return self.getZCfeat(d_sig)

    def getWLfeat(self, windows):
        """Extract Waveform Length (WL) feature.

        Parameters
        ----------
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
            The computed features associated with each window. 
        """
        feat = np.sum(np.abs(np.diff(windows,axis=2)),2)
        return feat

    def getLSfeat(self, windows):
        """Extract L-Score (LS) feature.

        Parameters
        ----------
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
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
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
            The computed features associated with each window. 
        """
        feat = np.log10(np.sum(np.abs(np.diff(windows, axis=2)),axis=2))
        return feat

    def getMSRfeat(self, windows):
        """Extract Mean Squared Ratio (MSR) feature.

        Parameters
        ----------
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
            The computed features associated with each window. 
        """
        feat = np.abs(np.mean(np.sqrt(windows.astype('complex')),axis=2))
        return feat

    def getWAMPfeat(self, windows, threshold=2e-3): # TODO: add optimization if threshold not passed, need class labels
        """Extract Willison Amplitude (WAMP) feature.

        Parameters
        ----------
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
            The computed features associated with each window. 
        """
        feat = np.sum(np.abs(np.diff(windows, axis=2)) > threshold, axis=2)
        return feat

    def getRMSfeat(self, windows):
        """Extract Root Mean Square (RMS) feature.

        Parameters
        ----------
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
            The computed features associated with each window. 
        """
        feat = np.sqrt(np.mean(np.square(windows),2))
        return feat

    def getIAVfeat(self, windows):
        """Extract Integral of Absolute Value (IAV) feature.

        Parameters
        ----------
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
            The computed features associated with each window. 
        """
        feat = np.sum(np.abs(windows),axis=2)
        return feat

    def getDASDVfeat(self, windows):
        """Difference Absolute Standard Deviation Value (DASDV) feature.

        Parameters
        ----------
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
            The computed features associated with each window. 
        """
        feat = np.abs(np.sqrt(np.mean(np.diff(np.square(windows.astype('complex')),2),2)))
        return feat

    def getVARfeat(self, windows):
        """Extract Variance (VAR) feature.

        Parameters
        ----------
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
            The computed features associated with each window. 
        """
        feat = np.var(windows,axis=2)
        return feat

    def getM0feat(self, windows):
        """Extract First Temporal Moment (M0) feature.

        Parameters
        ----------
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
            The computed features associated with each window. 
        """
        # There are 6 features per channel
        m0 = np.sqrt(np.sum(windows**2,axis=2))
        m0 = m0 ** 0.1 / 0.1
        #Feature extraction goes here
        return np.log(np.abs(m0))
    
    def getM2feat(self, windows):
        """Extract Second Temporal Moment (M2) feature.

        Parameters
        ----------
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
            The computed features associated with each window. 
        """
        # Prepare derivatives for higher order moments
        d1 = np.diff(windows, n=1, axis=2)
        # Root squared 2nd order moments normalized
        m2 = np.sqrt(np.sum(d1 **2, axis=2)/ (windows.shape[2]-1))
        m2 = m2 ** 0.1 / 0.1
        return np.log(np.abs(m2))

    def getM4feat(self, windows):
        """Extract Fourth Temporal Moment (M4) feature.

        Parameters
        ----------
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
            The computed features associated with each window. 
        """
        # Prepare derivatives for higher order moments
        d1 = np.diff(windows, n=1, axis=2)
        d2 = np.diff(d1     , n=1, axis=2)
        # Root squared 4th order moments normalized
        m4 = np.sqrt(np.sum(d2**2,axis=2) / (windows.shape[2]-1))
        m4 = m4 **0.1/0.1
        return np.log(np.abs(m4))
    
    def getSPARSIfeat(self, windows):
        """Extract Sparsness (SPARSI) feature.

        Parameters
        ----------
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
            The computed features associated with each window. 
        """
        m0 = self.getM0feat(windows)
        m2 = self.getM2feat(windows)
        m4 = self.getM4feat(windows)
        sparsi = m0/np.sqrt(np.abs((m0-m2)*(m0-m4)))
        return np.log(np.abs(sparsi))

    def getIRFfeat(self, windows):
        """Extract Irregularity Factor (IRF) feature.

        Parameters
        ----------
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
            The computed features associated with each window. 
        """
        m0 = self.getM0feat(windows)
        m2 = self.getM2feat(windows)
        m4 = self.getM4feat(windows)
        IRF = m2/np.sqrt(np.multiply(m0,m4))
        return np.log(np.abs(IRF))

    def getWLFfeat(self, windows):
        """Waveform Length Factor (WLF) feature.

        Parameters
        ----------
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
            The computed features associated with each window. 
        """
        # Prepare derivatives for higher order moments
        d1 = np.diff(windows, n=1, axis=2)
        d2 = np.diff(d1     , n=1, axis=2)
        # Waveform Length Ratio
        WLR = np.sum( np.abs(d1),axis=2)-np.sum(np.abs(d2),axis=2)
        return np.log(np.abs(WLR))


    def getARfeat(self, windows, order=4):
        """Extract Autoregressive Coefficients (AR) feature.

        Parameters
        ----------
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
            The computed features associated with each window. 
        """
        windows = np.asarray(windows)
        R = np.sum(windows ** 2, axis=2)
        for i in range(1, order + 1):
            r = np.sum(windows[:,:,i:] * windows[:,:,:-i], axis=2)
            R = np.hstack((R,r))
        return R

    def getCCfeat(self, windows, order =4):
        """Extract Cepstral Coefficient (CC) feature.

        Parameters
        ----------
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
            The computed features associated with each window. 
        """
        AR = self.getARfeat(windows, order)
        cc = np.zeros_like(AR)
        cc[:,:self.num_channels] = -1*AR[:,:self.num_channels]
        if order > 2:
            for p in range(2,order+2):
                for l in range(1, p):
                    cc[:,self.num_channels*(p-1):self.num_channels*(p)] = cc[:,self.num_channels*(p-1):self.num_channels*(p)]+(AR[:,self.num_channels*(p-1):self.num_channels*(p)] * cc[:,self.num_channels*(p-2):self.num_channels*(p-1)] * (1-(l/p)))
        return cc
    
    def getLDfeat(self, windows):
        """Extract Log Detector (LD) feature.

        Parameters
        ----------
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
            The computed features associated with each window. 
        """
        return np.exp(np.mean(np.log(np.abs(windows)+1), 2))

    def getMAVFDfeat(self, windows):
        """Extract Mean Absolute Value First Difference (MAVFD) feature.

        Parameters
        ----------
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
            The computed features associated with each window. 
        """
        dwindows = np.diff(windows,axis=2)
        mavfd = np.mean(dwindows,axis=2) / windows.shape[2]
        return mavfd

    def getMAVSLPfeat(self, windows, segment=2):
        """Extract Mean Absolute Value Slope (MAVSLP) feature.

        Parameters
        ----------
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
            The computed features associated with each window. 
        """
        m = int(round(windows.shape[2]/segment))
        mav = []
        mavslp = []
        for i in range(0,segment):
            mav.append(np.mean(np.abs(windows[:,:,i*m:(i+1)*m]), axis=2))
        for i in range (0, segment-1):
            mavslp.append(mav[i+1]- mav[i])
        return np.asarray(mavslp).squeeze()

    def getMDFfeat(self, windows,fs=1000):
        """Extract Median Frequency (MDF) feature.

        Parameters
        ----------
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
            The computed features associated with each window. 
        """
        spec = np.fft.fft(windows,axis=2)
        spec = spec[:,:,0:int(round(spec.shape[2]/2))]
        POW = spec * np.conj(spec)
        totalPOW = np.sum(POW, axis=2)
        cumPOW   = np.cumsum(POW, axis=2)
        medfreq = np.zeros((windows.shape[0], windows.shape[1]))
        for i in range(0, windows.shape[0]):
            for j in range(0, windows.shape[1]):
                medfreq[i,j] = fs*np.argwhere(cumPOW[i,j,:] > totalPOW[i,j] /2)[0]/windows.shape[2]/2
        return medfreq

    def getMNFfeat(self, windows, fs=1000):
        """Extract Mean Frequency (MNF) feature.

        Parameters
        ----------
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
            The computed features associated with each window. 
        """
        spec = np.fft.fft(windows, axis=2)
        f = np.fft.fftfreq(windows.shape[-1])*fs
        spec = spec[:,:,0:int(round(spec.shape[2]/2))]
        f = f[0:int(round(f.shape[0]/2))]
        f = np.repeat(f[np.newaxis, :], spec.shape[0], axis=0)
        f = np.repeat(f[:, np.newaxis,:], spec.shape[1], axis=1)
        POW = spec * np.conj(spec)
        return np.real(np.sum(POW*f,axis=2)/np.sum(POW,axis=2))

    def getMNPfeat(self, windows):
        """Extract Mean Power (MNP) feature.

        Parameters
        ----------
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
            The computed features associated with each window. 
        """
        spec = np.fft.fft(windows,axis=2)
        spec = spec[:,:,0:int(round(spec.shape[0]/2))]
        POW = np.real(spec*np.conj(spec))
        return np.sum(POW, axis=2)/POW.shape[2]

    def getMPKfeat(self, windows):
        """Extract (MPK) feature.

        Parameters
        ----------
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
            The computed features associated with each window. 
        """
        return windows.max(axis=2)

    def getSKEWfeat(self, windows):
        """Extract Skewness (SKEW) feature.

        Parameters
        ----------
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
            The computed features associated with each window. 
        """
        return skew(windows, axis=2)

    def getKURTfeat(self, windows):
        """Extract Kurtosis (KURT) feature.

        Parameters
        ----------
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
            The computed features associated with each window. 
        """
        return kurtosis(windows, axis=2)