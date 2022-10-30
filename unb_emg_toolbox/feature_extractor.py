import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from librosa import lpc

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
    

    def getSSCfeat(self, windows,threshold=0):
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
        w_2 = windows[:,:,2:]
        w_1 = windows[:,:,1:-1]
        w_0 = windows[:,:,:-2]
        con = (((w_1-w_0)*(w_1-w_2)) >= threshold)
        return np.sum(con,axis=2)

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
        feat = np.sqrt(np.mean(np.diff(windows,axis=2)**2,axis=2))
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
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
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
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
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
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
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
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
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
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
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


        # the way they are computed via the matlab script: linear predictive filter

        feature = np.reshape(lpc(windows, order=order,axis=2)[:,:,1:],(windows.shape[0],order*windows.shape[1]),order="C")
        return feature

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
        cc[:,::order] = -1*AR[:,::order]
        if order > 2:
            for p in range(1,order):
                for l in range(1, p):
                    cc[:,p::order] = cc[:,p::order]+(AR[:,p::order] * cc[:,p-l::order] * (1-(l/p)))
                cc[:,p::order] = -1*AR[:,p::order]-cc[:,p::order]
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
        mavfd = np.mean(np.abs(np.diff(windows,axis=2)),axis=2)
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
                medfreq[i,j] = (fs/2)*np.argwhere(cumPOW[i,j,:] > totalPOW[i,j] /2)[0]/(nextpow2/2)
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
        def closure(winsize):
            return 1 if winsize==0 else 2**math.ceil(math.log2(winsize))
        nextpow2 = closure(windows.shape[2])
        spec = np.fft.fft(windows, n=nextpow2,axis=2)/windows.shape[2]
        f = np.fft.fftfreq(nextpow2)*fs
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
        windows: array_like 
            A list of windows - should be computed using the utils.get_windows() function.
        
        Returns
        ----------
        array_like
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
        return kurtosis(windows, axis=2, fisher=False)

    def visualize(self, feature_dic):
        """Visualize a set of features.
        Parameters
        ----------
        feature_dic: dictionary
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
                    lab = "CH"+str(i)
                    ax[index].plot(x, feature_dic[f][:], label=lab)
                    if not lab in labels:
                        labels.append(lab)
                    ax[index].set_ylabel(f)
                index += 1
                fig.suptitle('Features')
                fig.legend(labels, loc='upper right')
        else:
            key = list(feature_dic.keys())[0]
            plt.title(key)
            plt.plot(list(range(0,len(feature_dic[key]))), feature_dic[key])
        plt.show()