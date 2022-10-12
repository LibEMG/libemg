# The MIT License (MIT)
#
# Copyright (c) 2022 Evan Campbell and Ethan Eddy
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.

import numpy as np
from scipy.stats import skew, kurtosis
import sampen

class FeatureExtractor:
    """
    Feature extraction class including feature groups, feature list, and feature extraction code.
    """
    def __init__(self, num_channels):
        self.num_channels = num_channels

    def get_feature_groups(self):
        feature_groups = ['HTD',
                          'TD4',
                          'TD9']
        return feature_groups

    def get_feature_list(self):
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
                        'SAMPEN',
                        'SKEW',
                        'KURT']
        return feature_list

    def extract(self, feature_list, windows):
        features = {}
        for feature in feature_list:
            method_to_call = getattr(self, 'get' + feature + 'feat')
            features[feature] = method_to_call(windows)
            
        return features

    def get_windows(self, data, window_size, window_increment):
        '''
        data is a NxM stream of data with N samples and M channels (numpy array)
        window_size is number of samples in window
        window_increment is number of samples that advances before the next window
        '''
        num_windows = int((data.shape[0]-window_size)/window_increment)
        windows = []
        st_id=0
        ed_id=st_id+window_size
        for w in range(num_windows):
            windows.append(data[st_id:ed_id,:].transpose())
            st_id += window_increment
            ed_id += window_increment
        return windows

    def getMAVfeat(self, windows):
        feat = np.mean(np.abs(windows),2)
        return feat
    
    def getZCfeat(self, windows):
        sgn_change = np.diff(np.sign(windows),axis=2)
        neg_change = sgn_change == -2
        pos_change = sgn_change ==  2
        feat_a = np.sum(neg_change,2)
        feat_b = np.sum(pos_change,2)
        return feat_a+feat_b
    
    def getSSCfeat(self, windows):
        d_sig = np.diff(windows,axis=2)
        return self.getZCfeat(d_sig)

    def getWLfeat(self, windows):
        feat = np.sum(np.abs(np.diff(windows,axis=2)),2)
        return feat

    def getLSfeat(self, windows):
        feat = np.zeros((windows.shape[0],windows.shape[1]))
        for w in range(0, windows.shape[0],1):
            for c in range(0, windows.shape[1],1):
                tmp = self.lmom(np.reshape(windows[w,c,:],(1,windows.shape[2])),2)
                feat[w,c] = tmp[0,1]
        return feat

    def lmom(self, signal, nL):
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
            Coeff = np.concatenate((Spc, self.LegendreShiftPoly(i)))
            l[0,i-1] = np.sum(Coeff * B)
        L = np.concatenate((b0, l),1)

        return L

    def LegendreShiftPoly(self, n):
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
        feat = np.log10(np.sum(np.abs(np.diff(windows, axis=2)),axis=2))
        return feat

    def getMSRfeat(self, windows):
        feat = np.abs(np.mean(np.sqrt(windows.astype('complex')),axis=2))
        return feat

    def getWAMPfeat(self, windows, threshold=2e-3): # TODO: add optimization if threshold not passed, need class labels
        feat = np.sum(np.abs(np.diff(windows, axis=2)) > threshold, axis=2)
        return feat

    def getRMSfeat(self, windows):
        feat = np.sqrt(np.mean(np.square(windows),2))
        return feat

    def getIAVfeat(self, windows):
        feat = np.sum(np.abs(windows),axis=2)
        return feat

    def getDASDVfeat(self, windows):
        feat = np.abs(np.sqrt(np.mean(np.diff(np.square(windows.astype('complex')),2),2)))
        return feat

    def getVARfeat(self, windows):
        feat = np.var(windows,axis=2)
        return feat

    def getM0feat(self, windows):
        # There are 6 features per channel
        m0 = np.sqrt(np.sum(windows**2,axis=2))
        m0 = m0 ** 0.1 / 0.1
        #Feature extraction goes here
        return np.log(np.abs(m0))
    
    def getM2feat(self, windows):
        # Prepare derivatives for higher order moments
        d1 = np.diff(windows, n=1, axis=2)
        # Root squared 2nd order moments normalized
        m2 = np.sqrt(np.sum(d1 **2, axis=2)/ (windows.shape[2]-1))
        m2 = m2 ** 0.1 / 0.1
        return np.log(np.abs(m2))

    def getM4feat(self, windows):
        # Prepare derivatives for higher order moments
        d1 = np.diff(windows, n=1, axis=2)
        d2 = np.diff(d1     , n=1, axis=2)
        # Root squared 4th order moments normalized
        m4 = np.sqrt(np.sum(d2**2,axis=2) / (windows.shape[2]-1))
        m4 = m4 **0.1/0.1
        return np.log(np.abs(m4))
    
    def getSPARSIfeat(self, windows):
        m0 = self.getM0feat(windows)
        m2 = self.getM2feat(windows)
        m4 = self.getM4feat(windows)
        sparsi = m0/np.sqrt(np.abs((m0-m2)*(m0-m4)))
        return np.log(np.abs(sparsi))

    def getIRFfeat(self, windows):
        m0 = self.getM0feat(windows)
        m2 = self.getM2feat(windows)
        m4 = self.getM4feat(windows)
        IRF = m2/np.sqrt(np.multiply(m0,m4))
        return np.log(np.abs(IRF))

    def getWLFfeat(self, windows):
        # Prepare derivatives for higher order moments
        d1 = np.diff(windows, n=1, axis=2)
        d2 = np.diff(d1     , n=1, axis=2)
        # Waveform Length Ratio
        WLR = np.sum( np.abs(d1),axis=2)-np.sum(np.abs(d2),axis=2)
        return np.log(np.abs(WLR))


    def getARfeat(self, windows, order=4):
        windows = np.asarray(windows)
        R = np.sum(windows ** 2, axis=2)
        for i in range(1, order + 1):
            r = np.sum(windows[:,:,i:] * windows[:,:,:-i], axis=2)
            R = np.hstack((R,r))
        return R

    def getCCfeat(self, windows, order =4):
        AR = self.getARfeat(windows, order)
        cc = np.zeros_like(AR)
        cc[:,:self.num_channels] = -1*AR[:,:self.num_channels]
        if order > 2:
            for p in range(2,order+2):
                for l in range(1, p):
                    cc[:,self.num_channels*(p-1):self.num_channels*(p)] = cc[:,self.num_channels*(p-1):self.num_channels*(p)]+(AR[:,self.num_channels*(p-1):self.num_channels*(p)] * cc[:,self.num_channels*(p-2):self.num_channels*(p-1)] * (1-(l/p)))
        return cc
    
    def getLDfeat(self, windows):
        return np.exp(np.mean(np.log(np.abs(windows)+1), 2))

    def getMAVFDfeat(self, windows):
        dwindows = np.diff(windows,axis=2)
        mavfd = np.mean(dwindows,axis=2) / windows.shape[2]
        return mavfd

    def getMAVSLPfeat(self, windows, segment=2):
        m = int(round(windows.shape[2]/segment))
        mav = []
        mavslp = []
        for i in range(0,segment):
            mav.append(np.mean(np.abs(windows[:,:,i*m:(i+1)*m]), axis=2))
        for i in range (0, segment-1):
            mavslp.append(mav[i+1]- mav[i])
        return np.asarray(mavslp).squeeze()

    def getMDFfeat(self, windows,fs=1000):
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
        spec = np.fft.fft(windows, axis=2)
        f = np.fft.fftfreq(windows.shape[-1])*fs
        spec = spec[:,:,0:int(round(spec.shape[2]/2))]
        f = f[0:int(round(f.shape[0]/2))]
        f = np.repeat(f[np.newaxis, :], spec.shape[0], axis=0)
        f = np.repeat(f[:, np.newaxis,:], spec.shape[1], axis=1)
        POW = spec * np.conj(spec)
        return np.real(np.sum(POW*f,axis=2)/np.sum(POW,axis=2))

    def getMNPfeat(self, windows):
        spec = np.fft.fft(windows,axis=2)
        spec = spec[:,:,0:int(round(spec.shape[0]/2))]
        POW = np.real(spec*np.conj(spec))
        return np.sum(POW, axis=2)/POW.shape[2]

    def getMPKfeat(self, windows):
        return windows.max(axis=2)

    def getSAMPENfeat(self, windows, m=2, r_multiply_by_sigma=.2):
        r = r_multiply_by_sigma * np.std(windows, axis=2)
        output = np.zeros((windows.shape[0], windows.shape[1]*(m+1)))
        for w in range(0, windows.shape[0]):
            for c in range(0, windows.shape[1]):
                output[w,c*(m+1):(c+1)*(m+1)] = np.array(sampen.sampen2(data=windows[w,c,:], mm=m, r=r[w,c]))[:,1]
        return output

    def getSKEWfeat(self, windows):
        return skew(windows, axis=2)

    def getKURTfeat(self, windows):
        return kurtosis(windows, axis=2)