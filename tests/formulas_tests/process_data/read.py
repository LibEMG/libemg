import numpy as np

def read_helper(file_path):
        data = np.loadtxt(file_path, delimiter=',')

        signal = data[:,2:]
        labels = data[:,1]

        return signal, labels
    
def read(sets):
    s1,l1 = read_helper(sets["set1"])
    s2,l2 = read_helper(sets["set2"])
    s3,l3 = read_helper(sets["set3"])

    return (s1,l1), (s2,l2), (s3,l3)