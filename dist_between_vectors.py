import numpy as np
import math
from distance import distance_vec

def filter_D(ids1,ids2,D, maxD = (2.5/0.62137119223733)*math.sqrt(2)*1000):
    ind = D <= maxD
    return np.c_[ids1[ind],ids2[ind],D[ind]]

def dist_between(mat1, mat2, maxD = None):
    M = np.zeros(shape=(len(mat1)*len(mat2),6))
    M[:,0:3] = np.repeat(mat1,repeats=len(mat2),axis=0)
    M[:,3:6] = np.tile(mat2,reps=(len(mat1),1))

    D = np.rint(distance_vec(M[:,2],M[:,1],M[:,5],M[:,4]))
    if maxD:
        D = filter_D(M[:,0],M[:,3],D, maxD=maxD)
    else:
        D = filter_D(M[:,0],M[:,3],D)
    return(D)
